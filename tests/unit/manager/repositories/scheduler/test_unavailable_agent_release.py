"""
Multi-agent integration tests for releasing sessions held by an unavailable agent.

Two agents, T and P, share one resource group. A real ``ScheduleCoordinator`` drives the
real ``ExpireStuckStartsLifecycleHandler`` (fence, then release through the real
``SchedulingController`` forced path) and the real ``CleanupForceTerminatedHandler``, on a
real database and a real Valkey. The agent goes away and comes back through the agent
repository's real exit and heartbeat paths, and placement is checked with the scheduler's
real agent query. Only the agent RPC is faked. Time moves by setting ``lost_at`` back,
never by sleeping.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from dateutil.tz import tzutc

from ai.backend.common.clients.valkey_client.valkey_schedule.client import (
    FORCE_TERMINATED_CLEANUP_TTL_SEC,
    ValkeyScheduleClient,
)
from ai.backend.common.identifier.domain import DomainID
from ai.backend.common.identifier.resource_group import ResourceGroupID
from ai.backend.common.typed_validators import HostPortPair as HostPortPairModel
from ai.backend.common.types import AccessKey, AgentId, KernelId, SessionId, ValkeyTarget
from ai.backend.manager.clients.agent import AgentClientPool
from ai.backend.manager.data.agent.types import AgentStatus
from ai.backend.manager.data.kernel.types import KernelStatus
from ai.backend.manager.data.session.types import SessionStatus
from ai.backend.manager.defs import UNAVAILABLE_AGENT_RELEASE_SEC
from ai.backend.manager.models.agent import AgentRow
from ai.backend.manager.models.kernel import KernelRow
from ai.backend.manager.models.resource_slot import ResourceAllocationRow
from ai.backend.manager.models.scheduling_history.row import SessionSchedulingHistoryRow
from ai.backend.manager.models.session import SessionRow
from ai.backend.manager.models.utils import ExtendedAsyncSAEngine
from ai.backend.manager.repositories.agent.db_source.db_source import AgentDBSource
from ai.backend.manager.repositories.agent.updaters import AgentStatusUpdaterSpec
from ai.backend.manager.repositories.base.updater import Updater
from ai.backend.manager.repositories.manager_admin.db_source.db_source import (
    ManagerAdminDBSource,
)
from ai.backend.manager.repositories.scheduler.db_source.db_source import ScheduleDBSource
from ai.backend.manager.repositories.scheduler.repository import SchedulerRepository
from ai.backend.manager.sokovan.scheduler.coordinator import ScheduleCoordinator
from ai.backend.manager.sokovan.scheduler.factory import CoordinatorHandlers
from ai.backend.manager.sokovan.scheduler.handlers.cleanup.force_terminated import (
    CleanupForceTerminatedHandler,
)
from ai.backend.manager.sokovan.scheduler.handlers.maintenance.expire_stuck_starts import (
    ExpireStuckStartsLifecycleHandler,
)
from ai.backend.manager.sokovan.scheduler.scheduler import SchedulerComponents
from ai.backend.manager.sokovan.scheduler.terminator.terminator import (
    SessionTerminator,
    SessionTerminatorArgs,
)
from ai.backend.manager.sokovan.scheduler.types import ScheduleType
from ai.backend.manager.sokovan.scheduling_controller import (
    SchedulingController,
    SchedulingControllerArgs,
)
from ai.backend.manager.types import OptionalState

from .conftest import (
    ResourceGroupRef,
    add_agent,
    create_pending_session_with_kernels,
    delete_force_terminated_cleanup_key,
    fetch_agent_resources,
    fetch_agent_row,
    fetch_schedulable,
    heartbeat_upsert,
    seed_agent_resources,
    set_unavailable_for,
)

HANG_TIMEOUT = "hang-timeout"
RELEASE_SEC = UNAVAILABLE_AGENT_RELEASE_SEC
CPU = Decimal("1")
MEM = Decimal("1024")


# --------------------------------------------------------------------------- fakes


@dataclass
class FakeAgents:
    """The agent RPC: records destroys; an unreachable agent refuses the connection."""

    unreachable: set[AgentId] = field(default_factory=set)
    destroys: list[tuple[AgentId, KernelId, SessionId, str]] = field(default_factory=list)

    def pool(self) -> MagicMock:
        pool = MagicMock(spec=AgentClientPool)

        @asynccontextmanager
        async def acquire(agent_id: AgentId) -> AsyncIterator[Any]:
            if agent_id in self.unreachable:
                raise ConnectionRefusedError(f"agent {agent_id} is unreachable")
            client = MagicMock()

            async def destroy_kernel(
                kernel_id: KernelId, session_id: SessionId, reason: str, **_: Any
            ) -> None:
                self.destroys.append((agent_id, kernel_id, session_id, reason))

            client.destroy_kernel = destroy_kernel
            yield client

        pool.acquire = MagicMock(side_effect=acquire)
        return pool


@dataclass
class Stack:
    db: ExtendedAsyncSAEngine
    valkey: ValkeyScheduleClient
    agents: FakeAgents
    controller: SchedulingController
    event_producer: AsyncMock
    coordinator: ScheduleCoordinator
    group: ResourceGroupRef

    async def watchdog_pass(self) -> None:
        assert await self.coordinator.process_lifecycle_schedule(ScheduleType.EXPIRE_STUCK_STARTS)

    async def cleanup_pass(self) -> None:
        assert await self.coordinator.process_lifecycle_schedule(
            ScheduleType.CLEANUP_FORCE_TERMINATED
        )

    async def queued(self) -> set[SessionId]:
        return set(await self.valkey.get_force_terminated_sessions())

    async def lose(self, agent_id: AgentId, status: AgentStatus, *, ago_sec: float) -> None:
        """The agent repository's exit path, then ``lost_at`` set back by ``ago_sec``."""
        now = datetime.now(tzutc())
        await AgentDBSource(self.db).update_agent_status_exit(
            Updater[AgentRow](
                spec=AgentStatusUpdaterSpec(
                    status=status, status_changed=now, lost_at=OptionalState.update(now)
                ),
                pk_value=agent_id,
            )
        )
        await set_unavailable_for(self.db, agent_id, ago_sec)

    async def heartbeat(self, agent_id: AgentId) -> None:
        await AgentDBSource(self.db).upsert_agent_with_state(
            heartbeat_upsert(agent_id, self.group.name)
        )

    async def placeable(self) -> set[AgentId]:
        """The agents the scheduler would place new sessions on in the group."""
        async with self.db.begin_readonly_session_read_committed() as db_sess:
            agents = await ScheduleDBSource(self.db)._fetch_agents(db_sess, self.group.id)
        return {agent.id for agent in agents}


# --------------------------------------------------------------------------- DB reads


async def _session_status(db: ExtendedAsyncSAEngine, session_id: SessionId) -> tuple[Any, Any]:
    async with db.begin_readonly() as conn:
        row = (
            await conn.execute(
                sa.select(SessionRow.status, SessionRow.status_info).where(
                    SessionRow.id == session_id
                )
            )
        ).one()
    return row.status, row.status_info


async def _kernels(db: ExtendedAsyncSAEngine, session_id: SessionId) -> list[Any]:
    async with db.begin_readonly() as conn:
        return list(
            (
                await conn.execute(
                    sa.select(
                        KernelRow.id, KernelRow.status, KernelRow.status_info, KernelRow.agent
                    ).where(KernelRow.session_id == session_id)
                )
            ).all()
        )


async def _all_freed(db: ExtendedAsyncSAEngine, session_id: SessionId) -> bool:
    async with db.begin_readonly() as conn:
        free_ats = (
            (
                await conn.execute(
                    sa.select(ResourceAllocationRow.free_at)
                    .join(KernelRow, KernelRow.id == ResourceAllocationRow.kernel_id)
                    .where(KernelRow.session_id == session_id)
                )
            )
            .scalars()
            .all()
        )
    return bool(free_ats) and all(free_at is not None for free_at in free_ats)


async def _used(db: ExtendedAsyncSAEngine, agent_id: AgentId) -> dict[str, Decimal]:
    rows = await fetch_agent_resources(db, agent_id)
    return {slot: row.used for slot, row in rows.items()}


async def _force_terminate_history(db: ExtendedAsyncSAEngine, session_id: SessionId) -> list[Any]:
    async with db.begin_readonly() as conn:
        return list(
            (
                await conn.execute(
                    sa.select(
                        SessionSchedulingHistoryRow.to_status, SessionSchedulingHistoryRow.attempts
                    ).where(
                        SessionSchedulingHistoryRow.session_id == session_id,
                        SessionSchedulingHistoryRow.phase == "force_terminate",
                    )
                )
            ).all()
        )


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
async def valkey_schedule(
    redis_container: tuple[str, HostPortPairModel],
) -> AsyncIterator[ValkeyScheduleClient]:
    _, hostport_pair = redis_container
    client = await ValkeyScheduleClient.create(
        valkey_target=ValkeyTarget(addr=hostport_pair.address),
        db_id=0,
        human_readable_name="test-unavailable-agent-release",
    )
    await delete_force_terminated_cleanup_key(client)
    try:
        yield client
    finally:
        await delete_force_terminated_cleanup_key(client)
        await client.close()


@pytest.fixture
def group(
    test_scaling_group_id: ResourceGroupID,
    test_scaling_group_name: str,
) -> ResourceGroupRef:
    return ResourceGroupRef(test_scaling_group_id, test_scaling_group_name)


@pytest.fixture
async def stack(
    db_with_cleanup: ExtendedAsyncSAEngine,
    valkey_schedule: ValkeyScheduleClient,
    group: ResourceGroupRef,
    resource_slot_types: None,
) -> Stack:
    agents = FakeAgents()
    repository = SchedulerRepository(
        db_with_cleanup,
        valkey_stat=MagicMock(),
        valkey_schedule=valkey_schedule,
        config_provider=MagicMock(),
        storage_manager=MagicMock(),
    )
    event_producer = AsyncMock()
    controller = SchedulingController(
        SchedulingControllerArgs(
            repository=repository,
            config_provider=MagicMock(),
            storage_manager=MagicMock(),
            event_producer=event_producer,
            valkey_schedule=valkey_schedule,
            network_plugin_ctx=MagicMock(),
            hook_plugin_ctx=MagicMock(),
            agent_selector=MagicMock(),
        )
    )
    terminator = SessionTerminator(
        SessionTerminatorArgs(
            repository=repository,
            agent_client_pool=agents.pool(),
            valkey_schedule=valkey_schedule,
        )
    )
    coordinator = ScheduleCoordinator(
        valkey_schedule=valkey_schedule,
        components=SchedulerComponents(
            provisioner=MagicMock(),
            launcher=MagicMock(),
            terminator=terminator,
            repository=repository,
            config_provider=MagicMock(),
            hook_registry=MagicMock(),
        ),
        handlers=CoordinatorHandlers(
            lifecycle_handlers={
                ScheduleType.EXPIRE_STUCK_STARTS: ExpireStuckStartsLifecycleHandler(
                    repository, controller
                ),
            },
            promotion_specs={},
            kernel_handlers={},
            kernel_observers={},
            cleanup_handlers={
                ScheduleType.CLEANUP_FORCE_TERMINATED: CleanupForceTerminatedHandler(
                    terminator=terminator,
                    repository=repository,
                    valkey_schedule=valkey_schedule,
                ),
            },
        ),
        scheduling_controller=controller,
        event_producer=AsyncMock(),
        lock_factory=MagicMock(),
    )
    return Stack(
        db=db_with_cleanup,
        valkey=valkey_schedule,
        agents=agents,
        controller=controller,
        event_producer=event_producer,
        coordinator=coordinator,
        group=group,
    )


@dataclass
class Sessions:
    db: ExtendedAsyncSAEngine
    kwargs: dict[str, Any]

    async def create(
        self,
        agent_ids: list[AgentId],
        *,
        status: SessionStatus,
    ) -> SessionId:
        """A session with one RUNNING-usage kernel per agent; TERMINATING kernels for a
        TERMINATING session."""
        session_id, _ = await create_pending_session_with_kernels(
            self.db,
            agent_assignments=[(agent_id, CPU, MEM) for agent_id in agent_ids],
            session_status=status,
            kernel_status=KernelStatus.RUNNING,
            assign_agents=True,
            **self.kwargs,
        )
        if status == SessionStatus.TERMINATING:
            async with self.db.begin() as conn:
                await conn.execute(
                    sa.update(KernelRow)
                    .where(KernelRow.session_id == session_id)
                    .values(status=KernelStatus.TERMINATING)
                )
        return session_id


@pytest.fixture
def sessions(
    db_with_cleanup: ExtendedAsyncSAEngine,
    test_domain_id: DomainID,
    test_domain_name: str,
    group: ResourceGroupRef,
    test_group_id: uuid.UUID,
    test_user_uuid: uuid.UUID,
    test_access_key: AccessKey,
) -> Sessions:
    return Sessions(
        db_with_cleanup,
        {
            "domain_id": test_domain_id,
            "domain_name": test_domain_name,
            "resource_group_id": group.id,
            "scaling_group_name": group.name,
            "group_id": test_group_id,
            "user_uuid": test_user_uuid,
            "access_key": test_access_key,
        },
    )


@dataclass
class Agents:
    t: AgentId
    p: AgentId


@pytest.fixture
async def two_agents(stack: Stack) -> Agents:
    """T and P, both ALIVE, with their capacity in agent_resources."""
    t = await add_agent(stack.db, stack.group, AgentStatus.ALIVE)
    p = await add_agent(stack.db, stack.group, AgentStatus.ALIVE)
    return Agents(t, p)


async def _seed_usage(db: ExtendedAsyncSAEngine, agent_id: AgentId, kernel_count: int) -> None:
    await seed_agent_resources(
        db,
        agent_id,
        cpu_capacity=Decimal("10"),
        mem_capacity=Decimal("10240"),
        cpu_used=CPU * kernel_count,
        mem_used=MEM * kernel_count,
    )


def _usage(kernel_count: int) -> dict[str, Decimal]:
    return {"cpu": CPU * kernel_count, "mem": MEM * kernel_count}


# --------------------------------------------------------------------------- tests


class TestUnavailableAgentRelease:
    async def test_crashed_agent_is_fenced_its_session_released_and_cleaned_up_on_return(
        self,
        stack: Stack,
        two_agents: Agents,
        sessions: Sessions,
    ) -> None:
        t, p = two_agents.t, two_agents.p
        held = await sessions.create([t], status=SessionStatus.TERMINATING)
        running = await sessions.create([t], status=SessionStatus.RUNNING)
        on_p = await sessions.create([p], status=SessionStatus.TERMINATING)
        await _seed_usage(stack.db, t, 2)
        await _seed_usage(stack.db, p, 1)
        stack.agents.unreachable.add(t)
        assert await stack.placeable() == {t, p}

        # T crashes: LOST, but not for long enough yet.
        await stack.lose(t, AgentStatus.LOST, ago_sec=RELEASE_SEC - 10)
        await stack.watchdog_pass()
        assert await _session_status(stack.db, held) == (SessionStatus.TERMINATING, "test")
        assert await fetch_schedulable(stack.db, t) is True

        # Past the bound: T is fenced, then the session held on it is released.
        await set_unavailable_for(stack.db, t, RELEASE_SEC + 1)
        await stack.watchdog_pass()
        assert await fetch_schedulable(stack.db, t) is False
        assert await _session_status(stack.db, held) == (SessionStatus.TERMINATED, HANG_TIMEOUT)
        assert [(k.status, k.status_info, k.agent) for k in await _kernels(stack.db, held)] == [
            (KernelStatus.TERMINATED, HANG_TIMEOUT, t)
        ]
        assert await _all_freed(stack.db, held)
        # T's usage is back to what the RUNNING session alone holds.
        assert await _used(stack.db, t) == _usage(1)
        assert await stack.queued() == {held}
        # Everything else is untouched: P, its session, and the RUNNING session on T.
        assert await fetch_schedulable(stack.db, p) is True
        assert await _session_status(stack.db, on_p) == (SessionStatus.TERMINATING, "test")
        assert await _used(stack.db, p) == _usage(1)
        assert (await _session_status(stack.db, running))[0] == SessionStatus.RUNNING
        assert {k.status for k in await _kernels(stack.db, running)} == {KernelStatus.RUNNING}
        assert await stack.placeable() == {p}

        # A second pass changes nothing.
        await stack.watchdog_pass()
        assert len(await _force_terminate_history(stack.db, held)) == 1

        # The cleanup cannot reach T: the destroy fails and the id stays queued.
        await stack.cleanup_pass()
        assert stack.agents.destroys == []
        assert await stack.queued() == {held}

        # The queue expired during the outage: the cleanup queues the id again.
        await delete_force_terminated_cleanup_key(stack.valkey)
        await stack.cleanup_pass()
        assert await stack.queued() == {held}

        # T comes back: ALIVE, but still fenced and never offered for placement.
        await stack.heartbeat(t)
        row = await fetch_agent_row(stack.db, t)
        assert (row["status"], row["lost_at"], row["schedulable"]) == (
            AgentStatus.ALIVE,
            None,
            False,
        )
        assert await stack.placeable() == {p}
        # The watchdog does not touch an ALIVE agent's sessions.
        await stack.watchdog_pass()
        assert await _session_status(stack.db, on_p) == (SessionStatus.TERMINATING, "test")

        # The destroy now reaches T, with the release's reason; the id leaves the queue.
        stack.agents.unreachable.discard(t)
        await stack.cleanup_pass()
        (kernel,) = await _kernels(stack.db, held)
        assert (t, kernel.id, held, HANG_TIMEOUT) in stack.agents.destroys
        assert await stack.queued() == set()
        # With T back, nothing is queued again.
        destroys = len(stack.agents.destroys)
        await stack.cleanup_pass()
        assert await stack.queued() == set()
        assert len(stack.agents.destroys) == destroys

        # Only an operator re-enables T.
        await ManagerAdminDBSource(stack.db).update_agent_schedulable([t], True)
        assert await stack.placeable() == {t, p}

    async def test_cleanly_stopped_agent_is_released_and_its_sessions_requeued(
        self,
        stack: Stack,
        two_agents: Agents,
        sessions: Sessions,
    ) -> None:
        t = two_agents.t
        held = await sessions.create([t], status=SessionStatus.TERMINATING)
        await _seed_usage(stack.db, t, 1)
        stack.agents.unreachable.add(t)

        await stack.lose(t, AgentStatus.TERMINATED, ago_sec=RELEASE_SEC)
        await stack.watchdog_pass()

        assert await fetch_schedulable(stack.db, t) is False
        assert await _session_status(stack.db, held) == (SessionStatus.TERMINATED, HANG_TIMEOUT)
        assert await _used(stack.db, t) == _usage(0)
        await delete_force_terminated_cleanup_key(stack.valkey)
        await stack.cleanup_pass()
        assert await stack.queued() == {held}
        ttl = await _cleanup_ttl(stack.valkey)
        assert ttl > FORCE_TERMINATED_CLEANUP_TTL_SEC - 10

    async def test_a_whole_group_lost_releases_nothing(
        self,
        stack: Stack,
        two_agents: Agents,
        sessions: Sessions,
    ) -> None:
        t, p = two_agents.t, two_agents.p
        on_t = await sessions.create([t], status=SessionStatus.TERMINATING)
        on_p = await sessions.create([p], status=SessionStatus.TERMINATING)
        await _seed_usage(stack.db, t, 1)
        await _seed_usage(stack.db, p, 1)

        await stack.lose(t, AgentStatus.LOST, ago_sec=RELEASE_SEC + 60)
        await stack.lose(p, AgentStatus.LOST, ago_sec=RELEASE_SEC + 60)
        await stack.watchdog_pass()

        for agent_id, session_id in [(t, on_t), (p, on_p)]:
            assert await fetch_schedulable(stack.db, agent_id) is True
            assert await _session_status(stack.db, session_id) == (
                SessionStatus.TERMINATING,
                "test",
            )
            assert await _used(stack.db, agent_id) == _usage(1)
        assert await stack.queued() == set()

    async def test_a_session_on_two_agents_is_neither_fenced_nor_released(
        self,
        stack: Stack,
        two_agents: Agents,
        sessions: Sessions,
    ) -> None:
        t, p = two_agents.t, two_agents.p
        spanning = await sessions.create([t, p], status=SessionStatus.TERMINATING)
        await _seed_usage(stack.db, t, 1)
        await _seed_usage(stack.db, p, 1)
        # A third agent keeps the group's ALIVE guard satisfied.
        await add_agent(stack.db, stack.group, AgentStatus.ALIVE)

        await stack.lose(t, AgentStatus.LOST, ago_sec=RELEASE_SEC + 60)
        await stack.watchdog_pass()

        assert await fetch_schedulable(stack.db, t) is True
        assert await _session_status(stack.db, spanning) == (SessionStatus.TERMINATING, "test")
        assert await _used(stack.db, t) == _usage(1)
        assert await stack.queued() == set()

    async def test_a_failing_release_leaves_the_agent_fenced_and_the_next_pass_releases(
        self,
        stack: Stack,
        two_agents: Agents,
        sessions: Sessions,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        t = two_agents.t
        held = await sessions.create([t], status=SessionStatus.TERMINATING)
        await _seed_usage(stack.db, t, 1)
        await stack.lose(t, AgentStatus.LOST, ago_sec=RELEASE_SEC + 60)
        real_release = stack.controller.mark_sessions_for_termination
        monkeypatch.setattr(
            stack.controller,
            "mark_sessions_for_termination",
            AsyncMock(side_effect=RuntimeError("release failed")),
        )

        await stack.watchdog_pass()

        # The fence came first and stays; nothing was released.
        assert await fetch_schedulable(stack.db, t) is False
        assert await _session_status(stack.db, held) == (SessionStatus.TERMINATING, "test")
        assert await _used(stack.db, t) == _usage(1)

        monkeypatch.setattr(stack.controller, "mark_sessions_for_termination", real_release)
        await stack.watchdog_pass()
        assert await _session_status(stack.db, held) == (SessionStatus.TERMINATED, HANG_TIMEOUT)
        assert await _used(stack.db, t) == _usage(0)

    async def test_two_concurrent_passes_release_once(
        self,
        stack: Stack,
        two_agents: Agents,
        sessions: Sessions,
    ) -> None:
        t = two_agents.t
        held = await sessions.create([t], status=SessionStatus.TERMINATING)
        await _seed_usage(stack.db, t, 1)
        await stack.lose(t, AgentStatus.LOST, ago_sec=RELEASE_SEC + 60)

        await asyncio.gather(stack.watchdog_pass(), stack.watchdog_pass())

        assert await _session_status(stack.db, held) == (SessionStatus.TERMINATED, HANG_TIMEOUT)
        assert await _force_terminate_history(stack.db, held) == [
            (SessionStatus.TERMINATED.name, 1)
        ]
        assert await _used(stack.db, t) == _usage(0)
        broadcasts = [
            event
            for call in stack.event_producer.broadcast_events_batch.await_args_list
            for event in call.args[0]
        ]
        assert [(e.session_id, e.reason) for e in broadcasts] == [(held, HANG_TIMEOUT)]


async def _cleanup_ttl(client: ValkeyScheduleClient) -> int:
    async with client._client.client() as conn:
        return await conn.ttl(client._get_force_terminated_cleanup_key())
