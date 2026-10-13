"""
Tests for fencing unavailable agents and for keeping their released sessions queued
for cleanup.

``ScheduleDBSource.fence_unavailable_agents`` makes an agent that has been LOST or
TERMINATED for long enough unschedulable, but only while another agent of its
resource group is ALIVE. It must be a compare-and-set against the agent heartbeat:
whichever of the two commits first wins, and a heartbeat that comes after the fence
never makes the agent schedulable again.

``ScheduleDBSource.get_sessions_terminated_on_fenced_agents`` finds the sessions
released from such an agent while it stays away, and
``CleanupForceTerminatedHandler.fetch_session_ids`` queues them for cleanup again.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from dateutil.tz import tzutc

from ai.backend.common.auth import PublicKey
from ai.backend.common.clients.valkey_client.valkey_schedule.client import (
    FORCE_TERMINATED_CLEANUP_TTL_SEC,
    ValkeyScheduleClient,
)
from ai.backend.common.data.agent.types import AgentInfo
from ai.backend.common.events.event_types.kernel.types import KernelLifecycleEventReason
from ai.backend.common.identifier.domain import DomainID
from ai.backend.common.identifier.resource_group import ResourceGroupID
from ai.backend.common.typed_validators import HostPortPair as HostPortPairModel
from ai.backend.common.types import (
    AccessKey,
    AgentId,
    DeviceName,
    ResourceSlot,
    SessionId,
    SlotName,
    SlotTypes,
    ValkeyTarget,
)
from ai.backend.manager.data.agent.types import AgentHeartbeatUpsert, AgentStatus
from ai.backend.manager.data.kernel.types import KernelStatus
from ai.backend.manager.data.session.types import SessionStatus
from ai.backend.manager.defs import (
    UNAVAILABLE_AGENT_CLEANUP_RETRY_SEC,
    UNAVAILABLE_AGENT_RELEASE_SEC,
)
from ai.backend.manager.models.agent import AgentRow
from ai.backend.manager.models.kernel import KernelRow
from ai.backend.manager.models.scaling_group import ScalingGroupOpts, ScalingGroupRow
from ai.backend.manager.models.utils import ExtendedAsyncSAEngine
from ai.backend.manager.repositories.agent.db_source.db_source import AgentDBSource
from ai.backend.manager.repositories.agent.updaters import AgentStatusUpdaterSpec
from ai.backend.manager.repositories.base.updater import Updater
from ai.backend.manager.repositories.scheduler.db_source.db_source import ScheduleDBSource
from ai.backend.manager.sokovan.scheduler.handlers.cleanup.force_terminated import (
    CleanupForceTerminatedHandler,
)
from ai.backend.manager.types import OptionalState
from ai.backend.manager.views.sokovan.agent import FencedAgent

from .conftest import create_pending_session_with_kernels

RELEASE_SEC = UNAVAILABLE_AGENT_RELEASE_SEC
LOCK_WAIT_TIMEOUT_SEC = 10.0


def test_release_bound_is_five_minutes() -> None:
    assert UNAVAILABLE_AGENT_RELEASE_SEC == 300


def test_released_sessions_stay_queued_for_a_week() -> None:
    assert UNAVAILABLE_AGENT_CLEANUP_RETRY_SEC == 7 * 24 * 60 * 60


def test_heartbeat_never_writes_schedulable() -> None:
    upsert = _heartbeat(AgentId("any-agent"), "any-group")
    assert "schedulable" not in upsert.insert_fields
    assert "schedulable" not in upsert.update_fields


@dataclass(frozen=True)
class _Group:
    id: ResourceGroupID
    name: str


async def _add_group(db: ExtendedAsyncSAEngine) -> _Group:
    group = _Group(ResourceGroupID(uuid.uuid4()), f"test-sgroup-{uuid.uuid4().hex[:8]}")
    async with db.begin_session() as db_sess:
        db_sess.add(
            ScalingGroupRow(
                id=group.id,
                name=group.name,
                driver="static",
                scheduler="fifo",
                scheduler_opts=ScalingGroupOpts(
                    allowed_session_types=[],
                    pending_timeout=timedelta(hours=1),
                    config={},
                ),
                driver_opts={},
                is_active=True,
            )
        )
    return group


async def _add_agent(
    db: ExtendedAsyncSAEngine,
    group: _Group,
    status: AgentStatus,
    *,
    unavailable_for_sec: float | None = None,
    schedulable: bool = True,
) -> AgentId:
    """Add an agent; ``unavailable_for_sec`` backdates ``lost_at`` on the database clock."""
    agent_id = AgentId(f"test-agent-{uuid.uuid4().hex[:8]}")
    async with db.begin_session() as db_sess:
        db_sess.add(
            AgentRow(
                id=agent_id,
                status=status,
                region="local",
                scaling_group=group.name,
                resource_group_id=group.id,
                schedulable=schedulable,
                available_slots=ResourceSlot({"cpu": Decimal("10"), "mem": Decimal("10240")}),
                occupied_slots=ResourceSlot(),
                addr="127.0.0.1:6001",
                version="1.0.0",
                architecture="x86_64",
            )
        )
    if unavailable_for_sec is not None:
        await _set_unavailable_for(db, agent_id, unavailable_for_sec)
    return agent_id


async def _set_unavailable_for(
    db: ExtendedAsyncSAEngine, agent_id: AgentId, seconds: float
) -> None:
    async with db.begin() as conn:
        await conn.execute(
            sa.update(AgentRow)
            .where(AgentRow.id == agent_id)
            .values(lost_at=sa.func.now() - sa.literal(timedelta(seconds=seconds), sa.Interval))
        )


async def _agent_row(db: ExtendedAsyncSAEngine, agent_id: AgentId) -> dict[str, Any]:
    async with db.begin_readonly() as conn:
        row = (
            await conn.execute(sa.select(AgentRow.__table__).where(AgentRow.id == agent_id))
        ).one()
    return dict(row._mapping)


async def _schedulable(db: ExtendedAsyncSAEngine, agent_id: AgentId) -> bool:
    return bool((await _agent_row(db, agent_id))["schedulable"])


def _heartbeat(agent_id: AgentId, group_name: str) -> AgentHeartbeatUpsert:
    return AgentHeartbeatUpsert.from_agent_info(
        agent_id=agent_id,
        agent_info=AgentInfo(
            ip="127.0.0.1",
            version="1.0.0",
            scaling_group=group_name,
            available_resource_slots=ResourceSlot({
                SlotName("cpu"): "10",
                SlotName("mem"): "10240",
            }),
            slot_key_and_units={
                SlotName("cpu"): SlotTypes.COUNT,
                SlotName("mem"): SlotTypes.BYTES,
            },
            compute_plugins={DeviceName("cpu"): {}},
            addr="127.0.0.1:6001",
            public_key=PublicKey(b"test-public-key"),
            public_host="127.0.0.1",
            images=b"\x82\xc4\x00\x00",
            region="local",
            architecture="x86_64",
            auto_terminate_abusing_kernel=False,
        ),
        heartbeat_received=datetime.now(tzutc()),
    )


class _PausingDB:
    """Delegates to the engine, but holds one transaction open after its body.

    The wrapped method yields as usual; when the caller's body is done (its
    statements ran and its row locks are held), it signals ``body_done`` and waits
    for ``release`` before the transaction commits. Everything else goes to the
    real engine, so the code under test runs unchanged.
    """

    def __init__(self, db: ExtendedAsyncSAEngine, method: str) -> None:
        self._db = db
        self._method = method
        self.body_done = asyncio.Event()
        self.release = asyncio.Event()

    def __getattr__(self, name: str) -> Any:
        real = getattr(self._db, name)
        if name != self._method:
            return real
        return self._pausing(real)

    def _pausing(
        self, real: Callable[[], AbstractAsyncContextManager[Any]]
    ) -> Callable[[], AbstractAsyncContextManager[Any]]:
        @asynccontextmanager
        async def wrapper() -> AsyncIterator[Any]:
            async with real() as conn:
                yield conn
                self.body_done.set()
                await self.release.wait()

        return wrapper


async def _wait_for_lock_waiter(db: ExtendedAsyncSAEngine, query_fragment: str) -> None:
    """Wait until a backend running ``query_fragment`` is blocked on a lock."""
    deadline = asyncio.get_running_loop().time() + LOCK_WAIT_TIMEOUT_SEC
    while True:
        async with db.begin_readonly() as conn:
            waiting = (
                await conn.execute(
                    sa.text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE wait_event_type = 'Lock' AND query ILIKE :fragment"
                    ),
                    {"fragment": f"%{query_fragment}%"},
                )
            ).scalar_one()
        if waiting:
            return
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"no backend running {query_fragment!r} waited on a lock")
        await asyncio.sleep(0.05)


class TestFenceUnavailableAgents:
    @pytest.fixture
    def db_source(self, db_with_cleanup: ExtendedAsyncSAEngine) -> ScheduleDBSource:
        return ScheduleDBSource(db_with_cleanup)

    @pytest.fixture
    async def group(self, db_with_cleanup: ExtendedAsyncSAEngine) -> _Group:
        return await _add_group(db_with_cleanup)

    @pytest.fixture
    async def alive_peer(self, db_with_cleanup: ExtendedAsyncSAEngine, group: _Group) -> AgentId:
        return await _add_agent(db_with_cleanup, group, AgentStatus.ALIVE)

    @pytest.mark.parametrize(
        ("status", "unavailable_for_sec"),
        [
            (AgentStatus.LOST, RELEASE_SEC + 1),
            (AgentStatus.TERMINATED, RELEASE_SEC),
            (AgentStatus.TERMINATED, RELEASE_SEC + 1),
            (AgentStatus.LOST, 86400),
        ],
    )
    async def test_fences_an_agent_unavailable_long_enough(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        group: _Group,
        alive_peer: AgentId,
        status: AgentStatus,
        unavailable_for_sec: float,
    ) -> None:
        target = await _add_agent(
            db_with_cleanup, group, status, unavailable_for_sec=unavailable_for_sec
        )
        before = await _agent_row(db_with_cleanup, target)

        fenced = await db_source.fence_unavailable_agents([target], group.id, RELEASE_SEC)

        assert fenced == [FencedAgent(agent_id=target, status=status, lost_at=before["lost_at"])]
        after = await _agent_row(db_with_cleanup, target)
        assert after["schedulable"] is False
        # Nothing but schedulable changes.
        assert {k: v for k, v in after.items() if k != "schedulable"} == {
            k: v for k, v in before.items() if k != "schedulable"
        }
        # The ALIVE peer is untouched.
        assert await _schedulable(db_with_cleanup, alive_peer) is True

    @pytest.mark.parametrize(
        ("status", "unavailable_for_sec"),
        [
            (AgentStatus.LOST, RELEASE_SEC - 1),
            (AgentStatus.TERMINATED, RELEASE_SEC - 1),
            (AgentStatus.LOST, None),
            (AgentStatus.TERMINATED, None),
            (AgentStatus.ALIVE, None),
            (AgentStatus.ALIVE, RELEASE_SEC + 1),
            (AgentStatus.RESTARTING, None),
            (AgentStatus.RESTARTING, RELEASE_SEC + 1),
        ],
    )
    async def test_does_not_fence_an_agent_not_unavailable_long_enough(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        group: _Group,
        alive_peer: AgentId,
        status: AgentStatus,
        unavailable_for_sec: float | None,
    ) -> None:
        target = await _add_agent(
            db_with_cleanup, group, status, unavailable_for_sec=unavailable_for_sec
        )
        before = await _agent_row(db_with_cleanup, target)

        assert await db_source.fence_unavailable_agents([target], group.id, RELEASE_SEC) == []
        assert await _agent_row(db_with_cleanup, target) == before

    async def test_does_not_fence_without_an_alive_agent_in_the_group(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        group: _Group,
    ) -> None:
        other_group = await _add_group(db_with_cleanup)
        # Peers that are present but not ALIVE do not count.
        await _add_agent(db_with_cleanup, group, AgentStatus.RESTARTING)
        await _add_agent(db_with_cleanup, group, AgentStatus.LOST, unavailable_for_sec=10)
        # An ALIVE agent in another group does not count either.
        await _add_agent(db_with_cleanup, other_group, AgentStatus.ALIVE)
        target = await _add_agent(
            db_with_cleanup, group, AgentStatus.LOST, unavailable_for_sec=RELEASE_SEC + 60
        )

        assert await db_source.fence_unavailable_agents([target], group.id, RELEASE_SEC) == []
        assert await _schedulable(db_with_cleanup, target) is True

    async def test_a_whole_group_lost_at_once_fences_nothing(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        group: _Group,
    ) -> None:
        agents = [
            await _add_agent(
                db_with_cleanup, group, AgentStatus.LOST, unavailable_for_sec=RELEASE_SEC + 60
            )
            for _ in range(2)
        ]

        assert await db_source.fence_unavailable_agents(agents, group.id, RELEASE_SEC) == []
        for agent_id in agents:
            assert await _schedulable(db_with_cleanup, agent_id) is True

    async def test_does_not_fence_an_agent_of_another_group(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        group: _Group,
        alive_peer: AgentId,
    ) -> None:
        other_group = await _add_group(db_with_cleanup)
        await _add_agent(db_with_cleanup, other_group, AgentStatus.ALIVE)
        target = await _add_agent(
            db_with_cleanup, other_group, AgentStatus.LOST, unavailable_for_sec=RELEASE_SEC + 60
        )

        # The caller names the wrong group: nothing is fenced, in either group.
        assert await db_source.fence_unavailable_agents([target], group.id, RELEASE_SEC) == []
        assert await _schedulable(db_with_cleanup, target) is True

    async def test_fences_only_the_qualifying_agents_of_a_batch(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        group: _Group,
        alive_peer: AgentId,
    ) -> None:
        old = await _add_agent(
            db_with_cleanup, group, AgentStatus.LOST, unavailable_for_sec=RELEASE_SEC + 60
        )
        recent = await _add_agent(db_with_cleanup, group, AgentStatus.LOST, unavailable_for_sec=10)

        fenced = await db_source.fence_unavailable_agents(
            [old, recent, alive_peer, AgentId("no-such-agent")], group.id, RELEASE_SEC
        )

        assert [agent.agent_id for agent in fenced] == [old]
        assert await _schedulable(db_with_cleanup, old) is False
        assert await _schedulable(db_with_cleanup, recent) is True
        assert await _schedulable(db_with_cleanup, alive_peer) is True

    async def test_an_already_fenced_agent_is_returned_again(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        group: _Group,
        alive_peer: AgentId,
    ) -> None:
        target = await _add_agent(
            db_with_cleanup,
            group,
            AgentStatus.LOST,
            unavailable_for_sec=RELEASE_SEC + 60,
            schedulable=False,
        )
        before = await _agent_row(db_with_cleanup, target)

        first = await db_source.fence_unavailable_agents([target], group.id, RELEASE_SEC)
        second = await db_source.fence_unavailable_agents([target], group.id, RELEASE_SEC)

        assert [agent.agent_id for agent in first] == [target]
        assert second == first
        assert await _agent_row(db_with_cleanup, target) == before

    async def test_no_agents_is_a_no_op(
        self,
        db_source: ScheduleDBSource,
        group: _Group,
    ) -> None:
        assert await db_source.fence_unavailable_agents([], group.id, RELEASE_SEC) == []

    async def test_agent_lost_through_the_exit_path_is_fenced(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        group: _Group,
        alive_peer: AgentId,
    ) -> None:
        """The agent repository's real exit path sets the lost_at the fence reads."""
        target = await _add_agent(db_with_cleanup, group, AgentStatus.ALIVE)
        now = datetime.now(tzutc())
        await AgentDBSource(db_with_cleanup).update_agent_status_exit(
            Updater[AgentRow](
                spec=AgentStatusUpdaterSpec(
                    status=AgentStatus.LOST,
                    status_changed=now,
                    lost_at=OptionalState.update(now),
                ),
                pk_value=target,
            )
        )
        row = await _agent_row(db_with_cleanup, target)
        assert row["status"] == AgentStatus.LOST
        assert row["lost_at"] is not None
        # Just lost: not yet.
        assert await db_source.fence_unavailable_agents([target], group.id, RELEASE_SEC) == []

        await _set_unavailable_for(db_with_cleanup, target, RELEASE_SEC + 1)
        fenced = await db_source.fence_unavailable_agents([target], group.id, RELEASE_SEC)
        assert [agent.agent_id for agent in fenced] == [target]


class TestFenceAgainstHeartbeat:
    """The fence and the heartbeat both lock the agent row; the first to commit wins."""

    @pytest.fixture
    async def group(self, db_with_cleanup: ExtendedAsyncSAEngine) -> _Group:
        return await _add_group(db_with_cleanup)

    @pytest.fixture
    async def alive_peer(self, db_with_cleanup: ExtendedAsyncSAEngine, group: _Group) -> AgentId:
        return await _add_agent(db_with_cleanup, group, AgentStatus.ALIVE)

    @pytest.fixture
    async def lost_agent(self, db_with_cleanup: ExtendedAsyncSAEngine, group: _Group) -> AgentId:
        return await _add_agent(
            db_with_cleanup, group, AgentStatus.LOST, unavailable_for_sec=RELEASE_SEC + 60
        )

    async def test_heartbeat_first_the_fence_waits_and_fences_nothing(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        group: _Group,
        alive_peer: AgentId,
        lost_agent: AgentId,
    ) -> None:
        heartbeat_db = _PausingDB(db_with_cleanup, "begin_session_read_committed")
        heartbeat = asyncio.create_task(
            AgentDBSource(heartbeat_db).upsert_agent_with_state(  # type: ignore[arg-type]
                _heartbeat(lost_agent, group.name)
            )
        )
        try:
            await asyncio.wait_for(heartbeat_db.body_done.wait(), LOCK_WAIT_TIMEOUT_SEC)
            fence = asyncio.create_task(
                ScheduleDBSource(db_with_cleanup).fence_unavailable_agents(
                    [lost_agent], group.id, RELEASE_SEC
                )
            )
            await _wait_for_lock_waiter(db_with_cleanup, "UPDATE agents SET schedulable")
            assert not fence.done()
        finally:
            heartbeat_db.release.set()
        result = await heartbeat
        assert result.was_revived is True

        assert await fence == []
        row = await _agent_row(db_with_cleanup, lost_agent)
        assert row["status"] == AgentStatus.ALIVE
        assert row["lost_at"] is None
        assert row["schedulable"] is True

    async def test_fence_first_the_heartbeat_waits_and_keeps_the_fence(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        group: _Group,
        alive_peer: AgentId,
        lost_agent: AgentId,
    ) -> None:
        fence_db = _PausingDB(db_with_cleanup, "begin_read_committed")
        fence = asyncio.create_task(
            ScheduleDBSource(fence_db).fence_unavailable_agents(  # type: ignore[arg-type]
                [lost_agent], group.id, RELEASE_SEC
            )
        )
        try:
            await asyncio.wait_for(fence_db.body_done.wait(), LOCK_WAIT_TIMEOUT_SEC)
            heartbeat = asyncio.create_task(
                AgentDBSource(db_with_cleanup).upsert_agent_with_state(
                    _heartbeat(lost_agent, group.name)
                )
            )
            await _wait_for_lock_waiter(db_with_cleanup, "FOR UPDATE")
            assert not heartbeat.done()
        finally:
            fence_db.release.set()
        assert [agent.agent_id for agent in await fence] == [lost_agent]

        result = await heartbeat
        assert result.was_revived is True
        row = await _agent_row(db_with_cleanup, lost_agent)
        assert row["status"] == AgentStatus.ALIVE
        assert row["lost_at"] is None
        assert row["schedulable"] is False

    async def test_heartbeats_and_restarts_keep_the_fence(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        group: _Group,
        alive_peer: AgentId,
        lost_agent: AgentId,
    ) -> None:
        fenced = await ScheduleDBSource(db_with_cleanup).fence_unavailable_agents(
            [lost_agent], group.id, RELEASE_SEC
        )
        assert [agent.agent_id for agent in fenced] == [lost_agent]
        agent_db_source = AgentDBSource(db_with_cleanup)

        # The heartbeat's update path.
        await agent_db_source.upsert_agent_with_state(_heartbeat(lost_agent, group.name))
        assert await _schedulable(db_with_cleanup, lost_agent) is False

        # The heartbeat's insert path, on a conflict with the existing row.
        async with db_with_cleanup.begin_session_read_committed() as db_sess:
            await agent_db_source._insert_new_agent(db_sess, _heartbeat(lost_agent, group.name))
        row = await _agent_row(db_with_cleanup, lost_agent)
        assert row["status"] == AgentStatus.ALIVE
        assert row["schedulable"] is False

        # The agent-started path.
        await agent_db_source.update_agent_status(
            Updater[AgentRow](
                spec=AgentStatusUpdaterSpec(
                    status=AgentStatus.ALIVE, status_changed=datetime.now(tzutc())
                ),
                pk_value=lost_agent,
            )
        )
        assert await _schedulable(db_with_cleanup, lost_agent) is False


HANG_TIMEOUT = KernelLifecycleEventReason.HANG_TIMEOUT.value
RETRY_SEC = UNAVAILABLE_AGENT_CLEANUP_RETRY_SEC
HOUR = 3600


class _SessionFactory:
    """Creates TERMINATED sessions whose kernels sit on the given agents."""

    def __init__(
        self,
        db: ExtendedAsyncSAEngine,
        *,
        domain_id: DomainID,
        domain_name: str,
        group: _Group,
        project_id: uuid.UUID,
        user_uuid: uuid.UUID,
        access_key: AccessKey,
    ) -> None:
        self._db = db
        self._domain_id = domain_id
        self._domain_name = domain_name
        self._group = group
        self._project_id = project_id
        self._user_uuid = user_uuid
        self._access_key = access_key

    async def terminated(
        self,
        agent_ids: list[AgentId],
        *,
        reason: str = HANG_TIMEOUT,
        terminated_ago_sec: float = HOUR,
        kernel_status: KernelStatus = KernelStatus.TERMINATED,
    ) -> SessionId:
        session_id, _ = await create_pending_session_with_kernels(
            self._db,
            domain_id=self._domain_id,
            domain_name=self._domain_name,
            resource_group_id=self._group.id,
            scaling_group_name=self._group.name,
            group_id=self._project_id,
            user_uuid=self._user_uuid,
            access_key=self._access_key,
            agent_assignments=[(agent_id, Decimal("1"), Decimal("1024")) for agent_id in agent_ids],
            session_status=SessionStatus.TERMINATED,
            kernel_status=kernel_status,
            assign_agents=True,
        )
        terminated_at = sa.func.now() - sa.literal(
            timedelta(seconds=terminated_ago_sec), sa.Interval
        )
        async with self._db.begin() as conn:
            await conn.execute(
                sa.update(KernelRow)
                .where(KernelRow.session_id == session_id)
                .values(status_info=reason, terminated_at=terminated_at)
            )
        return session_id


@pytest.fixture
async def conftest_group(
    test_scaling_group_id: ResourceGroupID,
    test_scaling_group_name: str,
) -> _Group:
    return _Group(test_scaling_group_id, test_scaling_group_name)


@pytest.fixture
async def sessions(
    db_with_cleanup: ExtendedAsyncSAEngine,
    test_domain_id: DomainID,
    test_domain_name: str,
    conftest_group: _Group,
    test_group_id: uuid.UUID,
    test_user_uuid: uuid.UUID,
    test_access_key: AccessKey,
    resource_slot_types: None,
) -> _SessionFactory:
    return _SessionFactory(
        db_with_cleanup,
        domain_id=test_domain_id,
        domain_name=test_domain_name,
        group=conftest_group,
        project_id=test_group_id,
        user_uuid=test_user_uuid,
        access_key=test_access_key,
    )


class TestSessionsTerminatedOnFencedAgents:
    @pytest.fixture
    def db_source(self, db_with_cleanup: ExtendedAsyncSAEngine) -> ScheduleDBSource:
        return ScheduleDBSource(db_with_cleanup)

    async def _fenced(
        self, db: ExtendedAsyncSAEngine, group: _Group, status: AgentStatus = AgentStatus.LOST
    ) -> AgentId:
        return await _add_agent(
            db, group, status, unavailable_for_sec=RELEASE_SEC + 60, schedulable=False
        )

    async def test_finds_sessions_released_from_an_agent_still_away(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        conftest_group: _Group,
        sessions: _SessionFactory,
    ) -> None:
        lost = await self._fenced(db_with_cleanup, conftest_group, AgentStatus.LOST)
        stopped = await self._fenced(db_with_cleanup, conftest_group, AgentStatus.TERMINATED)
        on_lost = await sessions.terminated([lost])
        on_stopped = await sessions.terminated([stopped], terminated_ago_sec=RETRY_SEC - HOUR)
        # Two kernels of one session on the agent: the session is listed once.
        two_kernels = await sessions.terminated([lost, lost])

        found = await db_source.get_sessions_terminated_on_fenced_agents(HANG_TIMEOUT, RETRY_SEC)

        assert sorted(found) == sorted([on_lost, on_stopped, two_kernels])

    @pytest.mark.parametrize(
        ("status", "schedulable"),
        [
            # Back: the agent heartbeats again (it stays fenced).
            (AgentStatus.ALIVE, False),
            (AgentStatus.RESTARTING, False),
            # Re-enabled by an operator while still away.
            (AgentStatus.LOST, True),
            (AgentStatus.TERMINATED, True),
        ],
    )
    async def test_skips_agents_back_or_re_enabled(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        conftest_group: _Group,
        sessions: _SessionFactory,
        status: AgentStatus,
        schedulable: bool,
    ) -> None:
        agent_id = await _add_agent(
            db_with_cleanup,
            conftest_group,
            status,
            unavailable_for_sec=RELEASE_SEC + 60,
            schedulable=schedulable,
        )
        await sessions.terminated([agent_id])

        assert (
            await db_source.get_sessions_terminated_on_fenced_agents(HANG_TIMEOUT, RETRY_SEC) == []
        )

        # With another agent still fenced and away, only its sessions are listed.
        away = await self._fenced(db_with_cleanup, conftest_group)
        on_away = await sessions.terminated([away])
        found = await db_source.get_sessions_terminated_on_fenced_agents(HANG_TIMEOUT, RETRY_SEC)
        assert found == [on_away]

    @pytest.mark.parametrize(
        ("reason", "terminated_ago_sec", "kernel_status"),
        [
            # Another reason.
            ("force-terminated", HOUR, KernelStatus.TERMINATED),
            ("user-requested", HOUR, KernelStatus.TERMINATED),
            # Released too long ago.
            (HANG_TIMEOUT, RETRY_SEC + HOUR, KernelStatus.TERMINATED),
            # Not terminated.
            (HANG_TIMEOUT, HOUR, KernelStatus.TERMINATING),
            (HANG_TIMEOUT, HOUR, KernelStatus.CANCELLED),
        ],
    )
    async def test_skips_other_kernels(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        db_source: ScheduleDBSource,
        conftest_group: _Group,
        sessions: _SessionFactory,
        reason: str,
        terminated_ago_sec: float,
        kernel_status: KernelStatus,
    ) -> None:
        agent_id = await self._fenced(db_with_cleanup, conftest_group)
        await sessions.terminated(
            [agent_id],
            reason=reason,
            terminated_ago_sec=terminated_ago_sec,
            kernel_status=kernel_status,
        )

        assert (
            await db_source.get_sessions_terminated_on_fenced_agents(HANG_TIMEOUT, RETRY_SEC) == []
        )


class TestCleanupRequeueWithValkey:
    """The real cleanup handler against the database and Valkey (the agent RPC is a stub)."""

    @pytest.fixture
    async def valkey_schedule(
        self,
        redis_container: tuple[str, HostPortPairModel],
    ) -> AsyncIterator[ValkeyScheduleClient]:
        _, hostport_pair = redis_container
        client = await ValkeyScheduleClient.create(
            valkey_target=ValkeyTarget(addr=hostport_pair.address),
            db_id=0,
            human_readable_name="test-unavailable-agent-fence",
        )
        await _delete_cleanup_key(client)
        try:
            yield client
        finally:
            await _delete_cleanup_key(client)
            await client.close()

    @pytest.fixture
    def handler(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        valkey_schedule: ValkeyScheduleClient,
    ) -> CleanupForceTerminatedHandler:
        terminator = AsyncMock()
        terminator.terminate_sessions_for_handler = AsyncMock(return_value=[])
        return CleanupForceTerminatedHandler(
            terminator=terminator,
            # The db source has the same method the handler calls on the repository.
            repository=ScheduleDBSource(db_with_cleanup),  # type: ignore[arg-type]
            valkey_schedule=valkey_schedule,
        )

    async def test_requeues_while_the_agent_is_away_and_stops_once_it_is_back(
        self,
        db_with_cleanup: ExtendedAsyncSAEngine,
        conftest_group: _Group,
        sessions: _SessionFactory,
        handler: CleanupForceTerminatedHandler,
        valkey_schedule: ValkeyScheduleClient,
    ) -> None:
        await _add_agent(db_with_cleanup, conftest_group, AgentStatus.ALIVE)
        agent_id = await _add_agent(
            db_with_cleanup,
            conftest_group,
            AgentStatus.LOST,
            unavailable_for_sec=RELEASE_SEC + 60,
        )
        fenced = await ScheduleDBSource(db_with_cleanup).fence_unavailable_agents(
            [agent_id], conftest_group.id, RELEASE_SEC
        )
        assert [agent.agent_id for agent in fenced] == [agent_id]
        released = await sessions.terminated([agent_id])
        other = await sessions.terminated([agent_id], reason="user-requested")

        # Nothing was queued (the set expired, or adding to it failed): queued again.
        assert await handler.fetch_session_ids() == [released]
        assert await valkey_schedule.get_force_terminated_sessions() == [released]
        assert await _cleanup_ttl(valkey_schedule) > FORCE_TERMINATED_CLEANUP_TTL_SEC - 10

        # Each pass refreshes the set's TTL.
        await _expire_cleanup_key(valkey_schedule, 5)
        assert await handler.fetch_session_ids() == [released]
        assert await _cleanup_ttl(valkey_schedule) > FORCE_TERMINATED_CLEANUP_TTL_SEC - 10

        # Ids queued by others are kept and returned once.
        await valkey_schedule.add_force_terminated_sessions([other])
        assert sorted(await handler.fetch_session_ids()) == sorted([released, other])
        await valkey_schedule.remove_force_terminated_sessions([other])

        # The agent comes back (it stays fenced): no more requeues, the set runs out.
        await AgentDBSource(db_with_cleanup).upsert_agent_with_state(
            _heartbeat(agent_id, conftest_group.name)
        )
        assert await _schedulable(db_with_cleanup, agent_id) is False
        assert await handler.fetch_session_ids() == [released]
        await _delete_cleanup_key(valkey_schedule)
        assert await handler.fetch_session_ids() == []
        assert await valkey_schedule.get_force_terminated_sessions() == []


async def _delete_cleanup_key(client: ValkeyScheduleClient) -> None:
    async with client._client.client() as conn:
        await conn.delete([client._get_force_terminated_cleanup_key()])


async def _expire_cleanup_key(client: ValkeyScheduleClient, seconds: int) -> None:
    async with client._client.client() as conn:
        await conn.expire(client._get_force_terminated_cleanup_key(), seconds)


async def _cleanup_ttl(client: ValkeyScheduleClient) -> int:
    async with client._client.client() as conn:
        return await conn.ttl(client._get_force_terminated_cleanup_key())
