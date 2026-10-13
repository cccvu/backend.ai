"""Scheduler transitions move a session only while it is still in the status it
was read in.

Everything here runs on a real PostgreSQL database, through the real
``ScheduleDBSource`` / ``SchedulerRepository`` and, where a whole pass matters,
a real ``ScheduleCoordinator`` with the real handlers. Only the agents, Valkey
and the event producer are stand-ins.

Covered:

- ``SessionConditions.by_expected_statuses`` and the compare-and-set in
  ``update_with_history``, including a row held by an uncommitted terminate.
- A terminate (normal or forced) during a START pass, DEPRIORITIZE racing a
  terminate, and promote-to-running racing DETECT_KERNEL_TERMINATION.
- ``mark_sessions_status`` never overwrites TERMINATING, and a requeue that
  races a terminate still reaches TERMINATED through TERMINATE's cancel of
  unplaced PENDING kernels.
- A cancel racing the scheduling pass returns ``agent_resources.reserved`` to
  its baseline.
- The destroy of a kernel whose start is reported after it was terminated.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, cast, override
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from dateutil.tz import tzutc

from ai.backend.common.events.event_types.kernel.anycast import KernelStartedAnycastEvent
from ai.backend.common.events.event_types.session.broadcast import SchedulingBroadcastEvent
from ai.backend.common.identifier.domain import DomainID
from ai.backend.common.identifier.resource_group import ResourceGroupID
from ai.backend.common.types import AccessKey, AgentId, KernelId, SessionId
from ai.backend.manager.clients.agent import AgentClientPool
from ai.backend.manager.data.kernel.types import KernelStatus
from ai.backend.manager.data.session.types import (
    SchedulingResult,
    SessionStatus,
    TransitionStatus,
)
from ai.backend.manager.models.kernel import KernelRow
from ai.backend.manager.models.scheduling_history.row import SessionSchedulingHistoryRow
from ai.backend.manager.models.session import SessionRow
from ai.backend.manager.models.session.conditions import SessionConditions
from ai.backend.manager.models.utils import ExtendedAsyncSAEngine
from ai.backend.manager.repositories.base.creator import BulkCreator
from ai.backend.manager.repositories.base.updater import BatchUpdater
from ai.backend.manager.repositories.scheduler.db_source.db_source import ScheduleDBSource
from ai.backend.manager.repositories.scheduler.repository import SchedulerRepository
from ai.backend.manager.repositories.scheduler.updaters import SessionStatusBatchUpdaterSpec
from ai.backend.manager.repositories.scheduling_history.creators import (
    SessionSchedulingHistoryCreatorSpec,
)
from ai.backend.manager.sokovan.scheduler.coordinator import ScheduleCoordinator
from ai.backend.manager.sokovan.scheduler.factory import (
    CoordinatorHandlers,
    _create_promotion_specs,
)
from ai.backend.manager.sokovan.scheduler.handlers.base import SessionLifecycleHandler
from ai.backend.manager.sokovan.scheduler.handlers.lifecycle.deprioritize_sessions import (
    DeprioritizeSessionsLifecycleHandler,
)
from ai.backend.manager.sokovan.scheduler.handlers.lifecycle.start_sessions import (
    StartSessionsLifecycleHandler,
)
from ai.backend.manager.sokovan.scheduler.handlers.lifecycle.terminate_sessions import (
    TerminateSessionsLifecycleHandler,
)
from ai.backend.manager.sokovan.scheduler.launcher.launcher import SessionLauncher
from ai.backend.manager.sokovan.scheduler.results import (
    SessionExecutionResult,
    SessionTransitionInfo,
)
from ai.backend.manager.sokovan.scheduler.scheduler import SchedulerComponents
from ai.backend.manager.sokovan.scheduler.terminator.terminator import (
    SessionTerminator,
    SessionTerminatorArgs,
)
from ai.backend.manager.sokovan.scheduler.types import ScheduleType
from ai.backend.manager.views.sokovan.image import ImageConfigData
from ai.backend.manager.views.sokovan.lifecycle import SessionDataForStart, SessionWithKernels

from .conftest import (
    create_pending_session_with_kernels,
    fetch_agent_resources,
    make_creation_info,
    make_session_allocations,
    seed_agent_resources,
)

# How long a test waits for a statement to block on a row lock.
LOCK_WAIT_SEC = 5.0

# =============================================================================
# Stand-ins and harness
# =============================================================================


@dataclass
class _Destroy:
    agent_id: AgentId
    kernel_id: KernelId
    session_id: SessionId
    reason: str
    checked_out_connections: int


@dataclass
class _AgentPool:
    """``AgentClientPool`` stand-in recording destroy requests.

    Each destroy records how many database connections were checked out while
    the RPC ran, to prove the caller released its connection first.
    """

    db: ExtendedAsyncSAEngine
    destroys: list[_Destroy] = field(default_factory=list)
    fail_with: Exception | None = None

    @asynccontextmanager
    async def acquire(self, agent_id: AgentId) -> AsyncIterator[_AgentClient]:
        yield _AgentClient(self, agent_id)


@dataclass
class _AgentClient:
    pool: _AgentPool
    agent_id: AgentId

    async def destroy_kernel(
        self,
        kernel_id: KernelId,
        session_id: SessionId,
        reason: str,
        suppress_events: bool = False,
    ) -> None:
        self.pool.destroys.append(
            _Destroy(
                agent_id=self.agent_id,
                kernel_id=kernel_id,
                session_id=session_id,
                reason=reason,
                checked_out_connections=self.pool.db.sync_engine.pool.checkedout(),  # type: ignore[attr-defined]
            )
        )
        if self.pool.fail_with is not None:
            raise self.pool.fail_with


class _FakeLauncher:
    """Records what START dispatches and runs a callback in the middle of it."""

    started: list[list[SessionId]]
    during_dispatch: Callable[[], Awaitable[None]] | None

    def __init__(self) -> None:
        self.started = []
        self.during_dispatch = None

    async def start_sessions_for_handler(
        self,
        sessions: list[SessionDataForStart],
        image_configs: Mapping[uuid.UUID, ImageConfigData],
    ) -> None:
        self.started.append([s.session_id for s in sessions])
        if self.during_dispatch is not None:
            await self.during_dispatch()


class _AfterExecute(SessionLifecycleHandler):
    """Wraps a real handler and runs a callback once its ``execute`` returned,
    before the coordinator applies the result."""

    def __init__(
        self, inner: SessionLifecycleHandler, after: Callable[[], Awaitable[None]]
    ) -> None:
        self._inner = inner
        self._after = after

    @override
    def name(self) -> str:  # type: ignore[override]
        return self._inner.name()

    @override
    def target_statuses(self) -> list[SessionStatus]:  # type: ignore[override]
        return self._inner.target_statuses()

    @override
    def target_kernel_statuses(self) -> list[KernelStatus] | None:  # type: ignore[override]
        return self._inner.target_kernel_statuses()

    @override
    def status_transitions(self) -> Any:  # type: ignore[override]
        return self._inner.status_transitions()

    @property
    @override
    def lock_id(self) -> Any:
        return self._inner.lock_id

    @override
    async def execute(
        self, resource_group_id: ResourceGroupID, sessions: Sequence[SessionWithKernels]
    ) -> SessionExecutionResult:
        result = await self._inner.execute(resource_group_id, sessions)
        await self._after()
        return result


@dataclass
class _Harness:
    db: ExtendedAsyncSAEngine
    repository: SchedulerRepository
    coordinator: ScheduleCoordinator
    event_producer: AsyncMock
    terminator_valkey: AsyncMock
    agent_pool: _AgentPool
    launcher: _FakeLauncher
    hook_registry: MagicMock

    def broadcast_transitions(self) -> list[tuple[SessionId, str]]:
        sent: list[tuple[SessionId, str]] = []
        for call in self.event_producer.broadcast_events_batch.await_args_list:
            for event in call.args[0]:
                assert isinstance(event, SchedulingBroadcastEvent)
                sent.append((event.session_id, event.status_transition))
        return sent


def _build_harness(
    db: ExtendedAsyncSAEngine,
    *,
    lifecycle_handlers: Callable[
        [SchedulerRepository, SessionTerminator, _FakeLauncher],
        Mapping[ScheduleType, SessionLifecycleHandler],
    ] = lambda repository, terminator, launcher: {},
) -> _Harness:
    repository = SchedulerRepository(db, AsyncMock(), AsyncMock(), MagicMock(), MagicMock())
    agent_pool = _AgentPool(db)
    terminator_valkey = AsyncMock()
    terminator = SessionTerminator(
        SessionTerminatorArgs(
            repository=repository,
            agent_client_pool=cast(AgentClientPool, agent_pool),
            valkey_schedule=terminator_valkey,
        )
    )
    launcher = _FakeLauncher()
    config_provider = MagicMock()
    config_provider.config.manager.session_schedule_lock_lifetime = 30.0
    hook_registry = MagicMock()
    hook_registry.get_hook.return_value = None
    event_producer = AsyncMock()
    coordinator = ScheduleCoordinator(
        valkey_schedule=AsyncMock(),
        components=SchedulerComponents(
            provisioner=MagicMock(),
            launcher=cast(SessionLauncher, launcher),
            terminator=terminator,
            repository=repository,
            config_provider=config_provider,
            hook_registry=hook_registry,
        ),
        handlers=CoordinatorHandlers(
            lifecycle_handlers=lifecycle_handlers(repository, terminator, launcher),
            promotion_specs=_create_promotion_specs(),
            kernel_handlers={},
            kernel_observers={},
            cleanup_handlers={},
        ),
        scheduling_controller=AsyncMock(),
        event_producer=event_producer,
        lock_factory=lambda lock_id, lifetime: nullcontext(),  # type: ignore[arg-type]
    )
    return _Harness(
        db=db,
        repository=repository,
        coordinator=coordinator,
        event_producer=event_producer,
        terminator_valkey=terminator_valkey,
        agent_pool=agent_pool,
        launcher=launcher,
        hook_registry=hook_registry,
    )


def _start_handlers(
    repository: SchedulerRepository, terminator: SessionTerminator, launcher: _FakeLauncher
) -> Mapping[ScheduleType, SessionLifecycleHandler]:
    return {
        ScheduleType.START: StartSessionsLifecycleHandler(
            launcher=cast(SessionLauncher, launcher), repository=repository
        ),
        ScheduleType.TERMINATE: TerminateSessionsLifecycleHandler(
            terminator=terminator, repository=repository
        ),
    }


@dataclass
class _Scope:
    """The FK-complete rows a session needs."""

    db: ExtendedAsyncSAEngine
    domain_id: DomainID
    domain_name: str
    resource_group_id: ResourceGroupID
    scaling_group_name: str
    group_id: uuid.UUID
    user_uuid: uuid.UUID
    access_key: AccessKey
    agent_id: str

    async def session(
        self,
        session_status: SessionStatus,
        kernel_status: KernelStatus,
        *,
        kernels: int = 1,
        assign_agents: bool = True,
    ) -> tuple[SessionId, list[KernelId]]:
        return await create_pending_session_with_kernels(
            self.db,
            domain_id=self.domain_id,
            domain_name=self.domain_name,
            resource_group_id=self.resource_group_id,
            scaling_group_name=self.scaling_group_name,
            group_id=self.group_id,
            user_uuid=self.user_uuid,
            access_key=self.access_key,
            agent_assignments=[(self.agent_id, Decimal("2"), Decimal("4096"))] * kernels,
            session_status=session_status,
            kernel_status=kernel_status,
            assign_agents=assign_agents,
        )


@pytest.fixture
def scope(
    db_with_cleanup: ExtendedAsyncSAEngine,
    test_domain_id: DomainID,
    test_domain_name: str,
    test_scaling_group_id: ResourceGroupID,
    test_scaling_group_name: str,
    test_group_id: uuid.UUID,
    test_user_uuid: uuid.UUID,
    test_access_key: AccessKey,
    test_agent_id: str,
    resource_slot_types: None,
) -> _Scope:
    return _Scope(
        db=db_with_cleanup,
        domain_id=test_domain_id,
        domain_name=test_domain_name,
        resource_group_id=test_scaling_group_id,
        scaling_group_name=test_scaling_group_name,
        group_id=test_group_id,
        user_uuid=test_user_uuid,
        access_key=test_access_key,
        agent_id=test_agent_id,
    )


async def _session_status(db: ExtendedAsyncSAEngine, session_id: SessionId) -> SessionStatus:
    async with db.begin_readonly_session() as db_sess:
        status = await db_sess.scalar(
            sa.select(SessionRow.status).where(SessionRow.id == session_id)
        )
    assert status is not None
    return SessionStatus(status)


async def _kernels(db: ExtendedAsyncSAEngine, session_id: SessionId) -> list[KernelRow]:
    async with db.begin_readonly_session() as db_sess:
        rows = (
            await db_sess.scalars(
                sa.select(KernelRow)
                .where(KernelRow.session_id == session_id)
                .order_by(KernelRow.cluster_idx)
            )
        ).all()
    return list(rows)


async def _history(
    db: ExtendedAsyncSAEngine, session_id: SessionId
) -> list[tuple[str, str | None, str | None]]:
    """(phase, from_status, to_status) of the session's history rows, oldest first."""
    async with db.begin_readonly_session() as db_sess:
        rows = (
            await db_sess.scalars(
                sa.select(SessionSchedulingHistoryRow)
                .where(SessionSchedulingHistoryRow.session_id == session_id)
                .order_by(SessionSchedulingHistoryRow.created_at)
            )
        ).all()
    return [(row.phase, row.from_status, row.to_status) for row in rows]


async def _set_kernel_status(
    db: ExtendedAsyncSAEngine, kernel_id: KernelId, status: KernelStatus
) -> None:
    async with db.begin_session() as db_sess:
        await db_sess.execute(
            sa.update(KernelRow).where(KernelRow.id == kernel_id).values(status=status)
        )


def _guarded_updater(
    expected: Mapping[SessionId, SessionStatus], to_status: SessionStatus
) -> BatchUpdater[SessionRow]:
    return BatchUpdater(
        spec=SessionStatusBatchUpdaterSpec(
            to_status=to_status,
            status_changed_at=datetime.now(tzutc()),
            reason="test",
        ),
        conditions=[
            SessionConditions.by_ids(list(expected)),
            SessionConditions.by_expected_statuses(expected),
        ],
    )


def _history_creator(
    expected: Mapping[SessionId, SessionStatus], to_status: SessionStatus
) -> BulkCreator[SessionSchedulingHistoryRow]:
    return BulkCreator(
        specs=[
            SessionSchedulingHistoryCreatorSpec(
                session_id=session_id,
                phase="test-phase",
                result=SchedulingResult.SUCCESS,
                message="test",
                from_status=from_status,
                to_status=to_status,
            )
            for session_id, from_status in expected.items()
        ]
    )


def _transition_info(session_id: SessionId, from_status: SessionStatus) -> SessionTransitionInfo:
    return SessionTransitionInfo(
        session_id=session_id,
        from_status=from_status,
        reason="test",
        creation_id=str(uuid.uuid4()),
        access_key=AccessKey("test"),
    )


async def _wait_for_lock_waiter(db: ExtendedAsyncSAEngine) -> None:
    """Wait until some backend of this database is blocked on a row lock."""
    async with asyncio.timeout(LOCK_WAIT_SEC):
        while True:
            async with db.begin_readonly() as conn:
                waiting = await conn.scalar(
                    sa.text(
                        "SELECT count(*) FROM pg_stat_activity "
                        "WHERE datname = current_database() AND wait_event_type = 'Lock'"
                    )
                )
            if waiting:
                return
            await asyncio.sleep(0.02)


# =============================================================================
# The compare-and-set itself
# =============================================================================


class TestExpectedStatusesGuard:
    async def test_matches_each_session_only_in_its_own_expected_status(
        self, scope: _Scope
    ) -> None:
        prepared, _ = await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)
        creating, _ = await scope.session(SessionStatus.CREATING, KernelStatus.CREATING)
        condition = SessionConditions.by_expected_statuses({
            prepared: SessionStatus.PREPARED,
            # Read as PREPARED, but it has moved on to CREATING.
            creating: SessionStatus.PREPARED,
        })
        async with scope.db.begin_readonly_session() as db_sess:
            matched = (await db_sess.scalars(sa.select(SessionRow.id).where(condition()))).all()
        assert matched == [prepared]

    async def test_an_empty_mapping_matches_nothing(self, scope: _Scope) -> None:
        await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)
        condition = SessionConditions.by_expected_statuses({})
        async with scope.db.begin_readonly_session() as db_sess:
            matched = (await db_sess.scalars(sa.select(SessionRow.id).where(condition()))).all()
        assert matched == []

    async def test_update_moves_and_records_only_the_sessions_still_in_place(
        self, scope: _Scope
    ) -> None:
        stays, _ = await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)
        moved_away, _ = await scope.session(SessionStatus.TERMINATING, KernelStatus.TERMINATING)
        expected = {stays: SessionStatus.PREPARED, moved_away: SessionStatus.PREPARED}

        moved = await ScheduleDBSource(scope.db).update_with_history(
            _guarded_updater(expected, SessionStatus.CREATING),
            _history_creator(expected, SessionStatus.CREATING),
        )

        assert moved == [stays]
        assert await _session_status(scope.db, stays) == SessionStatus.CREATING
        assert await _session_status(scope.db, moved_away) == SessionStatus.TERMINATING
        assert await _history(scope.db, stays) == [
            ("test-phase", str(SessionStatus.PREPARED), str(SessionStatus.CREATING))
        ]
        assert await _history(scope.db, moved_away) == []

    async def test_an_uncommitted_terminate_blocks_the_update_which_then_moves_nothing(
        self, scope: _Scope
    ) -> None:
        session_id, _ = await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)
        expected = {session_id: SessionStatus.PREPARED}
        db_source = ScheduleDBSource(scope.db)

        async with scope.db.connect() as blocker:
            transaction = await blocker.begin()
            await blocker.execute(
                sa.update(SessionRow.__table__)
                .where(SessionRow.__table__.c.id == session_id)
                .values(status=SessionStatus.TERMINATING)
            )
            update = asyncio.create_task(
                db_source.update_with_history(
                    _guarded_updater(expected, SessionStatus.CREATING),
                    _history_creator(expected, SessionStatus.CREATING),
                )
            )
            # The guarded update has to wait for the terminate's row lock.
            await _wait_for_lock_waiter(scope.db)
            assert not update.done()
            await transaction.commit()

        assert await asyncio.wait_for(update, LOCK_WAIT_SEC) == []
        assert await _session_status(scope.db, session_id) == SessionStatus.TERMINATING
        assert await _history(scope.db, session_id) == []

    async def test_a_rolled_back_terminate_lets_the_blocked_update_through(
        self, scope: _Scope
    ) -> None:
        session_id, _ = await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)
        expected = {session_id: SessionStatus.PREPARED}
        db_source = ScheduleDBSource(scope.db)

        async with scope.db.connect() as blocker:
            transaction = await blocker.begin()
            await blocker.execute(
                sa.update(SessionRow.__table__)
                .where(SessionRow.__table__.c.id == session_id)
                .values(status=SessionStatus.TERMINATING)
            )
            update = asyncio.create_task(
                db_source.update_with_history(
                    _guarded_updater(expected, SessionStatus.CREATING),
                    _history_creator(expected, SessionStatus.CREATING),
                )
            )
            await _wait_for_lock_waiter(scope.db)
            await transaction.rollback()

        assert await asyncio.wait_for(update, LOCK_WAIT_SEC) == [session_id]
        assert await _session_status(scope.db, session_id) == SessionStatus.CREATING


# =============================================================================
# Coordinator transitions
# =============================================================================


class TestApplyTransition:
    async def test_a_pending_reset_follows_only_the_sessions_that_moved(
        self, scope: _Scope
    ) -> None:
        harness = _build_harness(scope.db)
        moved_away, moved_away_kernels = await scope.session(
            SessionStatus.TERMINATING, KernelStatus.TERMINATING
        )
        stays, stays_kernels = await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)

        moved = await harness.coordinator._apply_transition(
            "start-sessions",
            [
                _transition_info(moved_away, SessionStatus.PREPARED),
                _transition_info(stays, SessionStatus.PREPARED),
            ],
            TransitionStatus(session=SessionStatus.PENDING, kernel=KernelStatus.PENDING),
            SchedulingResult.EXPIRED,
            {},
            datetime.now(tzutc()),
        )

        assert moved == {stays}
        assert await _session_status(scope.db, moved_away) == SessionStatus.TERMINATING
        (kept,) = await _kernels(scope.db, moved_away)
        assert kept.status == KernelStatus.TERMINATING
        assert kept.agent == scope.agent_id
        assert await _session_status(scope.db, stays) == SessionStatus.PENDING
        (reset,) = await _kernels(scope.db, stays)
        assert reset.status == KernelStatus.PENDING
        assert reset.agent is None
        assert [k.id for k in (kept, reset)] == [moved_away_kernels[0], stays_kernels[0]]

    async def test_the_scheduling_pass_never_writes_over_a_cancel_and_reserved_returns(
        self,
        scope: _Scope,
    ) -> None:
        """A cancel that lands between the pass's reservation and its
        PENDING -> SCHEDULED move leaves the session CANCELLED and gives the
        agent its reservation back."""
        harness = _build_harness(scope.db)
        await seed_agent_resources(
            scope.db, scope.agent_id, cpu_capacity=Decimal("10"), mem_capacity=Decimal("10240")
        )
        session_id, kernel_ids = await scope.session(
            SessionStatus.PENDING, KernelStatus.PENDING, assign_agents=False
        )
        db_source = ScheduleDBSource(scope.db)
        # The pass reserves and places the kernels; the session is still PENDING.
        allocated = await db_source.allocate_sessions(
            make_session_allocations(
                session_id=session_id,
                kernel_assignments=[(kernel_ids[0], scope.agent_id)],
            )
        )
        assert allocated == [session_id]
        reserved = await fetch_agent_resources(scope.db, scope.agent_id)
        assert reserved["cpu"].reserved == Decimal("2")
        assert reserved["mem"].reserved == Decimal("4096")

        # The user cancels before the pass moves the session.
        result = await harness.repository.mark_sessions_terminating([session_id])
        assert result.cancelled_sessions == [session_id]

        moved = await harness.coordinator._apply_transition(
            "schedule-sessions",
            [_transition_info(session_id, SessionStatus.PENDING)],
            TransitionStatus(session=SessionStatus.SCHEDULED, kernel=KernelStatus.SCHEDULED),
            SchedulingResult.SUCCESS,
            {},
            datetime.now(tzutc()),
        )

        assert moved == set()
        assert await _session_status(scope.db, session_id) == SessionStatus.CANCELLED
        assert [phase for phase, _, _ in await _history(scope.db, session_id)] == ["cancel"]
        released = await fetch_agent_resources(scope.db, scope.agent_id)
        assert released["cpu"].reserved == Decimal("0")
        assert released["mem"].reserved == Decimal("0")


class TestTerminateDuringStart:
    async def test_a_terminate_during_dispatch_leaves_the_session_terminating(
        self, scope: _Scope
    ) -> None:
        harness = _build_harness(scope.db, lifecycle_handlers=_start_handlers)
        session_id, _ = await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)

        async def terminate() -> None:
            result = await harness.repository.mark_sessions_terminating([session_id])
            assert result.terminating_sessions == [session_id]

        harness.launcher.during_dispatch = terminate
        assert await harness.coordinator.process_lifecycle_schedule(ScheduleType.START)

        assert harness.launcher.started == [[session_id]]
        assert await _session_status(scope.db, session_id) == SessionStatus.TERMINATING
        assert [to for _, _, to in await _history(scope.db, session_id)] == [
            str(SessionStatus.TERMINATING)
        ]
        assert harness.broadcast_transitions() == []

    async def test_a_forced_terminate_during_dispatch_stays_terminated(self, scope: _Scope) -> None:
        harness = _build_harness(scope.db, lifecycle_handlers=_start_handlers)
        session_id, _ = await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)

        async def force_terminate() -> None:
            result = await harness.repository.mark_sessions_terminating([session_id], forced=True)
            assert result.force_terminated_sessions == [session_id]

        harness.launcher.during_dispatch = force_terminate
        assert await harness.coordinator.process_lifecycle_schedule(ScheduleType.START)

        assert await _session_status(scope.db, session_id) == SessionStatus.TERMINATED
        assert str(SessionStatus.CREATING) not in [
            to for _, _, to in await _history(scope.db, session_id)
        ]
        assert harness.broadcast_transitions() == []

    async def test_a_session_terminated_before_the_reread_is_neither_started_nor_reported(
        self, scope: _Scope
    ) -> None:
        harness = _build_harness(scope.db, lifecycle_handlers=_start_handlers)
        session_id, _ = await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)
        other_id, _ = await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)
        reread = harness.repository.search_sessions_with_kernels_and_user

        async def terminate_then_reread(querier: Any) -> Any:
            await harness.repository.mark_sessions_terminating([session_id])
            return await reread(querier)

        harness.repository.search_sessions_with_kernels_and_user = terminate_then_reread  # type: ignore[method-assign]
        assert await harness.coordinator.process_lifecycle_schedule(ScheduleType.START)

        assert harness.launcher.started == [[other_id]]
        assert await _session_status(scope.db, session_id) == SessionStatus.TERMINATING
        assert [phase for phase, _, _ in await _history(scope.db, session_id)] == [
            "mark_terminating"
        ]
        assert await _session_status(scope.db, other_id) == SessionStatus.CREATING
        assert harness.broadcast_transitions() == [(other_id, str(SessionStatus.CREATING))]


class TestDeprioritizeRacingTerminate:
    async def test_a_terminate_wins_over_deprioritize_and_the_session_terminates(
        self, scope: _Scope
    ) -> None:
        session_id, kernel_ids = await scope.session(
            SessionStatus.DEPRIORITIZING, KernelStatus.PENDING, assign_agents=False
        )

        def handlers(
            repository: SchedulerRepository,
            terminator: SessionTerminator,
            launcher: _FakeLauncher,
        ) -> Mapping[ScheduleType, SessionLifecycleHandler]:
            async def terminate() -> None:
                result = await repository.mark_sessions_terminating([session_id])
                assert result.terminating_sessions == [session_id]

            return {
                ScheduleType.DEPRIORITIZE: _AfterExecute(
                    DeprioritizeSessionsLifecycleHandler(repository), terminate
                ),
                ScheduleType.TERMINATE: TerminateSessionsLifecycleHandler(
                    terminator=terminator, repository=repository
                ),
            }

        harness = _build_harness(scope.db, lifecycle_handlers=handlers)
        assert await harness.coordinator.process_lifecycle_schedule(ScheduleType.DEPRIORITIZE)

        assert await _session_status(scope.db, session_id) == SessionStatus.TERMINATING
        assert str(SessionStatus.PENDING) not in [
            to for _, _, to in await _history(scope.db, session_id)
        ]
        assert harness.broadcast_transitions() == []

        # The never-placed PENDING kernel is cancelled by TERMINATE, and the
        # session finishes.
        assert await harness.coordinator.process_lifecycle_schedule(ScheduleType.TERMINATE)
        (kernel,) = await _kernels(scope.db, session_id)
        assert kernel.id == kernel_ids[0]
        assert kernel.status == KernelStatus.CANCELLED
        harness.terminator_valkey.mark_schedules_needed_batch.assert_awaited_once_with([
            ScheduleType.CHECK_TERMINATING_PROGRESS.value
        ])
        assert harness.agent_pool.destroys == []
        assert await harness.coordinator.process_lifecycle_schedule(
            ScheduleType.CHECK_TERMINATING_PROGRESS
        )
        assert await _session_status(scope.db, session_id) == SessionStatus.TERMINATED


class TestPromotionRacingDetect:
    async def test_detect_wins_over_promote_to_running(self, scope: _Scope) -> None:
        """Both promotions read the CREATING session; DETECT moves it to
        TERMINATING while promote-to-running runs its RUNNING hook, so the
        promotion's own move must not land."""
        harness = _build_harness(scope.db)
        session_id, kernel_ids = await scope.session(
            SessionStatus.CREATING, KernelStatus.RUNNING, kernels=2
        )
        await _set_kernel_status(scope.db, kernel_ids[1], KernelStatus.TERMINATED)

        async def detect_meanwhile(session: SessionWithKernels) -> None:
            assert await harness.coordinator.process_lifecycle_schedule(
                ScheduleType.DETECT_KERNEL_TERMINATION
            )

        running_hook = MagicMock()
        running_hook.execute = AsyncMock(side_effect=detect_meanwhile)
        harness.hook_registry.get_hook.side_effect = lambda status: (
            running_hook if status == SessionStatus.RUNNING else None
        )

        assert await harness.coordinator.process_lifecycle_schedule(
            ScheduleType.CHECK_CREATING_PROGRESS
        )

        running_hook.execute.assert_awaited_once()
        assert await _session_status(scope.db, session_id) == SessionStatus.TERMINATING
        assert [to for _, _, to in await _history(scope.db, session_id)] == [
            str(SessionStatus.TERMINATING)
        ]
        assert harness.broadcast_transitions() == [(session_id, str(SessionStatus.TERMINATING))]

    async def test_a_session_that_moved_before_the_hook_fetch_gets_no_hook(
        self, scope: _Scope
    ) -> None:
        harness = _build_harness(scope.db)
        session_id, _ = await scope.session(SessionStatus.CREATING, KernelStatus.RUNNING)
        fetch = harness.repository.search_sessions_with_kernels_for_handler

        async def terminate_then_fetch(querier: Any) -> Any:
            result = await harness.repository.mark_sessions_terminating([session_id])
            assert result.terminating_sessions == [session_id]
            return await fetch(querier)

        harness.repository.search_sessions_with_kernels_for_handler = terminate_then_fetch  # type: ignore[method-assign]
        running_hook = MagicMock()
        running_hook.execute = AsyncMock()
        harness.hook_registry.get_hook.side_effect = lambda status: (
            running_hook if status == SessionStatus.RUNNING else None
        )

        assert await harness.coordinator.process_lifecycle_schedule(
            ScheduleType.CHECK_CREATING_PROGRESS
        )

        running_hook.execute.assert_not_awaited()
        assert await _session_status(scope.db, session_id) == SessionStatus.TERMINATING
        assert harness.broadcast_transitions() == []


# =============================================================================
# mark_sessions_status and the requeue racing a terminate
# =============================================================================


class TestMarkSessionsStatus:
    async def test_a_terminating_session_is_not_marked(self, scope: _Scope) -> None:
        terminating, _ = await scope.session(SessionStatus.TERMINATING, KernelStatus.TERMINATING)
        running, _ = await scope.session(SessionStatus.RUNNING, KernelStatus.RUNNING)
        repository = _build_harness(scope.db).repository

        marked = await repository.mark_sessions_status(
            [terminating, running], SessionStatus.RESCHEDULING, reason="test"
        )

        assert marked == [running]
        assert await _session_status(scope.db, terminating) == SessionStatus.TERMINATING
        assert await _history(scope.db, terminating) == []

    async def test_a_requeue_racing_a_terminate_still_reaches_terminated(
        self, scope: _Scope
    ) -> None:
        harness = _build_harness(scope.db, lifecycle_handlers=_start_handlers)
        session_id, _ = await scope.session(SessionStatus.RESCHEDULING, KernelStatus.TERMINATED)

        # The requeue resets the kernels, then a terminate lands before the
        # session is moved back to PENDING.
        await harness.repository.reset_kernels_to_pending_for_sessions([session_id], "requeue")
        result = await harness.repository.mark_sessions_terminating([session_id])
        assert result.terminating_sessions == [session_id]
        requeued = await harness.repository.mark_sessions_status(
            [session_id], SessionStatus.PENDING, reason="requeue"
        )
        assert requeued == []
        assert await _session_status(scope.db, session_id) == SessionStatus.TERMINATING
        (kernel,) = await _kernels(scope.db, session_id)
        assert kernel.status == KernelStatus.PENDING
        assert kernel.agent is None

        assert await harness.coordinator.process_lifecycle_schedule(ScheduleType.TERMINATE)
        (kernel,) = await _kernels(scope.db, session_id)
        assert kernel.status == KernelStatus.CANCELLED
        assert harness.agent_pool.destroys == []
        assert await harness.coordinator.process_lifecycle_schedule(
            ScheduleType.CHECK_TERMINATING_PROGRESS
        )
        assert await _session_status(scope.db, session_id) == SessionStatus.TERMINATED

    async def test_terminate_leaves_placed_and_non_pending_kernels_to_the_agent(
        self, scope: _Scope
    ) -> None:
        harness = _build_harness(scope.db, lifecycle_handlers=_start_handlers)
        session_id, kernel_ids = await scope.session(
            SessionStatus.TERMINATING, KernelStatus.TERMINATING, kernels=2
        )

        assert await harness.coordinator.process_lifecycle_schedule(ScheduleType.TERMINATE)

        assert [k.status for k in await _kernels(scope.db, session_id)] == [
            KernelStatus.TERMINATING,
            KernelStatus.TERMINATING,
        ]
        assert sorted(d.kernel_id for d in harness.agent_pool.destroys) == sorted(kernel_ids)
        harness.terminator_valkey.mark_schedules_needed_batch.assert_not_awaited()


# =============================================================================
# A kernel reported started after it was terminated
# =============================================================================


def _started_event(session_id: SessionId, kernel_id: KernelId) -> KernelStartedAnycastEvent:
    return KernelStartedAnycastEvent(
        kernel_id=kernel_id,
        session_id=session_id,
        reason="started",
        creation_info=dataclasses.asdict(make_creation_info()),
    )


class TestKernelStartedAfterTermination:
    @pytest.mark.parametrize(
        ("session_status", "kernel_status"),
        [
            (SessionStatus.TERMINATED, KernelStatus.TERMINATED),
            (SessionStatus.TERMINATING, KernelStatus.TERMINATED),
            (SessionStatus.CANCELLED, KernelStatus.CANCELLED),
            (SessionStatus.ERROR, KernelStatus.ERROR),
        ],
    )
    async def test_a_terminal_kernel_of_a_finished_session_is_destroyed_once(
        self,
        scope: _Scope,
        session_status: SessionStatus,
        kernel_status: KernelStatus,
    ) -> None:
        harness = _build_harness(scope.db)
        session_id, (kernel_id,) = await scope.session(session_status, kernel_status)

        accepted = await harness.coordinator.handle_kernel_running(
            _started_event(session_id, kernel_id), source=AgentId(scope.agent_id)
        )

        assert accepted is False
        assert len(harness.agent_pool.destroys) == 1
        destroy = harness.agent_pool.destroys[0]
        assert destroy.agent_id == scope.agent_id
        assert destroy.kernel_id == kernel_id
        assert destroy.session_id == session_id
        assert destroy.reason == "already-terminated"
        # The database connection was given back before the agent was called.
        assert destroy.checked_out_connections == 0
        (kernel,) = await _kernels(scope.db, session_id)
        assert kernel.status == kernel_status

    @pytest.mark.parametrize(
        "session_status",
        [SessionStatus.RESCHEDULING, SessionStatus.PENDING, SessionStatus.SCHEDULED],
    )
    async def test_no_destroy_while_the_session_may_reuse_the_kernel(
        self, scope: _Scope, session_status: SessionStatus
    ) -> None:
        harness = _build_harness(scope.db)
        session_id, (kernel_id,) = await scope.session(session_status, KernelStatus.TERMINATED)

        accepted = await harness.coordinator.handle_kernel_running(
            _started_event(session_id, kernel_id), source=AgentId(scope.agent_id)
        )

        assert accepted is False
        assert harness.agent_pool.destroys == []

    async def test_no_destroy_for_an_event_from_another_agent(self, scope: _Scope) -> None:
        harness = _build_harness(scope.db)
        session_id, (kernel_id,) = await scope.session(
            SessionStatus.TERMINATED, KernelStatus.TERMINATED
        )

        accepted = await harness.coordinator.handle_kernel_running(
            _started_event(session_id, kernel_id), source=AgentId("some-other-agent")
        )

        assert accepted is False
        assert harness.agent_pool.destroys == []

    async def test_no_destroy_for_a_kernel_without_an_agent(self, scope: _Scope) -> None:
        harness = _build_harness(scope.db)
        session_id, (kernel_id,) = await scope.session(
            SessionStatus.CANCELLED, KernelStatus.CANCELLED, assign_agents=False
        )

        accepted = await harness.coordinator.handle_kernel_running(
            _started_event(session_id, kernel_id), source=AgentId(scope.agent_id)
        )

        assert accepted is False
        assert harness.agent_pool.destroys == []

    async def test_no_destroy_for_a_kernel_that_is_not_terminal(self, scope: _Scope) -> None:
        harness = _build_harness(scope.db)
        session_id, (kernel_id,) = await scope.session(
            SessionStatus.TERMINATING, KernelStatus.TERMINATING
        )

        accepted = await harness.coordinator.handle_kernel_running(
            _started_event(session_id, kernel_id), source=AgentId(scope.agent_id)
        )

        assert accepted is False
        assert harness.agent_pool.destroys == []

    async def test_an_accepted_start_destroys_nothing(self, scope: _Scope) -> None:
        harness = _build_harness(scope.db)
        session_id, (kernel_id,) = await scope.session(
            SessionStatus.CREATING, KernelStatus.CREATING
        )

        accepted = await harness.coordinator.handle_kernel_running(
            _started_event(session_id, kernel_id), source=AgentId(scope.agent_id)
        )

        assert accepted is True
        assert harness.agent_pool.destroys == []
        (kernel,) = await _kernels(scope.db, session_id)
        assert kernel.status == KernelStatus.RUNNING

    async def test_a_failing_destroy_does_not_raise(self, scope: _Scope) -> None:
        harness = _build_harness(scope.db)
        harness.agent_pool.fail_with = ConnectionError("agent unreachable")
        session_id, (kernel_id,) = await scope.session(
            SessionStatus.TERMINATED, KernelStatus.TERMINATED
        )

        accepted = await harness.coordinator.handle_kernel_running(
            _started_event(session_id, kernel_id), source=AgentId(scope.agent_id)
        )

        assert accepted is False
        assert len(harness.agent_pool.destroys) == 1

    async def test_a_failing_lookup_does_not_raise(self, scope: _Scope) -> None:
        harness = _build_harness(scope.db)
        session_id, (kernel_id,) = await scope.session(
            SessionStatus.TERMINATED, KernelStatus.TERMINATED
        )
        harness.repository.search_kernels_for_handler = AsyncMock(  # type: ignore[method-assign]
            side_effect=ConnectionError("database unavailable")
        )

        accepted = await harness.coordinator.handle_kernel_running(
            _started_event(session_id, kernel_id), source=AgentId(scope.agent_id)
        )

        assert accepted is False
        assert harness.agent_pool.destroys == []


# =============================================================================
# START never sends a session again once a kernel is past PREPARED
# =============================================================================


class TestStartSkipsSessionsAlreadySent:
    async def test_a_partly_started_session_is_skipped_and_stays_prepared(
        self, scope: _Scope
    ) -> None:
        harness = _build_harness(scope.db, lifecycle_handlers=_start_handlers)
        partial_id, partial_kernels = await scope.session(
            SessionStatus.PREPARED, KernelStatus.PREPARED, kernels=2
        )
        await _set_kernel_status(scope.db, partial_kernels[1], KernelStatus.CREATING)
        fresh_id, _ = await scope.session(SessionStatus.PREPARED, KernelStatus.PREPARED)

        assert await harness.coordinator.process_lifecycle_schedule(ScheduleType.START)

        assert harness.launcher.started == [[fresh_id]]
        assert await _session_status(scope.db, partial_id) == SessionStatus.PREPARED
        assert await _history(scope.db, partial_id) == [
            ("start-sessions", str(SessionStatus.PREPARED), str(SessionStatus.PREPARED))
        ]
        assert [k.status for k in await _kernels(scope.db, partial_id)] == [
            KernelStatus.PREPARED,
            KernelStatus.CREATING,
        ]
        assert await _session_status(scope.db, fresh_id) == SessionStatus.CREATING
        assert harness.broadcast_transitions() == [(fresh_id, str(SessionStatus.CREATING))]
