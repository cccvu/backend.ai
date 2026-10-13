"""Unit tests for ExpireStuckStartsLifecycleHandler (the stuck-start expiry)."""

from __future__ import annotations

import dataclasses
import logging
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai.backend.common.identifier.resource_group import ResourceGroupID
from ai.backend.manager.data.kernel.types import KernelStatus
from ai.backend.manager.data.session.types import SessionStatus, TransitionStatus
from ai.backend.manager.defs import (
    AGENT_CREATE_KERNELS_TIMEOUT_SEC,
    SESSION_CREATING_EXPIRY_SEC,
    SESSION_PULLING_EXPIRY_SEC,
)
from ai.backend.manager.repositories.scheduler.updaters import SessionStatusBatchUpdaterSpec
from ai.backend.manager.sokovan.scheduler.coordinator import ScheduleCoordinator
from ai.backend.manager.sokovan.scheduler.factory import (
    CoordinatorHandlers,
    _create_lifecycle_handlers,
)
from ai.backend.manager.sokovan.scheduler.handlers.maintenance.expire_stuck_starts import (
    EXPIRY_REASON,
    ExpireStuckStartsLifecycleHandler,
)
from ai.backend.manager.sokovan.scheduler.post_processors.base import PostProcessorContext
from ai.backend.manager.sokovan.scheduler.post_processors.schedule_marking import (
    ScheduleMarkingPostProcessor,
)
from ai.backend.manager.sokovan.scheduler.results import SessionExecutionResult
from ai.backend.manager.sokovan.scheduler.scheduler import SchedulerComponents
from ai.backend.manager.sokovan.scheduler.types import ScheduleType
from ai.backend.manager.views.sokovan.lifecycle import SessionWithKernels

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_RG = ResourceGroupID(uuid.uuid4())

SessionFactory = Callable[..., SessionWithKernels]


def _stuck(
    session_factory: SessionFactory,
    status: SessionStatus,
    kernel_statuses: Sequence[KernelStatus],
    age: float,
    *,
    agentless: bool = False,
) -> SessionWithKernels:
    """A session that entered ``status`` ``age`` seconds before ``_NOW``."""
    session = session_factory(status=status, kernel_count=len(kernel_statuses))
    entered_at = _NOW - timedelta(seconds=age)
    lifecycle = dataclasses.replace(
        session.session_info.lifecycle,
        status_changed=entered_at,
        # Earlier statuses carry older timestamps; only the current one counts.
        status_history={
            SessionStatus.PENDING.name: (entered_at - timedelta(days=1)).isoformat(),
            status.name: entered_at.isoformat(),
        },
    )
    kernels = [
        dataclasses.replace(
            kernel,
            lifecycle=dataclasses.replace(kernel.lifecycle, status=kernel_status),
            resource=dataclasses.replace(
                kernel.resource,
                agent=None if agentless else kernel.resource.agent,
                agent_addr=None if agentless else kernel.resource.agent_addr,
            ),
        )
        for kernel, kernel_status in zip(session.kernel_infos, kernel_statuses, strict=True)
    ]
    return SessionWithKernels(
        session_info=dataclasses.replace(session.session_info, lifecycle=lifecycle),
        kernel_infos=kernels,
    )


def _ids(sessions: Sequence[SessionWithKernels]) -> list[uuid.UUID]:
    return [s.session_info.identity.id for s in sessions]


class TestExpireStuckStartsLifecycleHandler:
    @pytest.fixture
    def repository(self, mock_repository: AsyncMock) -> AsyncMock:
        mock_repository.get_db_now = AsyncMock(return_value=_NOW)
        return mock_repository

    @pytest.fixture
    def handler(self, repository: AsyncMock) -> ExpireStuckStartsLifecycleHandler:
        return ExpireStuckStartsLifecycleHandler(repository, AsyncMock())

    # ------------------------------------------------------------------ declaration

    def test_bounds(self) -> None:
        assert SESSION_CREATING_EXPIRY_SEC == AGENT_CREATE_KERNELS_TIMEOUT_SEC + 300 == 900
        # Above the agent's default image pull timeout (2 h).
        assert SESSION_PULLING_EXPIRY_SEC == 8100 > 7200

    def test_declaration(self, handler: ExpireStuckStartsLifecycleHandler) -> None:
        assert set(handler.target_statuses()) == {
            SessionStatus.SCHEDULED,
            SessionStatus.PREPARING,
            SessionStatus.PULLING,
            SessionStatus.PREPARED,
            SessionStatus.CREATING,
            # For the release of sessions held by an unavailable agent (reports nothing).
            SessionStatus.TERMINATING,
        }
        # Sessions whose kernels are all PENDING must be seen too.
        assert handler.target_kernel_statuses() is None
        assert handler.lock_id is None
        transitions = handler.status_transitions()
        assert transitions.success == TransitionStatus(
            session=SessionStatus.TERMINATING,
            kernel=None,
            status_info=EXPIRY_REASON,
        )
        assert transitions.need_retry is None
        assert transitions.expired is None
        assert transitions.give_up is None
        assert EXPIRY_REASON == "bootstrap-timeout"

    # ------------------------------------------------------------------ bounds

    @pytest.mark.parametrize(
        ("status", "kernel_status"),
        [
            (SessionStatus.SCHEDULED, KernelStatus.SCHEDULED),
            (SessionStatus.PREPARING, KernelStatus.PREPARING),
            (SessionStatus.PULLING, KernelStatus.PREPARING),
            (SessionStatus.PREPARED, KernelStatus.PREPARED),
            (SessionStatus.CREATING, KernelStatus.CREATING),
        ],
    )
    async def test_expires_one_second_past_the_bound_not_at_it(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
        status: SessionStatus,
        kernel_status: KernelStatus,
    ) -> None:
        at_bound = _stuck(session_factory, status, [kernel_status], SESSION_CREATING_EXPIRY_SEC)
        past = _stuck(session_factory, status, [kernel_status], SESSION_CREATING_EXPIRY_SEC + 1)

        result = await handler.execute(_RG, [at_bound, past])

        assert [s.session_id for s in result.successes] == _ids([past])
        info = result.successes[0]
        assert info.from_status == status
        assert info.reason == EXPIRY_REASON
        assert info.creation_id == past.session_info.identity.creation_id
        assert info.access_key == past.session_info.metadata.access_key
        assert result.failures == []
        assert result.skipped == []

    async def test_scheduled_expires_past_the_bound(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
    ) -> None:
        session = _stuck(session_factory, SessionStatus.SCHEDULED, [KernelStatus.SCHEDULED], 3600)

        result = await handler.execute(_RG, [session])

        assert [s.session_id for s in result.successes] == _ids([session])

    async def test_preparing_with_only_pending_kernels_expires(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
    ) -> None:
        """Agentless PENDING kernels (e.g. reset under a PREPARING session) still expire it."""
        session = _stuck(
            session_factory,
            SessionStatus.PREPARING,
            [KernelStatus.PENDING, KernelStatus.PENDING],
            SESSION_CREATING_EXPIRY_SEC + 1,
            agentless=True,
        )

        result = await handler.execute(_RG, [session])

        assert [s.session_id for s in result.successes] == _ids([session])

    @pytest.mark.parametrize("status", [SessionStatus.PREPARING, SessionStatus.PULLING])
    async def test_a_pulling_kernel_extends_the_bound(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
        status: SessionStatus,
    ) -> None:
        kernels = [KernelStatus.PULLING, KernelStatus.PREPARED]
        short = _stuck(session_factory, status, kernels, SESSION_CREATING_EXPIRY_SEC + 1)
        at_bound = _stuck(session_factory, status, kernels, SESSION_PULLING_EXPIRY_SEC)
        past = _stuck(session_factory, status, kernels, SESSION_PULLING_EXPIRY_SEC + 1)

        result = await handler.execute(_RG, [short, at_bound, past])

        assert [s.session_id for s in result.successes] == _ids([past])

    @pytest.mark.parametrize("status", [SessionStatus.PREPARING, SessionStatus.PULLING])
    async def test_without_a_pulling_kernel_the_short_bound_applies(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
        status: SessionStatus,
    ) -> None:
        kernels = [KernelStatus.PREPARING, KernelStatus.PREPARED]
        at_bound = _stuck(session_factory, status, kernels, SESSION_CREATING_EXPIRY_SEC)
        past = _stuck(session_factory, status, kernels, SESSION_CREATING_EXPIRY_SEC + 1)

        result = await handler.execute(_RG, [at_bound, past])

        assert [s.session_id for s in result.successes] == _ids([past])

    async def test_a_kernel_waiting_on_another_sessions_pull_gets_the_pull_bound(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
    ) -> None:
        """The agent pulls an image once and reports the pull to the first session only: a
        session placed on the same agent for the same image waits in PREPARING with its kernel
        short of PULLING for as long as the pull takes."""
        image_id, other_image_id = uuid.uuid4(), uuid.uuid4()

        def on(session: SessionWithKernels, agent: str, image: uuid.UUID) -> SessionWithKernels:
            kernels = [
                dataclasses.replace(
                    k,
                    image=dataclasses.replace(k.image, image_id=image),
                    resource=dataclasses.replace(k.resource, agent=agent),
                )
                for k in session.kernel_infos
            ]
            return dataclasses.replace(session, kernel_infos=kernels)

        puller = on(
            _stuck(session_factory, SessionStatus.PREPARING, [KernelStatus.PULLING], 60),
            "i-a",
            image_id,
        )
        waiter, waiter_past, other_agent, other_image = (
            on(
                _stuck(session_factory, SessionStatus.PREPARING, [KernelStatus.SCHEDULED], age),
                agent,
                image,
            )
            for age, agent, image in (
                (SESSION_CREATING_EXPIRY_SEC + 1, "i-a", image_id),
                (SESSION_PULLING_EXPIRY_SEC + 1, "i-a", image_id),
                (SESSION_CREATING_EXPIRY_SEC + 1, "i-b", image_id),
                (SESSION_CREATING_EXPIRY_SEC + 1, "i-a", other_image_id),
            )
        )

        result = await handler.execute(_RG, [puller, waiter, waiter_past, other_agent, other_image])

        assert sorted(s.session_id for s in result.successes) == sorted(
            _ids([waiter_past, other_agent, other_image])
        )

    async def test_stuck_prepared_with_all_kernels_running_expires(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
    ) -> None:
        session = _stuck(
            session_factory,
            SessionStatus.PREPARED,
            [KernelStatus.RUNNING, KernelStatus.RUNNING],
            SESSION_CREATING_EXPIRY_SEC + 1,
        )

        result = await handler.execute(_RG, [session])

        assert [s.session_id for s in result.successes] == _ids([session])

    async def test_healthy_sessions_awaiting_promotion_are_untouched(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
    ) -> None:
        """A start that already finished on every kernel is the promotion's to advance."""
        long_ago = SESSION_PULLING_EXPIRY_SEC * 10
        creating_all_running = _stuck(
            session_factory,
            SessionStatus.CREATING,
            [KernelStatus.RUNNING, KernelStatus.RUNNING],
            long_ago,
        )
        creating_partly_terminated = _stuck(
            session_factory,
            SessionStatus.CREATING,
            [KernelStatus.RUNNING, KernelStatus.TERMINATED],
            long_ago,
        )
        preparing_all_prepared = _stuck(
            session_factory,
            SessionStatus.PREPARING,
            [KernelStatus.PREPARED, KernelStatus.PREPARED],
            long_ago,
        )

        result = await handler.execute(
            _RG, [creating_all_running, creating_partly_terminated, preparing_all_prepared]
        )

        assert result.successes == []
        assert result.failures == []
        assert result.skipped == []

    async def test_young_sessions_are_untouched(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
    ) -> None:
        sessions = [
            _stuck(session_factory, SessionStatus.SCHEDULED, [KernelStatus.SCHEDULED], 1),
            _stuck(session_factory, SessionStatus.CREATING, [KernelStatus.CREATING], 600),
            _stuck(session_factory, SessionStatus.PULLING, [KernelStatus.PULLING], 7200),
        ]

        result = await handler.execute(_RG, sessions)

        assert result.successes == []

    async def test_age_is_measured_on_the_database_clock(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        repository: AsyncMock,
        session_factory: SessionFactory,
    ) -> None:
        session = _stuck(
            session_factory,
            SessionStatus.CREATING,
            [KernelStatus.CREATING],
            SESSION_CREATING_EXPIRY_SEC + 1,
        )
        # The database clock is behind the session's age: not expired yet.
        repository.get_db_now.return_value = _NOW - timedelta(seconds=2)

        result = await handler.execute(_RG, [session])

        assert result.successes == []
        repository.get_db_now.assert_awaited_once()

    async def test_a_session_without_a_timestamp_for_its_status_is_left_alone(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        session = _stuck(session_factory, SessionStatus.CREATING, [KernelStatus.CREATING], 10_000)
        # Only an older status's timestamp: the age in CREATING is unknown.
        session.session_info.lifecycle.status_history = {
            SessionStatus.PREPARED.name: (_NOW - timedelta(days=1)).isoformat()
        }

        with caplog.at_level(logging.WARNING):
            result = await handler.execute(_RG, [session])

        assert result.successes == []
        assert str(session.session_info.identity.id) in caplog.text

    async def test_other_statuses_are_ignored_and_no_clock_is_read(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        repository: AsyncMock,
        session_factory: SessionFactory,
    ) -> None:
        """Only the starting family is expired; another status reaching execute() is not.

        (TERMINATING sessions go to the release branch, which reports nothing.)"""
        sessions = [
            _stuck(session_factory, status, [KernelStatus.RUNNING], 10**6)
            for status in (
                SessionStatus.RUNNING,
                SessionStatus.PENDING,
            )
        ]

        result = await handler.execute(_RG, sessions)

        assert result.successes == []
        repository.get_db_now.assert_not_awaited()

    async def test_expiry_logs_a_warning_naming_session_status_age_and_agents(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        session = _stuck(
            session_factory,
            SessionStatus.CREATING,
            [KernelStatus.CREATING, KernelStatus.PREPARED],
            1000,
        )

        with caplog.at_level(logging.WARNING):
            await handler.execute(_RG, [session])

        records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(records) == 1
        message = records[0].getMessage()
        assert str(session.session_info.identity.id) in message
        assert "CREATING" in message
        assert "1000s" in message
        for kernel in session.kernel_infos:
            assert str(kernel.resource.agent) in message

    # ------------------------------------------------------------------ scenarios

    async def test_silent_agent_failure_expires_at_the_bound(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
    ) -> None:
        """The agent took create_kernels and then failed without an event.

        The kernel stays CREATING (or PREPARED, if no event arrived at all) and nothing else
        advances the session: the expiry ends it one second past the bound, not before.
        """
        kernel_statuses = [[KernelStatus.CREATING], [KernelStatus.PREPARED]]
        before = [
            _stuck(session_factory, SessionStatus.CREATING, ks, SESSION_CREATING_EXPIRY_SEC)
            for ks in kernel_statuses
        ]
        after = [
            dataclasses.replace(
                s,
                session_info=dataclasses.replace(
                    s.session_info,
                    lifecycle=dataclasses.replace(
                        s.session_info.lifecycle,
                        status_history={
                            SessionStatus.CREATING.name: (
                                _NOW - timedelta(seconds=SESSION_CREATING_EXPIRY_SEC + 1)
                            ).isoformat()
                        },
                    ),
                ),
            )
            for s in before
        ]

        assert (await handler.execute(_RG, before)).successes == []
        result = await handler.execute(_RG, after)

        assert [s.session_id for s in result.successes] == _ids(after)
        assert all(s.from_status == SessionStatus.CREATING for s in result.successes)

    async def test_pre_dispatch_failure_expires(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
    ) -> None:
        """START failed before dispatching (e.g. the image config was missing).

        START still moved the session to CREATING, but no agent ever received its kernels,
        which stay PREPARED on an agent: the expiry ends the session.
        """
        session = _stuck(
            session_factory,
            SessionStatus.CREATING,
            [KernelStatus.PREPARED, KernelStatus.PREPARED],
            SESSION_CREATING_EXPIRY_SEC + 1,
        )
        assert all(k.resource.agent for k in session.kernel_infos)

        result = await handler.execute(_RG, [session])

        assert [s.session_id for s in result.successes] == _ids([session])

    async def test_mixed_batch_reports_only_the_expired(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        session_factory: SessionFactory,
    ) -> None:
        expired = [
            _stuck(session_factory, SessionStatus.SCHEDULED, [KernelStatus.SCHEDULED], 901),
            _stuck(session_factory, SessionStatus.PULLING, [KernelStatus.PULLING], 8101),
        ]
        alive = [
            _stuck(session_factory, SessionStatus.PULLING, [KernelStatus.PULLING], 901),
            _stuck(session_factory, SessionStatus.CREATING, [KernelStatus.RUNNING], 901),
        ]

        result = await handler.execute(_RG, [alive[0], expired[0], alive[1], expired[1]])

        assert sorted(s.session_id for s in result.successes) == sorted(_ids(expired))


class TestExpireStuckStartsInTheCoordinator:
    """A real ScheduleCoordinator drives the handler; the repository is mocked."""

    @pytest.fixture
    def repository(self) -> AsyncMock:
        repository = AsyncMock()
        repository.get_all_resource_groups = AsyncMock(return_value=[_RG])
        repository.get_last_session_histories = AsyncMock(return_value={})
        repository.get_db_now = AsyncMock(return_value=_NOW)
        return repository

    @pytest.fixture
    def scheduling_controller(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def event_producer(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def lock_factory(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def coordinator(
        self,
        repository: AsyncMock,
        scheduling_controller: AsyncMock,
        event_producer: AsyncMock,
        lock_factory: MagicMock,
    ) -> ScheduleCoordinator:
        return ScheduleCoordinator(
            valkey_schedule=AsyncMock(),
            components=SchedulerComponents(
                provisioner=MagicMock(),
                launcher=MagicMock(),
                terminator=MagicMock(),
                repository=repository,
                config_provider=MagicMock(),
                hook_registry=MagicMock(),
            ),
            handlers=CoordinatorHandlers(
                lifecycle_handlers={
                    ScheduleType.EXPIRE_STUCK_STARTS: ExpireStuckStartsLifecycleHandler(
                        repository, scheduling_controller
                    ),
                },
                promotion_specs={},
                kernel_handlers={},
                kernel_observers={},
                cleanup_handlers={},
            ),
            scheduling_controller=scheduling_controller,
            event_producer=event_producer,
            lock_factory=lock_factory,
        )

    async def test_expired_sessions_move_to_terminating_with_the_reason(
        self,
        coordinator: ScheduleCoordinator,
        repository: AsyncMock,
        scheduling_controller: AsyncMock,
        event_producer: AsyncMock,
        lock_factory: MagicMock,
        session_factory: SessionFactory,
    ) -> None:
        stuck = _stuck(
            session_factory,
            SessionStatus.CREATING,
            [KernelStatus.PREPARED],
            SESSION_CREATING_EXPIRY_SEC + 1,
        )
        healthy = _stuck(session_factory, SessionStatus.CREATING, [KernelStatus.CREATING], 60)
        repository.get_sessions_for_handler = AsyncMock(return_value=[stuck, healthy])
        stuck_id = stuck.session_info.identity.id
        repository.update_with_history = AsyncMock(return_value=[stuck_id])

        assert await coordinator.process_lifecycle_schedule(ScheduleType.EXPIRE_STUCK_STARTS)

        # Every starting status is read, regardless of the kernels' statuses.
        _, statuses, kernel_statuses = repository.get_sessions_for_handler.await_args.args
        assert set(statuses) == set(ExpireStuckStartsLifecycleHandler.target_statuses())
        assert kernel_statuses is None
        # No lock is taken.
        lock_factory.assert_not_called()
        # Only the stuck session moves, to TERMINATING, with the expiry's reason.
        repository.update_with_history.assert_awaited_once()
        updater, history = repository.update_with_history.await_args.args
        assert isinstance(updater.spec, SessionStatusBatchUpdaterSpec)
        assert updater.spec.to_status == SessionStatus.TERMINATING
        assert updater.spec.build_values()["status_info"] == EXPIRY_REASON
        assert [(h.session_id, h.from_status, h.to_status) for h in history.specs] == [
            (stuck_id, SessionStatus.CREATING, SessionStatus.TERMINATING)
        ]
        # TERMINATE is marked to destroy its kernels and free its resources.
        marked = scheduling_controller.mark_scheduling_needed.await_args.args[0]
        assert ScheduleType.TERMINATE in marked
        # The broadcast carries the reason.
        (events,) = event_producer.broadcast_events_batch.await_args.args
        assert [(e.session_id, e.reason) for e in events] == [(stuck_id, EXPIRY_REASON)]

    async def test_nothing_expired_writes_nothing(
        self,
        coordinator: ScheduleCoordinator,
        repository: AsyncMock,
        scheduling_controller: AsyncMock,
        session_factory: SessionFactory,
    ) -> None:
        healthy = _stuck(session_factory, SessionStatus.PULLING, [KernelStatus.PULLING], 7200)
        repository.get_sessions_for_handler = AsyncMock(return_value=[healthy])

        assert await coordinator.process_lifecycle_schedule(ScheduleType.EXPIRE_STUCK_STARTS)

        repository.update_with_history.assert_not_awaited()
        repository.create_scheduling_history.assert_not_awaited()
        scheduling_controller.mark_scheduling_needed.assert_not_awaited()


class TestExpireStuckStartsWiring:
    def test_task_spec_runs_on_the_long_cycle_only_after_a_grace_period(self) -> None:
        specs = [
            spec
            for spec in ScheduleCoordinator._create_task_specs()
            if spec.schedule_type == ScheduleType.EXPIRE_STUCK_STARTS
        ]
        assert len(specs) == 1
        (spec,) = specs
        assert spec.short_interval is None
        assert spec.long_interval == 60.0
        assert spec.initial_delay == 300.0

    def test_factory_registers_the_handler(self) -> None:
        args: Any = MagicMock()
        handlers = _create_lifecycle_handlers(args)
        handler = handlers[ScheduleType.EXPIRE_STUCK_STARTS]
        assert isinstance(handler, ExpireStuckStartsLifecycleHandler)
        assert handler.name() == "expire-stuck-starts"


class TestScheduleMarking:
    @pytest.mark.parametrize(
        ("status", "schedule_type"),
        [
            (SessionStatus.PREPARING, ScheduleType.CHECK_PULLING_PROGRESS),
            (SessionStatus.CREATING, ScheduleType.CHECK_CREATING_PROGRESS),
            (SessionStatus.TERMINATING, ScheduleType.TERMINATE),
        ],
    )
    async def test_a_transition_marks_the_next_check(
        self,
        status: SessionStatus,
        schedule_type: ScheduleType,
    ) -> None:
        """A kernel event that arrived before its session's transition found nothing to
        promote; the transition's own mark makes the check run again at once."""
        controller = AsyncMock()
        processor = ScheduleMarkingPostProcessor(controller)

        await processor.execute(
            PostProcessorContext(result=SessionExecutionResult(), target_statuses={status})
        )

        controller.mark_scheduling_needed.assert_awaited_once_with([schedule_type])
