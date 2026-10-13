"""Unit tests for the release of TERMINATING sessions held by an unavailable agent.

``ExpireStuckStartsLifecycleHandler`` fences the agent first and then terminates by force
only the sessions of agents the fence returned.
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai.backend.common.events.event_types.kernel.types import KernelLifecycleEventReason
from ai.backend.common.identifier.resource_group import ResourceGroupID
from ai.backend.common.types import AgentId, SessionId
from ai.backend.manager.data.agent.types import AgentStatus
from ai.backend.manager.data.kernel.types import KernelStatus
from ai.backend.manager.data.session.types import SessionStatus
from ai.backend.manager.defs import SESSION_CREATING_EXPIRY_SEC, UNAVAILABLE_AGENT_RELEASE_SEC
from ai.backend.manager.sokovan.scheduler.handlers.maintenance.expire_stuck_starts import (
    RELEASE_REASON,
    ExpireStuckStartsLifecycleHandler,
    sole_agent,
)
from ai.backend.manager.views.sokovan.agent import FencedAgent
from ai.backend.manager.views.sokovan.lifecycle import SessionWithKernels
from ai.backend.manager.views.sokovan.session import MarkTerminatingResult

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_RG = ResourceGroupID(uuid.uuid4())
_LOST_AT = _NOW - timedelta(seconds=UNAVAILABLE_AGENT_RELEASE_SEC + 60)
T = AgentId("agent-t")
P = AgentId("agent-p")

SessionFactory = Callable[..., SessionWithKernels]


def _session(
    session_factory: SessionFactory,
    kernels: Sequence[tuple[KernelStatus, AgentId | None]],
    *,
    status: SessionStatus = SessionStatus.TERMINATING,
    age: float = 60.0,
) -> SessionWithKernels:
    """A session in ``status`` whose kernels have the given statuses and agents."""
    session = session_factory(status=status, kernel_count=len(kernels))
    entered_at = _NOW - timedelta(seconds=age)
    lifecycle = dataclasses.replace(
        session.session_info.lifecycle,
        status_changed=entered_at,
        status_history={status.name: entered_at.isoformat()},
    )
    kernel_infos = [
        dataclasses.replace(
            kernel,
            lifecycle=dataclasses.replace(kernel.lifecycle, status=kernel_status),
            resource=dataclasses.replace(
                kernel.resource,
                agent=agent,
                agent_addr=f"tcp://{agent}:6001" if agent else None,
            ),
        )
        for kernel, (kernel_status, agent) in zip(session.kernel_infos, kernels, strict=True)
    ]
    return SessionWithKernels(
        session_info=dataclasses.replace(session.session_info, lifecycle=lifecycle),
        kernel_infos=kernel_infos,
    )


def _sid(session: SessionWithKernels) -> SessionId:
    return session.session_info.identity.id


def _fenced(*agent_ids: AgentId) -> list[FencedAgent]:
    return [
        FencedAgent(agent_id=agent_id, status=AgentStatus.LOST, lost_at=_LOST_AT)
        for agent_id in agent_ids
    ]


def _released(session_ids: Sequence[SessionId]) -> MarkTerminatingResult:
    return MarkTerminatingResult(
        cancelled_sessions=[],
        terminating_sessions=[],
        force_terminated_sessions=list(session_ids),
        skipped_sessions=[],
    )


class TestReleaseOnUnavailableAgents:
    @pytest.fixture
    def repository(self, mock_repository: AsyncMock) -> AsyncMock:
        mock_repository.get_db_now = AsyncMock(return_value=_NOW)
        mock_repository.fence_unavailable_agents = AsyncMock(return_value=[])
        return mock_repository

    @pytest.fixture
    def controller(self) -> AsyncMock:
        controller = AsyncMock()
        controller.mark_sessions_for_termination = AsyncMock(
            side_effect=lambda ids, **_: _released(ids)
        )
        return controller

    @pytest.fixture
    def handler(
        self, repository: AsyncMock, controller: AsyncMock
    ) -> ExpireStuckStartsLifecycleHandler:
        return ExpireStuckStartsLifecycleHandler(repository, controller)

    def test_release_reason_is_hang_timeout(self) -> None:
        assert RELEASE_REASON == KernelLifecycleEventReason.HANG_TIMEOUT.value == "hang-timeout"

    async def test_a_session_on_one_unavailable_agent_is_fenced_then_released(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        repository: AsyncMock,
        controller: AsyncMock,
        session_factory: SessionFactory,
    ) -> None:
        # Terminal kernels do not count; the open ones all sit on T.
        session = _session(
            session_factory,
            [
                (KernelStatus.RUNNING, T),
                (KernelStatus.TERMINATING, T),
                (KernelStatus.TERMINATED, P),
            ],
        )
        calls = MagicMock()
        repository.fence_unavailable_agents.side_effect = lambda *a, **k: (
            calls.fence(*a, **k),
            _fenced(T),
        )[1]
        controller.mark_sessions_for_termination.side_effect = lambda ids, **k: (
            calls.release(ids, **k),
            _released(ids),
        )[1]

        result = await handler.execute(_RG, [session])

        assert result.successes == []
        assert result.failures == []
        # The fence comes first, with the bound, and the release only after it.
        assert [c[0] for c in calls.mock_calls] == ["fence", "release"]
        repository.fence_unavailable_agents.assert_awaited_once_with(
            [T], _RG, UNAVAILABLE_AGENT_RELEASE_SEC
        )
        controller.mark_sessions_for_termination.assert_awaited_once_with(
            [_sid(session)], reason="hang-timeout", forced=True
        )

    @pytest.mark.parametrize(
        "kernels",
        [
            # An agentless PENDING kernel beside one on T.
            [(KernelStatus.RUNNING, T), (KernelStatus.PENDING, None)],
            # Only agentless kernels.
            [(KernelStatus.PENDING, None)],
            # Open kernels on two agents.
            [(KernelStatus.RUNNING, T), (KernelStatus.RUNNING, P)],
            # Every kernel terminal.
            [(KernelStatus.TERMINATED, T), (KernelStatus.CANCELLED, T), (KernelStatus.ERROR, T)],
        ],
        ids=["agentless-pending", "agentless-only", "two-agents", "all-terminal"],
    )
    async def test_sessions_that_do_not_qualify_never_reach_the_fence(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        repository: AsyncMock,
        controller: AsyncMock,
        session_factory: SessionFactory,
        kernels: list[tuple[KernelStatus, AgentId | None]],
    ) -> None:
        session = _session(session_factory, kernels)

        assert sole_agent(session) is None
        result = await handler.execute(_RG, [session])

        assert result.successes == []
        repository.fence_unavailable_agents.assert_not_awaited()
        controller.mark_sessions_for_termination.assert_not_awaited()

    async def test_only_sessions_of_fenced_agents_are_released(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        repository: AsyncMock,
        controller: AsyncMock,
        session_factory: SessionFactory,
    ) -> None:
        on_t = [_session(session_factory, [(KernelStatus.RUNNING, T)]) for _ in range(2)]
        on_p = _session(session_factory, [(KernelStatus.RUNNING, P)])
        skipped = _session(session_factory, [(KernelStatus.RUNNING, T), (KernelStatus.RUNNING, P)])
        repository.fence_unavailable_agents.return_value = _fenced(T)

        await handler.execute(_RG, [*on_t, on_p, skipped])

        # Each candidate agent is passed once; the session on two agents adds none.
        repository.fence_unavailable_agents.assert_awaited_once_with(
            [P, T], _RG, UNAVAILABLE_AGENT_RELEASE_SEC
        )
        controller.mark_sessions_for_termination.assert_awaited_once_with(
            [_sid(s) for s in on_t], reason=RELEASE_REASON, forced=True
        )

    async def test_nothing_fenced_releases_nothing(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        repository: AsyncMock,
        controller: AsyncMock,
        session_factory: SessionFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        session = _session(session_factory, [(KernelStatus.RUNNING, T)])
        repository.fence_unavailable_agents.return_value = []

        with caplog.at_level(logging.WARNING):
            await handler.execute(_RG, [session])

        repository.fence_unavailable_agents.assert_awaited_once()
        controller.mark_sessions_for_termination.assert_not_awaited()
        assert "fenced agent" not in caplog.text

    @pytest.mark.parametrize("outcome", ["released-none", "raises"])
    async def test_the_fence_is_logged_whatever_the_release_does(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        repository: AsyncMock,
        controller: AsyncMock,
        session_factory: SessionFactory,
        caplog: pytest.LogCaptureFixture,
        outcome: str,
    ) -> None:
        sessions = [_session(session_factory, [(KernelStatus.RUNNING, T)]) for _ in range(3)]
        repository.fence_unavailable_agents.return_value = _fenced(T)
        if outcome == "raises":
            controller.mark_sessions_for_termination.side_effect = RuntimeError("db down")
        else:
            controller.mark_sessions_for_termination.side_effect = None
            controller.mark_sessions_for_termination.return_value = _released([])

        with caplog.at_level(logging.WARNING):
            result = await handler.execute(_RG, sessions)

        assert result.successes == []
        fence_lines = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "fenced agent" in r.message
        ]
        assert len(fence_lines) == 1
        line = fence_lines[0].message
        assert T in line
        assert "LOST" in line
        assert _LOST_AT.isoformat() in line
        assert "force-releasing 3 terminating sessions" in line
        assert "Reconcile its containers before re-enabling it" in line

    async def test_a_failing_release_keeps_the_expiry_results(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        repository: AsyncMock,
        session_factory: SessionFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        stuck = _session(
            session_factory,
            [(KernelStatus.CREATING, T)],
            status=SessionStatus.CREATING,
            age=SESSION_CREATING_EXPIRY_SEC + 1,
        )
        terminating = _session(session_factory, [(KernelStatus.RUNNING, T)])
        repository.fence_unavailable_agents.side_effect = RuntimeError("db down")

        with caplog.at_level(logging.ERROR):
            result = await handler.execute(_RG, [stuck, terminating])

        assert [s.session_id for s in result.successes] == [_sid(stuck)]
        assert "failed to release terminating sessions" in caplog.text

    async def test_running_and_starting_sessions_are_never_fenced(
        self,
        handler: ExpireStuckStartsLifecycleHandler,
        repository: AsyncMock,
        controller: AsyncMock,
        session_factory: SessionFactory,
    ) -> None:
        sessions = [
            _session(session_factory, [(KernelStatus.RUNNING, T)], status=SessionStatus.RUNNING),
            _session(session_factory, [(KernelStatus.CREATING, T)], status=SessionStatus.CREATING),
        ]

        await handler.execute(_RG, sessions)

        repository.fence_unavailable_agents.assert_not_awaited()
        controller.mark_sessions_for_termination.assert_not_awaited()
