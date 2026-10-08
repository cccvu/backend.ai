"""Agent RPCs awaited by SessionTerminator are bounded.

The scheduler awaits these calls while holding a global lock, so an agent
that accepts a call and never replies must not stall the sweep or the
termination of sessions on other agents.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai.backend.common.types import AgentId, KernelId, SessionId
from ai.backend.manager.data.kernel.types import KernelInfo
from ai.backend.manager.sokovan.recorder import RecorderContext
from ai.backend.manager.sokovan.scheduler.terminator import terminator as terminator_module
from ai.backend.manager.sokovan.scheduler.terminator.terminator import (
    SessionTerminator,
    SessionTerminatorArgs,
)
from ai.backend.manager.views.sokovan.session import (
    TerminatingKernelData,
    TerminatingSessionData,
)

HUNG_AGENT = AgentId("agent-hung")
HEALTHY_AGENT = AgentId("agent-healthy")
RPC_TIMEOUT = 0.1
# Generous upper bound: a few timeouts' worth, far below the serialised total.
BOUNDED = 1.0


async def _hang(*args: Any, **kwargs: Any) -> Any:
    await asyncio.Event().wait()


@pytest.fixture
def valkey_schedule() -> AsyncMock:
    client = AsyncMock()
    # No presence record: every kernel is a stale candidate.
    client.check_kernel_presence_status_batch = AsyncMock(return_value={})
    return client


@pytest.fixture
def terminator(per_agent_client_pool: MagicMock, valkey_schedule: AsyncMock) -> SessionTerminator:
    return SessionTerminator(
        SessionTerminatorArgs(
            repository=AsyncMock(),
            agent_client_pool=per_agent_client_pool,
            valkey_schedule=valkey_schedule,
        )
    )


@pytest.fixture(autouse=True)
def short_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(terminator_module, "AGENT_CHECK_RUNNING_TIMEOUT_SEC", RPC_TIMEOUT)
    monkeypatch.setattr(terminator_module, "AGENT_DESTROY_KERNEL_TIMEOUT_SEC", RPC_TIMEOUT)


class TestStaleKernelSweepTimeouts:
    async def test_hung_agent_does_not_block_other_agents(
        self,
        terminator: SessionTerminator,
        per_agent_client_pool: MagicMock,
        kernel_info_factory: Callable[..., KernelInfo],
    ) -> None:
        hung_kernel = kernel_info_factory(agent_id=HUNG_AGENT)
        dead_kernel = kernel_info_factory(agent_id=HEALTHY_AGENT)
        per_agent_client_pool.client(HUNG_AGENT).check_running.side_effect = _hang
        per_agent_client_pool.client(HEALTHY_AGENT).check_running.return_value = False

        started = time.monotonic()
        result = await terminator.check_stale_kernels([hung_kernel, dead_kernel])
        elapsed = time.monotonic() - started

        assert result == [KernelId(dead_kernel.id)]
        assert elapsed < BOUNDED

    async def test_many_kernels_on_hung_agent_cost_one_timeout(
        self,
        terminator: SessionTerminator,
        per_agent_client_pool: MagicMock,
        kernel_info_factory: Callable[..., KernelInfo],
    ) -> None:
        kernels = [kernel_info_factory(agent_id=HUNG_AGENT) for _ in range(20)]
        per_agent_client_pool.client(HUNG_AGENT).check_running.side_effect = _hang

        started = time.monotonic()
        result = await terminator.check_stale_kernels(kernels)
        elapsed = time.monotonic() - started

        assert result == []
        assert per_agent_client_pool.client(HUNG_AGENT).check_running.await_count == 20
        # Sequential checks would take 20 * RPC_TIMEOUT = 2.0s.
        assert elapsed < BOUNDED

    async def test_cancelled_error_from_client_is_skipped(
        self,
        terminator: SessionTerminator,
        per_agent_client_pool: MagicMock,
        kernel_info_factory: Callable[..., KernelInfo],
    ) -> None:
        cancelled_kernel = kernel_info_factory(agent_id=HUNG_AGENT)
        dead_kernel = kernel_info_factory(agent_id=HEALTHY_AGENT)
        per_agent_client_pool.client(
            HUNG_AGENT
        ).check_running.side_effect = asyncio.CancelledError()
        per_agent_client_pool.client(HEALTHY_AGENT).check_running.return_value = False

        result = await terminator.check_stale_kernels([cancelled_kernel, dead_kernel])

        assert result == [KernelId(dead_kernel.id)]

    async def test_outer_cancellation_propagates(
        self,
        terminator: SessionTerminator,
        per_agent_client_pool: MagicMock,
        kernel_info_factory: Callable[..., KernelInfo],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(terminator_module, "AGENT_CHECK_RUNNING_TIMEOUT_SEC", 30.0)
        kernels = [
            kernel_info_factory(agent_id=HUNG_AGENT),
            kernel_info_factory(agent_id=HEALTHY_AGENT),
        ]
        per_agent_client_pool.client(HUNG_AGENT).check_running.side_effect = _hang
        per_agent_client_pool.client(HEALTHY_AGENT).check_running.side_effect = _hang

        task = asyncio.create_task(terminator.check_stale_kernels(kernels))
        await asyncio.sleep(0.05)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()

    async def test_next_sweep_recovers_once_agent_answers(
        self,
        terminator: SessionTerminator,
        per_agent_client_pool: MagicMock,
        kernel_info_factory: Callable[..., KernelInfo],
    ) -> None:
        kernel = kernel_info_factory(agent_id=HUNG_AGENT)
        per_agent_client_pool.client(HUNG_AGENT).check_running.side_effect = _hang

        assert await terminator.check_stale_kernels([kernel]) == []

        per_agent_client_pool.client(HUNG_AGENT).check_running.side_effect = None
        per_agent_client_pool.client(HUNG_AGENT).check_running.return_value = False

        assert await terminator.check_stale_kernels([kernel]) == [KernelId(kernel.id)]

    async def test_only_explicit_false_marks_dead(
        self,
        terminator: SessionTerminator,
        per_agent_client_pool: MagicMock,
        kernel_info_factory: Callable[..., KernelInfo],
    ) -> None:
        running_kernel = kernel_info_factory(agent_id=HUNG_AGENT)
        unknown_kernel = kernel_info_factory(agent_id=HEALTHY_AGENT)
        per_agent_client_pool.client(HUNG_AGENT).check_running.return_value = True
        per_agent_client_pool.client(HEALTHY_AGENT).check_running.return_value = None

        assert await terminator.check_stale_kernels([running_kernel, unknown_kernel]) == []


class TestTerminationTimeouts:
    @pytest.fixture
    def session_on(
        self,
        terminating_kernel_factory: Callable[..., TerminatingKernelData],
        terminating_session_factory: Callable[..., TerminatingSessionData],
    ) -> Callable[..., TerminatingSessionData]:
        """Build a session whose kernels sit on the given agents (None: no agent)."""

        def build(*agent_ids: AgentId | None) -> TerminatingSessionData:
            kernels = []
            for agent_id in agent_ids:
                kernel = terminating_kernel_factory()
                kernel.agent_id = agent_id
                kernels.append(kernel)
            return terminating_session_factory(kernels=kernels)

        return build

    async def test_hung_destroy_is_bounded_and_excluded(
        self,
        terminator: SessionTerminator,
        per_agent_client_pool: MagicMock,
        session_on: Callable[..., TerminatingSessionData],
    ) -> None:
        hung_session = session_on(HUNG_AGENT)
        healthy_session = session_on(HEALTHY_AGENT)
        per_agent_client_pool.client(HUNG_AGENT).destroy_kernel.side_effect = _hang

        session_ids = [hung_session.session_id, healthy_session.session_id]
        started = time.monotonic()
        with RecorderContext[SessionId].scope("terminate", entity_ids=session_ids):
            succeeded = await terminator.terminate_sessions_for_handler([
                hung_session,
                healthy_session,
            ])
        elapsed = time.monotonic() - started

        assert succeeded == [healthy_session.session_id]
        per_agent_client_pool.client(HEALTHY_AGENT).destroy_kernel.assert_awaited_once()
        assert elapsed < BOUNDED

    async def test_hung_destroy_maps_to_failed_result(
        self,
        terminator: SessionTerminator,
        per_agent_client_pool: MagicMock,
        session_on: Callable[..., TerminatingSessionData],
    ) -> None:
        session = session_on(HUNG_AGENT)
        kernel = session.kernels[0]
        per_agent_client_pool.client(HUNG_AGENT).destroy_kernel.side_effect = _hang

        result = await terminator._terminate_kernel(
            HUNG_AGENT,
            kernel.kernel_id,
            session.session_id,
            session.status_info,
            kernel.occupied_slots,
        )

        assert result.success is False
        assert result.error is not None
        assert "timed out" in result.error

    async def test_session_succeeds_only_when_all_agent_kernels_succeed(
        self,
        terminator: SessionTerminator,
        per_agent_client_pool: MagicMock,
        session_on: Callable[..., TerminatingSessionData],
    ) -> None:
        split_session = session_on(HUNG_AGENT, HEALTHY_AGENT)
        per_agent_client_pool.client(HUNG_AGENT).destroy_kernel.side_effect = _hang

        with RecorderContext[SessionId].scope("terminate", entity_ids=[split_session.session_id]):
            succeeded = await terminator.terminate_sessions_for_handler([split_session])

        assert succeeded == []
        per_agent_client_pool.client(HEALTHY_AGENT).destroy_kernel.assert_awaited_once()

    async def test_agentless_kernels_do_not_count_against_session(
        self,
        terminator: SessionTerminator,
        per_agent_client_pool: MagicMock,
        session_on: Callable[..., TerminatingSessionData],
    ) -> None:
        agentless_session = session_on(None)
        mixed_session = session_on(HEALTHY_AGENT, None)

        session_ids = [agentless_session.session_id, mixed_session.session_id]
        with RecorderContext[SessionId].scope("terminate", entity_ids=session_ids):
            succeeded = await terminator.terminate_sessions_for_handler([
                agentless_session,
                mixed_session,
            ])

        assert succeeded == session_ids
        per_agent_client_pool.client(HEALTHY_AGENT).destroy_kernel.assert_awaited_once()
