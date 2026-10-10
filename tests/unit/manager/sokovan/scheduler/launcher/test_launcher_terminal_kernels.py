"""Kernels that reach a terminal status while being created are destroyed on their agents.

A session can be terminated while its kernel creation is in flight. Once every agent's
create_kernels call has ended, the launcher re-reads the kernels' statuses and sends a
bounded destroy_kernel for each terminal one to the agent it was created on.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from ai.backend.common.types import AgentId, KernelId, SessionId
from ai.backend.manager.data.kernel.types import KernelInfo, KernelListResult, KernelStatus
from ai.backend.manager.sokovan.recorder import RecorderContext
from ai.backend.manager.sokovan.scheduler.launcher import launcher as launcher_module
from ai.backend.manager.sokovan.scheduler.launcher.launcher import SessionLauncher
from ai.backend.manager.views.sokovan.image import ImageConfigData
from ai.backend.manager.views.sokovan.lifecycle import SessionDataForStart

# Far below AGENT_CREATE_KERNELS_TIMEOUT_SEC and AGENT_DESTROY_KERNEL_TIMEOUT_SEC.
BOUND = 2.0

AGENT_1 = AgentId("agent-1")
AGENT_2 = AgentId("agent-2")


@pytest.fixture
def mock_agent_client_pool(per_agent_client_pool: MagicMock) -> MagicMock:
    """Overrides the shared single-client pool with one stub client per agent."""
    return per_agent_client_pool


def _kernel(
    kernel_id: KernelId, status: KernelStatus, status_info: str | None = None
) -> KernelInfo:
    kernel = MagicMock()
    kernel.id = kernel_id
    kernel.lifecycle.status = status
    kernel.lifecycle.status_info = status_info
    return kernel


def _kernels(*kernels: KernelInfo) -> KernelListResult:
    return KernelListResult(
        items=list(kernels),
        total_count=len(kernels),
        has_next_page=False,
        has_previous_page=False,
    )


async def _start_and_drain(
    launcher: SessionLauncher,
    session: SessionDataForStart,
    image_configs: dict[UUID, ImageConfigData],
    drain: Callable[[SessionLauncher], Awaitable[None]],
) -> asyncio.Task[None]:
    async with asyncio.timeout(BOUND):
        with RecorderContext[SessionId].scope("test", entity_ids=[session.session_id]):
            await launcher.start_sessions_for_handler([session], image_configs)
        (task,) = launcher._kernel_creations
        await drain(launcher)
    return task


class TestDestroyKernelsTerminatedDuringCreation:
    async def test_terminal_after_successful_reply_is_destroyed(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        drain: Callable[[SessionLauncher], Awaitable[None]],
    ) -> None:
        session = session_for_start_single_kernel
        kernel_id = session.kernels[0].kernel_id
        mock_repository.search_kernels_for_handler.return_value = _kernels(
            _kernel(kernel_id, KernelStatus.CREATING)
        )

        async def terminated_while_creating(*args: Any, **kwargs: Any) -> None:
            # The session is terminated before the agent replies.
            mock_repository.search_kernels_for_handler.return_value = _kernels(
                _kernel(kernel_id, KernelStatus.TERMINATED, "user-requested")
            )

        client = per_agent_client_pool.client(AGENT_1)
        client.create_kernels.side_effect = terminated_while_creating

        task = await _start_and_drain(launcher, session, image_config_default, drain)

        assert task.exception() is None
        mock_repository.search_kernels_for_handler.assert_awaited_once()
        client.destroy_kernel.assert_awaited_once_with(
            kernel_id, session.session_id, "user-requested", suppress_events=True
        )

    async def test_terminal_after_failed_reply_is_destroyed(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        drain: Callable[[SessionLauncher], Awaitable[None]],
    ) -> None:
        session = session_for_start_single_kernel
        kernel_id = session.kernels[0].kernel_id
        mock_repository.search_kernels_for_handler.return_value = _kernels(
            _kernel(kernel_id, KernelStatus.CANCELLED)
        )
        client = per_agent_client_pool.client(AGENT_1)
        client.create_kernels.side_effect = ConnectionError("lost")

        task = await _start_and_drain(launcher, session, image_config_default, drain)

        assert task.exception() is None
        mock_valkey_schedule.record_session_failed_agents.assert_awaited_once_with(
            session.session_id, [AGENT_1]
        )
        # No recognized reason recorded on the kernel: a generic one is sent.
        client.destroy_kernel.assert_awaited_once_with(
            kernel_id, session.session_id, "unknown", suppress_events=True
        )

    @pytest.mark.parametrize(
        "status",
        [
            KernelStatus.PREPARED,
            KernelStatus.CREATING,
            KernelStatus.RUNNING,
            # Still being terminated: the terminator sends its own destroy.
            KernelStatus.TERMINATING,
        ],
    )
    async def test_non_terminal_is_left_alone(
        self,
        status: KernelStatus,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        drain: Callable[[SessionLauncher], Awaitable[None]],
    ) -> None:
        session = session_for_start_single_kernel
        mock_repository.search_kernels_for_handler.return_value = _kernels(
            _kernel(session.kernels[0].kernel_id, status)
        )

        await _start_and_drain(launcher, session, image_config_default, drain)

        mock_repository.search_kernels_for_handler.assert_awaited_once()
        per_agent_client_pool.client(AGENT_1).destroy_kernel.assert_not_awaited()

    async def test_hung_destroy_is_bounded_and_logged(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        drain: Callable[[SessionLauncher], Awaitable[None]],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(launcher_module, "AGENT_DESTROY_KERNEL_TIMEOUT_SEC", 0.05)
        session = session_for_start_single_kernel
        kernel_id = session.kernels[0].kernel_id
        mock_repository.search_kernels_for_handler.return_value = _kernels(
            _kernel(kernel_id, KernelStatus.TERMINATED)
        )

        async def hang(*args: Any, **kwargs: Any) -> None:
            await asyncio.Event().wait()

        client = per_agent_client_pool.client(AGENT_1)
        client.destroy_kernel.side_effect = hang

        task = await _start_and_drain(launcher, session, image_config_default, drain)

        assert task.exception() is None
        client.destroy_kernel.assert_awaited_once()
        assert f"failed to destroy kernel {kernel_id} on agent agent-1: TimeoutError()" in (
            caplog.text
        )

    async def test_failed_status_read_is_logged_without_destroying(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        drain: Callable[[SessionLauncher], Awaitable[None]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_repository.search_kernels_for_handler.side_effect = RuntimeError("db down")

        task = await _start_and_drain(
            launcher, session_for_start_single_kernel, image_config_default, drain
        )

        assert task.exception() is None
        assert "failed to re-read kernel statuses after creation" in caplog.text
        assert "db down" in caplog.text
        per_agent_client_pool.client(AGENT_1).destroy_kernel.assert_not_awaited()

    async def test_multi_agent_destroys_only_terminal_kernels_on_their_agents(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        session_for_start_multi_node: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        drain: Callable[[SessionLauncher], Awaitable[None]],
    ) -> None:
        session = session_for_start_multi_node
        kernel_on = {k.agent_id: k.kernel_id for k in session.kernels}
        mock_repository.search_kernels_for_handler.return_value = _kernels(
            _kernel(kernel_on[AGENT_1], KernelStatus.RUNNING),
            _kernel(kernel_on[AGENT_2], KernelStatus.ERROR, "failed-to-start"),
        )

        await _start_and_drain(launcher, session, image_config_default, drain)

        per_agent_client_pool.client(AGENT_1).destroy_kernel.assert_not_awaited()
        per_agent_client_pool.client(AGENT_2).destroy_kernel.assert_awaited_once_with(
            kernel_on[AGENT_2], session.session_id, "failed-to-start", suppress_events=True
        )

    async def test_multi_agent_destroys_concurrently_each_on_its_own_agent(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        session_for_start_multi_node: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        drain: Callable[[SessionLauncher], Awaitable[None]],
    ) -> None:
        session = session_for_start_multi_node
        kernel_on = {k.agent_id: k.kernel_id for k in session.kernels}
        mock_repository.search_kernels_for_handler.return_value = _kernels(
            _kernel(kernel_on[AGENT_1], KernelStatus.TERMINATED),
            _kernel(kernel_on[AGENT_2], KernelStatus.CANCELLED),
        )
        both_entered = asyncio.Barrier(2)

        async def wait_for_the_other(*args: Any, **kwargs: Any) -> None:
            # Returns only once both destroys are in flight at the same time.
            await both_entered.wait()

        for agent_id in (AGENT_1, AGENT_2):
            per_agent_client_pool.client(agent_id).destroy_kernel.side_effect = wait_for_the_other

        await _start_and_drain(launcher, session, image_config_default, drain)

        for agent_id in (AGENT_1, AGENT_2):
            per_agent_client_pool.client(agent_id).destroy_kernel.assert_awaited_once_with(
                kernel_on[agent_id], session.session_id, "unknown", suppress_events=True
            )

    async def test_close_cancels_a_pending_destroy(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
    ) -> None:
        session = session_for_start_single_kernel
        mock_repository.search_kernels_for_handler.return_value = _kernels(
            _kernel(session.kernels[0].kernel_id, KernelStatus.TERMINATED)
        )
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def hang(*args: Any, **kwargs: Any) -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        per_agent_client_pool.client(AGENT_1).destroy_kernel.side_effect = hang

        async with asyncio.timeout(BOUND):
            with RecorderContext[SessionId].scope("test", entity_ids=[session.session_id]):
                await launcher.start_sessions_for_handler([session], image_config_default)
            await entered.wait()
            await launcher.close()

        assert cancelled.is_set()
        assert not launcher._kernel_creations
