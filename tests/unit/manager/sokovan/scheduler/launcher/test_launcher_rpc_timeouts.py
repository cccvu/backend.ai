"""Agent RPCs awaited by SessionLauncher are bounded.

An agent that accepts a call and never replies must not stall the scheduler;
the timeout follows each call site's existing failure path.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from ai.backend.common.types import AgentId
from ai.backend.manager.sokovan.recorder import RecorderContext
from ai.backend.manager.sokovan.scheduler.launcher import launcher as launcher_module
from ai.backend.manager.sokovan.scheduler.launcher.launcher import SessionLauncher
from ai.backend.manager.views.sokovan.image import ImageConfigData
from ai.backend.manager.views.sokovan.lifecycle import (
    SessionDataForPull,
    SessionDataForStart,
)

RPC_TIMEOUT = 0.1
# Generous upper bound, far below any hang.
BOUNDED = 2.0


async def _hang(*args: Any, **kwargs: Any) -> Any:
    await asyncio.Event().wait()


@pytest.fixture
def mock_agent_client_pool(per_agent_client_pool: MagicMock) -> MagicMock:
    """Overrides the shared single-client pool with one stub client per agent."""
    return per_agent_client_pool


@pytest.fixture(autouse=True)
def short_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AGENT_CHECK_AND_PULL_TIMEOUT_SEC",
        "AGENT_CREATE_KERNELS_TIMEOUT_SEC",
        "AGENT_NETWORK_RPC_TIMEOUT_SEC",
        "AGENT_ASSIGN_PORT_TIMEOUT_SEC",
    ):
        monkeypatch.setattr(launcher_module, name, RPC_TIMEOUT)


class TestLauncherRPCTimeouts:
    async def test_check_and_pull_hang_does_not_block_other_agents(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        sessions_for_pull_multiple: list[SessionDataForPull],
        image_config_default: dict[UUID, ImageConfigData],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        per_agent_client_pool.client(AgentId("agent-1")).check_and_pull.side_effect = _hang
        per_agent_client_pool.client(AgentId("agent-2")).check_and_pull.return_value = {}

        session_ids = [s.session_id for s in sessions_for_pull_multiple]
        started = time.monotonic()
        with RecorderContext.scope("test", entity_ids=session_ids):
            await launcher.trigger_image_pulling(sessions_for_pull_multiple, image_config_default)
        elapsed = time.monotonic() - started

        per_agent_client_pool.client(AgentId("agent-1")).check_and_pull.assert_awaited_once()
        per_agent_client_pool.client(AgentId("agent-2")).check_and_pull.assert_awaited_once()
        assert elapsed < BOUNDED
        assert "Failed to trigger image pulling on agent agent-1: TimeoutError()" in caplog.text

    async def test_create_kernels_hang_is_bounded_and_recorded_as_failed_agent(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_valkey_schedule: AsyncMock,
        session_for_start_multi_node: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        caplog: pytest.LogCaptureFixture,
        drain: Callable[[SessionLauncher], Awaitable[None]],
    ) -> None:
        per_agent_client_pool.client(AgentId("agent-1")).create_kernels.side_effect = _hang
        per_agent_client_pool.client(AgentId("agent-2")).create_kernels.return_value = None

        started = time.monotonic()
        with RecorderContext.scope("test", entity_ids=[session_for_start_multi_node.session_id]):
            await launcher.start_sessions_for_handler(
                [session_for_start_multi_node],
                image_config_default,
            )
        # A background task awaits the replies under the (shortened) per-RPC bound.
        await drain(launcher)
        elapsed = time.monotonic() - started

        per_agent_client_pool.client(AgentId("agent-1")).create_kernels.assert_awaited_once()
        per_agent_client_pool.client(AgentId("agent-2")).create_kernels.assert_awaited_once()
        mock_valkey_schedule.record_session_failed_agents.assert_awaited_once_with(
            session_for_start_multi_node.session_id, [AgentId("agent-1")]
        )
        assert elapsed < BOUNDED
        assert "recording failed agents: {'agent-1': 'TimeoutError()'}" in caplog.text

    async def test_assign_port_hang_is_bounded_and_records_error(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        session_for_start_host_network: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
    ) -> None:
        agent_client = per_agent_client_pool.client(AgentId("agent-1"))
        agent_client.assign_port.side_effect = _hang

        started = time.monotonic()
        with RecorderContext.scope("test", entity_ids=[session_for_start_host_network.session_id]):
            await launcher.start_sessions_for_handler(
                [session_for_start_host_network],
                image_config_default,
            )
        elapsed = time.monotonic() - started

        agent_client.assign_port.assert_awaited_once()
        # The session failed before dispatch: no kernel creation was spawned.
        assert not launcher._kernel_creations
        agent_client.create_kernels.assert_not_awaited()
        mock_repository.update_session_error_info.assert_awaited_once()
        assert "TimeoutError" in str(mock_repository.update_session_error_info.await_args)
        assert elapsed < BOUNDED

    async def test_create_local_network_hang_is_bounded_and_records_error(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        session_for_start_multi_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
    ) -> None:
        agent_client = per_agent_client_pool.client(AgentId("agent-1"))
        agent_client.create_local_network.side_effect = _hang

        started = time.monotonic()
        with RecorderContext.scope("test", entity_ids=[session_for_start_multi_kernel.session_id]):
            await launcher.start_sessions_for_handler(
                [session_for_start_multi_kernel],
                image_config_default,
            )
        elapsed = time.monotonic() - started

        agent_client.create_local_network.assert_awaited_once()
        # The session failed before dispatch: no kernel creation was spawned.
        assert not launcher._kernel_creations
        agent_client.create_kernels.assert_not_awaited()
        mock_repository.update_session_error_info.assert_awaited_once()
        assert "TimeoutError" in str(mock_repository.update_session_error_info.await_args)
        assert elapsed < BOUNDED
