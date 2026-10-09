"""Starting a session hands create_kernels off and does not await the agents' replies.

The scheduler holds its START lock while it starts sessions, so a hung agent must not hold
the pass: the pass waits only until each request is handed off, and a tracked background
task awaits the replies. Timing is enforced with ``asyncio.timeout`` and synchronised with
events, under the real per-RPC bound.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, override
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from ai.backend.common.types import AgentId, SessionId
from ai.backend.manager.sokovan.recorder import RecorderContext
from ai.backend.manager.sokovan.recorder.pool import RecordPool
from ai.backend.manager.sokovan.scheduler.launcher import launcher as launcher_module
from ai.backend.manager.sokovan.scheduler.launcher.launcher import SessionLauncher
from ai.backend.manager.views.sokovan.image import ImageConfigData
from ai.backend.manager.views.sokovan.lifecycle import SessionDataForStart

# Far below AGENT_CREATE_KERNELS_TIMEOUT_SEC, which these tests leave at its real value.
BOUND = 2.0

AGENT_1 = AgentId("agent-1")
AGENT_2 = AgentId("agent-2")


class _UnprintableError(Exception):
    @override
    def __repr__(self) -> str:
        raise RuntimeError("repr failed")


@pytest.fixture
def mock_agent_client_pool(per_agent_client_pool: MagicMock) -> MagicMock:
    """Overrides the shared single-client pool with one stub client per agent."""
    return per_agent_client_pool


async def _start(
    launcher: SessionLauncher,
    session: SessionDataForStart,
    image_configs: dict[UUID, ImageConfigData],
) -> RecordPool[SessionId]:
    with RecorderContext[SessionId].scope("test", entity_ids=[session.session_id]) as pool:
        await launcher.start_sessions_for_handler([session], image_configs)
    return pool


class TestKernelCreationDispatch:
    async def test_hung_agent_does_not_hold_start(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_valkey_schedule: AsyncMock,
        session_for_start_multi_node: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
    ) -> None:
        hung_entered = asyncio.Event()
        healthy_entered = asyncio.Event()

        async def hang(*args: Any, **kwargs: Any) -> None:
            hung_entered.set()
            await asyncio.Event().wait()

        async def reply(*args: Any, **kwargs: Any) -> None:
            healthy_entered.set()

        per_agent_client_pool.client(AGENT_1).create_kernels.side_effect = hang
        per_agent_client_pool.client(AGENT_2).create_kernels.side_effect = reply

        async with asyncio.timeout(BOUND):
            await _start(launcher, session_for_start_multi_node, image_config_default)

        # Both requests were handed off before start returned.
        assert hung_entered.is_set()
        assert healthy_entered.is_set()
        # The background task is still awaiting the hung agent; nothing is recorded yet.
        assert len(launcher._kernel_creations) == 1
        mock_valkey_schedule.record_session_failed_agents.assert_not_awaited()

    async def test_start_waits_for_handoff(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
    ) -> None:
        gate = asyncio.Event()
        acquiring = asyncio.Event()
        client = per_agent_client_pool.client(AGENT_1)

        @asynccontextmanager
        async def gated_acquire(agent_id: AgentId) -> AsyncIterator[AsyncMock]:
            # First contact: the pool looks the agent up before it can hand out a client.
            acquiring.set()
            await gate.wait()
            yield per_agent_client_pool.client(agent_id)

        per_agent_client_pool.acquire.side_effect = gated_acquire

        start_tasks: list[asyncio.Task[RecordPool[SessionId]]] = []
        start_done_on_entry: list[bool] = []

        async def record_entry(*args: Any, **kwargs: Any) -> None:
            start_done_on_entry.append(start_tasks[0].done())

        client.create_kernels.side_effect = record_entry

        async with asyncio.timeout(BOUND):
            start = asyncio.create_task(
                _start(launcher, session_for_start_single_kernel, image_config_default)
            )
            start_tasks.append(start)
            await acquiring.wait()
            for _ in range(10):
                await asyncio.sleep(0)
            assert not start.done()
            client.create_kernels.assert_not_awaited()

            gate.set()
            await start

        assert start_done_on_entry == [False]

    async def test_handoff_wait_is_bounded(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(launcher_module, "KERNEL_CREATION_HANDOFF_TIMEOUT_SEC", 0.05)

        @asynccontextmanager
        async def stuck_acquire(agent_id: AgentId) -> AsyncIterator[AsyncMock]:
            await asyncio.Event().wait()
            yield per_agent_client_pool.client(agent_id)

        per_agent_client_pool.acquire.side_effect = stuck_acquire
        tasks_before = asyncio.all_tasks()

        async with asyncio.timeout(BOUND):
            await _start(launcher, session_for_start_single_kernel, image_config_default)

        assert "kernel creation not handed off to every agent within 0.05s" in caplog.text
        per_agent_client_pool.client(AGENT_1).create_kernels.assert_not_awaited()

        await launcher.close()
        assert not launcher._kernel_creations
        assert asyncio.all_tasks() - tasks_before == set()

    async def test_task_cancelled_before_first_step_releases_handoff(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        session_for_start_multi_node: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A bound far above the enforced limit: only the release can end the pass's wait.
        monkeypatch.setattr(launcher_module, "KERNEL_CREATION_HANDOFF_TIMEOUT_SEC", 60)
        spawn = launcher._spawn_kernel_creation
        spawned: list[asyncio.Task[None]] = []

        def spawn_then_cancel(
            session_id: SessionId, coro: Coroutine[Any, Any, None]
        ) -> asyncio.Task[None]:
            # As close() would, before the loop runs the task's first step.
            task = spawn(session_id, coro)
            task.cancel()
            spawned.append(task)
            return task

        monkeypatch.setattr(launcher, "_spawn_kernel_creation", spawn_then_cancel)

        async with asyncio.timeout(1):
            await _start(launcher, session_for_start_multi_node, image_config_default)

        (task,) = spawned
        assert task.cancelled()
        assert not launcher._kernel_creations
        per_agent_client_pool.client(AGENT_1).create_kernels.assert_not_awaited()
        per_agent_client_pool.client(AGENT_2).create_kernels.assert_not_awaited()

    async def test_close_cancels_in_flight_creation(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
    ) -> None:
        cancelled = asyncio.Event()

        async def hang(*args: Any, **kwargs: Any) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        per_agent_client_pool.client(AGENT_1).create_kernels.side_effect = hang

        async with asyncio.timeout(BOUND):
            await _start(launcher, session_for_start_single_kernel, image_config_default)
            assert len(launcher._kernel_creations) == 1
            await launcher.close()

        assert cancelled.is_set()
        assert not launcher._kernel_creations
        mock_repository.update_session_error_info.assert_not_awaited()
        mock_valkey_schedule.record_session_failed_agents.assert_not_awaited()

    async def test_background_task_cannot_reach_recorder(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        drain: Callable[[SessionLauncher], Awaitable[None]],
    ) -> None:
        lookups: list[type[BaseException] | None] = []

        async def inspect_context(*args: Any, **kwargs: Any) -> None:
            try:
                RecorderContext[SessionId].current_pool()
            except LookupError as e:
                lookups.append(type(e))
            else:
                lookups.append(None)

        per_agent_client_pool.client(AGENT_1).create_kernels.side_effect = inspect_context

        async with asyncio.timeout(BOUND):
            pool = await _start(launcher, session_for_start_single_kernel, image_config_default)
            await drain(launcher)

        assert lookups == [LookupError]
        record = pool.build_all_records()[session_for_start_single_kernel.session_id]
        steps = {(phase.name, step.name) for phase in record.phases for step in phase.steps}
        assert ("trigger_kernel_creation", "create_kernels") in steps

    async def test_task_reference_held_until_done(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
    ) -> None:
        proceed = asyncio.Event()

        async def reply_when_told(*args: Any, **kwargs: Any) -> None:
            await proceed.wait()

        per_agent_client_pool.client(AGENT_1).create_kernels.side_effect = reply_when_told

        async with asyncio.timeout(BOUND):
            await _start(launcher, session_for_start_single_kernel, image_config_default)
            (task,) = launcher._kernel_creations
            assert task.get_name() == (
                f"sokovan.create_kernels:{session_for_start_single_kernel.session_id}"
            )
            assert not task.done()

            proceed.set()
            await task
            await asyncio.sleep(0)

        assert not launcher._kernel_creations

    async def test_unexpected_error_is_logged_not_left_on_task(
        self,
        launcher: SessionLauncher,
        per_agent_client_pool: MagicMock,
        session_for_start_single_kernel: SessionDataForStart,
        image_config_default: dict[UUID, ImageConfigData],
        drain: Callable[[SessionLauncher], Awaitable[None]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Formatting the failed agents raises inside the background body.
        per_agent_client_pool.client(AGENT_1).create_kernels.side_effect = _UnprintableError()

        async with asyncio.timeout(BOUND):
            await _start(launcher, session_for_start_single_kernel, image_config_default)
            (task,) = launcher._kernel_creations
            await drain(launcher)

        assert not task.cancelled()
        assert task.exception() is None
        assert "kernel creation failed unexpectedly" in caplog.text
        assert "repr failed" in caplog.text
