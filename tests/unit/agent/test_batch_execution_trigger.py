"""
Tests for the idempotency of ``AbstractAgent.create_batch_execution_task()``.

The manager may send ``trigger_batch_execution`` more than once for the same kernel
(e.g., a retry after a timed-out or lost reply, or overlapping scheduler ticks).
The agent must start the batch job only once per kernel, while keeping the marker
in memory so that a restarted agent process never skips the batch job.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from ai.backend.agent.agent import AbstractAgent
from ai.backend.agent.types import ContainerLifecycleEvent, LifecycleEvent
from ai.backend.common.events.event_types.kernel.types import KernelLifecycleEventReason
from ai.backend.common.types import ContainerId, KernelId, SessionId


class _BatchRecorder:
    """Records ``execute_batch()`` calls and keeps the spawned tasks pending until released."""

    def __init__(self) -> None:
        self.calls: list[KernelId] = []
        self.release = asyncio.Event()

    async def __call__(
        self,
        session_id: SessionId,
        kernel_id: KernelId,
        startup_command: str,
        timeout_seconds: float | None = None,
    ) -> None:
        self.calls.append(kernel_id)
        await self.release.wait()


def _make_agent(recorder: _BatchRecorder) -> Any:
    """A stub carrying only what the batch trigger and the cleanup paths touch."""
    agent = MagicMock()
    agent._batch_started_kernels = set()
    agent._ongoing_exec_batch_tasks = weakref.WeakSet()
    agent.execute_batch = recorder
    agent.kernel_registry = {}
    agent.registry_lock = asyncio.Lock()
    agent.restarting_kernels = {}
    agent._ongoing_destruction_tasks = {}
    agent.stat_ctx.remove_kernel_metric = AsyncMock()
    agent.clean_kernel = AsyncMock()
    agent.reconstruct_resource_usage = AsyncMock()
    agent.anycast_and_broadcast_event = AsyncMock()
    agent.produce_error_event = AsyncMock()
    return agent


def _kernel_obj() -> MagicMock:
    kernel_obj = MagicMock()
    kernel_obj.runner.close = AsyncMock()
    kernel_obj.close = AsyncMock()
    kernel_obj.get.return_value = None
    kernel_obj.clean_event = None
    return kernel_obj


async def _trigger(agent: Any, session_id: SessionId, kernel_id: KernelId) -> None:
    await AbstractAgent.create_batch_execution_task(agent, session_id, kernel_id, "run.sh", None)


async def _settle(agent: Any, recorder: _BatchRecorder) -> None:
    """Let the spawned tasks record their calls, then release and finish them."""
    for _ in range(3):
        await asyncio.sleep(0)
    recorder.release.set()
    tasks = list(agent._ongoing_exec_batch_tasks)
    if tasks:
        await asyncio.gather(*tasks)


class TestBatchExecutionTrigger:
    async def test_duplicate_trigger_spawns_once(self) -> None:
        recorder = _BatchRecorder()
        agent = _make_agent(recorder)
        session_id, kernel_id = SessionId(uuid4()), KernelId(uuid4())
        agent.kernel_registry[kernel_id] = _kernel_obj()

        await _trigger(agent, session_id, kernel_id)
        await _trigger(agent, session_id, kernel_id)
        await _settle(agent, recorder)

        assert recorder.calls == [kernel_id]

    async def test_concurrent_triggers_spawn_once(self) -> None:
        recorder = _BatchRecorder()
        agent = _make_agent(recorder)
        session_id, kernel_id = SessionId(uuid4()), KernelId(uuid4())
        agent.kernel_registry[kernel_id] = _kernel_obj()

        await asyncio.gather(
            _trigger(agent, session_id, kernel_id),
            _trigger(agent, session_id, kernel_id),
        )
        await _settle(agent, recorder)

        assert recorder.calls == [kernel_id]

    async def test_different_kernels_each_spawn(self) -> None:
        recorder = _BatchRecorder()
        agent = _make_agent(recorder)
        session_id = SessionId(uuid4())
        kernel_a, kernel_b = KernelId(uuid4()), KernelId(uuid4())
        agent.kernel_registry[kernel_a] = _kernel_obj()
        agent.kernel_registry[kernel_b] = _kernel_obj()

        await _trigger(agent, session_id, kernel_a)
        await _trigger(agent, session_id, kernel_b)
        await _settle(agent, recorder)

        assert sorted(recorder.calls) == sorted([kernel_a, kernel_b])

    async def test_retry_after_agent_restart_spawns_again(self) -> None:
        """The marker is in memory only, so a new agent process never skips the batch job."""
        session_id, kernel_id = SessionId(uuid4()), KernelId(uuid4())
        first_recorder = _BatchRecorder()
        first_agent = _make_agent(first_recorder)
        first_agent.kernel_registry[kernel_id] = _kernel_obj()
        await _trigger(first_agent, session_id, kernel_id)
        await _settle(first_agent, first_recorder)

        # A fresh process starts with an empty marker set.
        second_recorder = _BatchRecorder()
        second_agent = _make_agent(second_recorder)
        second_agent.kernel_registry[kernel_id] = _kernel_obj()
        await _trigger(second_agent, session_id, kernel_id)
        await _settle(second_agent, second_recorder)

        assert first_recorder.calls == [kernel_id]
        assert second_recorder.calls == [kernel_id]

    async def test_marker_survives_execute_batch_completion(self) -> None:
        """A finished (or failed) batch task does not re-enable the trigger."""
        recorder = _BatchRecorder()
        agent = _make_agent(recorder)
        session_id, kernel_id = SessionId(uuid4()), KernelId(uuid4())
        agent.kernel_registry[kernel_id] = _kernel_obj()

        await _trigger(agent, session_id, kernel_id)
        await _settle(agent, recorder)
        await _trigger(agent, session_id, kernel_id)
        await _settle(agent, recorder)

        assert recorder.calls == [kernel_id]

    async def test_unknown_kernel_leaves_no_marker(self) -> None:
        """A trigger for a kernel the agent does not have neither runs nor blocks a later one."""
        recorder = _BatchRecorder()
        agent = _make_agent(recorder)
        session_id, kernel_id = SessionId(uuid4()), KernelId(uuid4())

        await _trigger(agent, session_id, kernel_id)
        await _settle(agent, recorder)
        assert recorder.calls == []
        assert kernel_id not in agent._batch_started_kernels

        agent.kernel_registry[kernel_id] = _kernel_obj()
        await _trigger(agent, session_id, kernel_id)
        await _settle(agent, recorder)
        assert recorder.calls == [kernel_id]


class TestBatchExecutionMarkerCleanup:
    async def test_clean_event_forgets_marker(self) -> None:
        recorder = _BatchRecorder()
        agent = _make_agent(recorder)
        session_id, kernel_id = SessionId(uuid4()), KernelId(uuid4())
        agent.kernel_registry[kernel_id] = _kernel_obj()
        await _trigger(agent, session_id, kernel_id)

        await AbstractAgent._handle_clean_event(
            agent,
            ContainerLifecycleEvent(
                kernel_id,
                session_id,
                ContainerId("container-id"),
                LifecycleEvent.CLEAN,
                KernelLifecycleEventReason.SELF_TERMINATED,
            ),
        )

        assert kernel_id not in agent.kernel_registry
        assert kernel_id not in agent._batch_started_kernels
        # A new container registered under the same kernel id may be triggered again.
        agent.kernel_registry[kernel_id] = _kernel_obj()
        await _trigger(agent, session_id, kernel_id)
        await _settle(agent, recorder)
        assert recorder.calls == [kernel_id, kernel_id]

    async def test_clean_event_for_restart_forgets_marker(self) -> None:
        """A restarted kernel keeps its id but runs in a new container, so it may be re-triggered."""
        recorder = _BatchRecorder()
        agent = _make_agent(recorder)
        session_id, kernel_id = SessionId(uuid4()), KernelId(uuid4())
        agent.kernel_registry[kernel_id] = _kernel_obj()
        restart_tracker = MagicMock()
        agent.restarting_kernels[kernel_id] = restart_tracker
        await _trigger(agent, session_id, kernel_id)

        await AbstractAgent._handle_clean_event(
            agent,
            ContainerLifecycleEvent(
                kernel_id,
                session_id,
                ContainerId("container-id"),
                LifecycleEvent.CLEAN,
                KernelLifecycleEventReason.RESTARTING,
            ),
        )

        assert kernel_id in agent.kernel_registry
        restart_tracker.destroy_event.set.assert_called_once()
        assert kernel_id not in agent._batch_started_kernels
        await _trigger(agent, session_id, kernel_id)
        await _settle(agent, recorder)
        assert recorder.calls == [kernel_id, kernel_id]

    async def test_dangling_kernel_cleanup_forgets_marker(self) -> None:
        recorder = _BatchRecorder()
        agent = _make_agent(recorder)
        session_id, kernel_id = SessionId(uuid4()), KernelId(uuid4())
        agent.kernel_registry[kernel_id] = _kernel_obj()
        await _trigger(agent, session_id, kernel_id)

        await AbstractAgent._clean_kernel_object(agent, kernel_id)

        assert kernel_id not in agent.kernel_registry
        assert kernel_id not in agent._batch_started_kernels
        # A new container registered under the same kernel id may be triggered again.
        agent.kernel_registry[kernel_id] = _kernel_obj()
        await _trigger(agent, session_id, kernel_id)
        await _settle(agent, recorder)
        assert recorder.calls == [kernel_id, kernel_id]
