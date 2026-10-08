from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai.backend.common.bgtask.bgtask import BackgroundTaskManager, BackgroundTaskManagerArgs
from ai.backend.common.bgtask.hooks.base import TaskContext
from ai.backend.common.bgtask.hooks.event_hook import EventProducerHook
from ai.backend.common.bgtask.reporter import ProgressReporter
from ai.backend.common.bgtask.types import TaskID, agent_bgtask_cache_scope, bgtask_cache_id
from ai.backend.common.events.types import EventCacheDomain


class TestCacheId:
    def test_default_scope_keeps_the_unscoped_key(self) -> None:
        task_id = uuid.uuid4()
        assert bgtask_cache_id(task_id) == f"bgtask.{task_id}"
        assert bgtask_cache_id(task_id) == EventCacheDomain.BGTASK.cache_id(str(task_id))

    def test_agent_scope(self) -> None:
        task_id = uuid.uuid4()
        scope = agent_bgtask_cache_scope("agent-1")
        assert scope == "agent.agent-1"
        assert bgtask_cache_id(task_id, scope) == f"bgtask.agent.agent-1.{task_id}"
        # The agent ID is a whole segment, so one ID never prefixes another's keys.
        other = bgtask_cache_id(task_id, agent_bgtask_cache_scope("agent-10"))
        assert not other.startswith("bgtask.agent.agent-1.")


def _make_manager(cache_scope: str | None) -> tuple[BackgroundTaskManager, AsyncMock]:
    event_producer = MagicMock()
    event_producer.broadcast_event_with_cache = AsyncMock()
    manager = BackgroundTaskManager(
        BackgroundTaskManagerArgs(
            event_producer=event_producer,
            valkey_client=MagicMock(),
            server_id="test-server",
            cache_scope=cache_scope,
        )
    )
    return manager, event_producer.broadcast_event_with_cache


async def _run_task(manager: BackgroundTaskManager, broadcast: AsyncMock) -> uuid.UUID:
    async def _task(reporter: ProgressReporter) -> str:
        await reporter.update(1, message="halfway")
        return "done"

    task_id = await manager.start(_task)
    # Started, the progress update and the result.
    for _ in range(500):
        if broadcast.await_count >= 3:
            break
        await asyncio.sleep(0.01)
    return task_id


class TestBackgroundTaskManagerCacheScope:
    @pytest.mark.parametrize(
        ("cache_scope", "prefix"),
        [
            pytest.param(None, "bgtask.", id="default"),
            pytest.param("agent.agent-1", "bgtask.agent.agent-1.", id="agent"),
        ],
    )
    async def test_every_cached_event_uses_the_scope(
        self, cache_scope: str | None, prefix: str
    ) -> None:
        manager, broadcast = _make_manager(cache_scope)
        task_id = await _run_task(manager, broadcast)

        cache_ids = [call.args[0] for call in broadcast.await_args_list]
        assert len(cache_ids) == 3
        assert set(cache_ids) == {f"{prefix}{task_id}"}


class TestEventProducerHookCacheScope:
    @pytest.mark.parametrize(
        ("cache_scope", "prefix"),
        [
            pytest.param(None, "bgtask.", id="default"),
            pytest.param("agent.agent-1", "bgtask.agent.agent-1.", id="agent"),
        ],
    )
    async def test_hook_uses_the_scope(self, cache_scope: str | None, prefix: str) -> None:
        event_producer = MagicMock()
        event_producer.broadcast_event_with_cache = AsyncMock()
        hook = EventProducerHook(event_producer, cache_scope)
        context = TaskContext(task_name=MagicMock(), task_id=TaskID(uuid.uuid4()))

        async with hook.apply(context):
            pass

        (call,) = event_producer.broadcast_event_with_cache.await_args_list
        assert call.args[0] == f"{prefix}{context.task_id}"
