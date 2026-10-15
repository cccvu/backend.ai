from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, cast, override
from unittest.mock import AsyncMock, MagicMock

import pytest
from glide import Batch, GlideClient

from ai.backend.common.bgtask.types import (
    BgTaskKey,
    TaskID,
    TaskInfo,
    TaskStatus,
    TaskSubKeyInfo,
    TaskTotalInfo,
    TaskType,
    validate_bgtask_key_scope,
)
from ai.backend.common.clients.valkey_client.client import AbstractValkeyClient
from ai.backend.common.clients.valkey_client.valkey_bgtask.client import (
    TaskSetKey,
    ValkeyBgtaskClient,
)
from ai.backend.common.exception import InvalidConfigError

TASK_KEY_KINDS = ("meta", "subkey", "subtask")


class _FakeValkeyClient(AbstractValkeyClient):
    """Records every key the background task client touches."""

    conn: MagicMock
    keys: list[str]

    def __init__(self) -> None:
        self.conn = MagicMock()
        self.conn.exec = AsyncMock(side_effect=self._exec)
        self.conn.smembers = AsyncMock(return_value={b"sub-1"})
        self.conn.hgetall = AsyncMock(return_value={})
        self.conn.invoke_script = AsyncMock(return_value=b"ttl_insufficient")
        self.keys = []

    async def _exec(self, batch: Batch, raise_on_error: bool) -> list[Any]:
        for _, args in batch.commands:
            self.keys.append(args[0])
        return [0 for _ in batch.commands]

    @override
    async def connect(self) -> None:
        pass

    @override
    async def disconnect(self) -> None:
        pass

    @override
    async def ping(self) -> None:
        pass

    @override
    async def need_reconnect(self) -> bool:
        return False

    @override
    async def check_health(self) -> None:
        pass

    @override
    def client(self) -> AbstractAsyncContextManager[GlideClient]:
        return self._client()

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[GlideClient]:
        yield cast(GlideClient, self.conn)

    def touched(self) -> list[str]:
        keys = list(self.keys)
        for method in (self.conn.smembers, self.conn.hgetall):
            keys.extend(call.args[0] for call in method.await_args_list)
        for call in self.conn.invoke_script.await_args_list:
            keys.extend(call.kwargs["keys"])
        return keys


def _total_info(task_id: TaskID) -> TaskTotalInfo:
    return TaskTotalInfo(
        task_info=TaskInfo(
            task_id=task_id,
            task_name="test-task",
            task_type=TaskType.BATCH,
            body={},
            ongoing_count=1,
        ),
        task_key_list=[
            TaskSubKeyInfo(
                task_id=task_id,
                key=BgTaskKey("sub-1"),
                status=TaskStatus.ONGOING,
                last_message="",
            )
        ],
    )


async def _exercise(client: ValkeyBgtaskClient, task_id: TaskID, task_set_key: TaskSetKey) -> None:
    await client.register_task(_total_info(task_id), task_set_key)
    await client.claim_task(task_id, task_set_key)
    await client.heartbeat([_total_info(task_id)], task_set_key)
    await client.finish_subtask(task_id, "sub-1", TaskStatus.SUCCESS, "done")
    await client.unregister_task(task_id, task_set_key)
    await client.fetch_unmanaged_tasks(task_set_key)


class TestKeyScope:
    @pytest.mark.parametrize(
        ("key_scope", "infix"),
        [
            pytest.param(None, "", id="default"),
            pytest.param("storage-proxy.node-1", "storage-proxy.node-1:", id="scoped"),
        ],
    )
    async def test_every_task_key_carries_the_scope(
        self, key_scope: str | None, infix: str
    ) -> None:
        fake = _FakeValkeyClient()
        client = ValkeyBgtaskClient(fake, key_scope=key_scope)
        task_id = TaskID(uuid.uuid4())
        fake.conn.smembers.side_effect = lambda key: (
            {task_id.hex.encode()}
            if key.startswith(("bgtask:server:", "bgtask:tag:"))
            else {b"sub-1"}
        )
        task_set_key = TaskSetKey(server_id="node-1", tags={"tag-1"})

        await _exercise(client, task_id, task_set_key)

        keys = fake.touched()
        task_keys = [k for k in keys if k.split(":")[1] in TASK_KEY_KINDS]
        assert set(task_keys) == {
            f"bgtask:meta:{infix}{task_id}",
            f"bgtask:subkey:{infix}{task_id}",
            f"bgtask:subtask:{infix}{task_id}:sub-1",
        }
        # The server and tag sets are never scoped.
        assert {k for k in keys if k not in task_keys} == {
            "bgtask:server:node-1",
            "bgtask:tag:tag-1",
        }

    def test_the_default_keys_are_unchanged(self) -> None:
        client = ValkeyBgtaskClient(_FakeValkeyClient())
        task_id = TaskID(uuid.uuid4())
        assert client._get_task_key(task_id) == f"bgtask:meta:{task_id}"
        assert client._get_task_subkey_set_key(task_id) == f"bgtask:subkey:{task_id}"
        assert client._get_subtask_key(task_id, "k") == f"bgtask:subtask:{task_id}:k"

    @pytest.mark.parametrize("key_scope", ["", "a:b", "a*", "a?", "a[b]", "a b", "a\\b", "é"])
    def test_a_scope_that_is_not_one_key_segment_is_refused(self, key_scope: str) -> None:
        with pytest.raises(InvalidConfigError):
            validate_bgtask_key_scope(key_scope)
        with pytest.raises(InvalidConfigError):
            ValkeyBgtaskClient(_FakeValkeyClient(), key_scope=key_scope)

    @pytest.mark.parametrize("key_scope", ["storage-proxy.node-1", "A_b-9"])
    def test_a_scope_of_one_key_segment_is_accepted(self, key_scope: str) -> None:
        assert validate_bgtask_key_scope(key_scope) == key_scope
