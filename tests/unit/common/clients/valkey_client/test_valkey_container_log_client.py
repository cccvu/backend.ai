from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai.backend.common.clients.valkey_client.valkey_container_log.client import (
    ValkeyContainerLogClient,
)
from ai.backend.common.log.types import ContainerLogData, ContainerLogError, ContainerLogType
from ai.backend.common.types import AgentId


@pytest.fixture
def conn() -> MagicMock:
    conn = MagicMock()
    conn.exec = AsyncMock()
    conn.llen = AsyncMock(return_value=0)
    conn.lpop_count = AsyncMock(return_value=None)
    conn.delete = AsyncMock()
    return conn


@pytest.fixture
def client(conn: MagicMock) -> ValkeyContainerLogClient:
    valkey_client = MagicMock()

    @asynccontextmanager
    async def _client() -> AsyncIterator[MagicMock]:
        yield conn

    valkey_client.client = _client
    return ValkeyContainerLogClient(valkey_client)


class TestContainerLogKeys:
    async def test_every_operation_uses_the_agent_scoped_key(
        self, client: ValkeyContainerLogClient, conn: MagicMock
    ) -> None:
        agent_id = AgentId("i-agent-1")
        logs = ContainerLogData.from_log(ContainerLogType.ZLIB, b"hello")

        await client.enqueue_container_logs(agent_id, "cid", logs)
        await client.container_log_len(agent_id, "cid")
        await client.pop_container_logs(agent_id, "cid")
        await client.clear_container_logs(agent_id, "cid")

        batch = conn.exec.await_args.args[0]
        assert [args[0] for _, args in batch.commands] == [
            "containerlog.i-agent-1.cid",  # RPUSH
            "containerlog.i-agent-1.cid",  # EXPIRE
        ]
        conn.llen.assert_awaited_once_with("containerlog.i-agent-1.cid")
        conn.lpop_count.assert_awaited_once_with("containerlog.i-agent-1.cid", 1)
        conn.delete.assert_awaited_once_with(["containerlog.i-agent-1.cid"])

    def test_key_contains_the_agent_id_as_a_whole_segment(
        self, client: ValkeyContainerLogClient
    ) -> None:
        assert client._container_log_key(AgentId("agent-1"), "c") == "containerlog.agent-1.c"
        assert not client._container_log_key(AgentId("agent-10"), "c").startswith("containerlog.agent-1.")


class TestPopContainerLogs:
    async def test_oversized_element_is_rejected_without_another_pop(
        self, client: ValkeyContainerLogClient, conn: MagicMock
    ) -> None:
        element = ContainerLogData.from_log(ContainerLogType.PLAINTEXT, b"x" * 100).serialize()
        conn.lpop_count.return_value = [element]
        with pytest.raises(ContainerLogError):
            await client.pop_container_logs(AgentId("a"), "c", max_element_size=len(element) - 1)
        # A rejected element is not retried, which would pop and lose the next one.
        conn.lpop_count.assert_awaited_once()

    async def test_malformed_element_is_rejected_without_another_pop(
        self, client: ValkeyContainerLogClient, conn: MagicMock
    ) -> None:
        conn.lpop_count.return_value = [b"not json"]
        with pytest.raises(ContainerLogError):
            await client.pop_container_logs(AgentId("a"), "c")
        conn.lpop_count.assert_awaited_once()
