"""Unit tests for KernelEventHandler.handle_kernel_log."""

from __future__ import annotations

import tracemalloc
import uuid
from base64 import b64encode
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai.backend.common.clients.valkey_client.valkey_container_log.client import (
    ValkeyContainerLogClient,
)
from ai.backend.common.events.event_types.kernel.anycast import DoSyncKernelLogsEvent
from ai.backend.common.log.types import ContainerLogData, ContainerLogType
from ai.backend.common.types import AgentId, KernelId
from ai.backend.manager.event_dispatcher.handlers import kernel as kernel_handlers
from ai.backend.manager.event_dispatcher.handlers.kernel import KernelEventHandler

AGENT = AgentId("i-agent-1")
OTHER_AGENT = AgentId("i-agent-2")
CONTAINER_ID = "c0ffee"
KEY = f"containerlog.{AGENT}.{CONTAINER_ID}"


class _FakeLists:
    """An in-memory stand-in for the Valkey list commands the client uses."""

    def __init__(self) -> None:
        self.lists: dict[str, list[bytes]] = {}
        self.deleted: list[str] = []
        self.popped = 0

    async def llen(self, key: str) -> int:
        return len(self.lists.get(key, []))

    async def lpop_count(self, key: str, count: int) -> list[bytes] | None:
        items = self.lists.get(key)
        if not items:
            return None
        self.popped += 1
        popped, self.lists[key] = items[:count], items[count:]
        return popped

    async def delete(self, keys: list[str]) -> int:
        for key in keys:
            self.deleted.append(key)
            self.lists.pop(key, None)
        return len(keys)


def _make_log_client(fake: _FakeLists) -> ValkeyContainerLogClient:
    valkey_client = MagicMock()

    @asynccontextmanager
    async def _client() -> AsyncIterator[_FakeLists]:
        yield fake

    valkey_client.client = _client
    return ValkeyContainerLogClient(valkey_client)


class _FakeDB:
    def __init__(self, row: Any, *, fail_update: bool = False) -> None:
        self.row = row
        self.fail_update = fail_update
        self.stored_logs: list[bytes] = []

    @asynccontextmanager
    async def begin_readonly(self) -> AsyncIterator[MagicMock]:
        conn = MagicMock()
        result = MagicMock()
        result.first.return_value = self.row
        conn.execute = AsyncMock(return_value=result)
        yield conn

    @asynccontextmanager
    async def begin(self) -> AsyncIterator[MagicMock]:
        conn = MagicMock()

        async def _execute(query: Any) -> None:
            if self.fail_update:
                raise RuntimeError("update failed")
            self.stored_logs.append(query.compile().params["container_log"])

        conn.execute = _execute
        yield conn


def _row(agent: str | None = AGENT, container_id: str | None = CONTAINER_ID) -> MagicMock:
    row = MagicMock()
    row.agent = agent
    row.container_id = container_id
    return row


def _make_handler(fake: _FakeLists, db: _FakeDB) -> KernelEventHandler:
    return KernelEventHandler(
        valkey_container_log=_make_log_client(fake),
        valkey_stat=MagicMock(),
        valkey_stream=MagicMock(),
        registry=MagicMock(),
        db=db,  # type: ignore[arg-type]
        schedule_coordinator=MagicMock(),
    )


def _event(container_id: str = CONTAINER_ID) -> DoSyncKernelLogsEvent:
    return DoSyncKernelLogsEvent(KernelId(uuid.uuid4()), container_id)


def _chunk(data: bytes) -> bytes:
    return ContainerLogData.from_log(ContainerLogType.ZLIB, data).serialize()


class TestHandleKernelLog:
    async def test_reads_the_kernel_agents_key(self) -> None:
        fake = _FakeLists()
        fake.lists[KEY] = [_chunk(b"hello "), _chunk(b"world\n")]
        fake.lists[f"containerlog.{OTHER_AGENT}.{CONTAINER_ID}"] = [_chunk(b"other\n")]
        db = _FakeDB(_row())

        await _make_handler(fake, db).handle_kernel_log(None, AGENT, _event())

        assert db.stored_logs == [b"hello world\n"]
        assert fake.deleted == [KEY]
        assert f"containerlog.{OTHER_AGENT}.{CONTAINER_ID}" in fake.lists

    @pytest.mark.parametrize(
        ("source", "row", "event_container_id"),
        [
            pytest.param(OTHER_AGENT, _row(), CONTAINER_ID, id="other-agent"),
            pytest.param(AGENT, _row(), "other-container", id="other-container"),
            pytest.param(AGENT, _row(agent=None), CONTAINER_ID, id="no-agent"),
            pytest.param(AGENT, _row(container_id=None), CONTAINER_ID, id="no-container"),
            pytest.param(AGENT, None, CONTAINER_ID, id="no-kernel"),
        ],
    )
    async def test_unbound_sync_is_dropped(
        self, source: AgentId, row: Any, event_container_id: str
    ) -> None:
        fake = _FakeLists()
        fake.lists[KEY] = [_chunk(b"log\n")]
        db = _FakeDB(row)

        await _make_handler(fake, db).handle_kernel_log(None, source, _event(event_container_id))

        # Neither read nor cleared: the logs stay for the agent that hosts the kernel.
        assert db.stored_logs == []
        assert fake.deleted == []
        assert fake.lists[KEY] == [_chunk(b"log\n")]

    async def test_decompression_bomb_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kernel_handlers, "MAX_CONTAINER_LOG_SIZE", 1024)
        fake = _FakeLists()
        # Each chunk expands to 16 MiB but only the budget is ever decompressed.
        bomb = _chunk(b"\0" * (16 * 1024 * 1024))
        fake.lists[KEY] = [_chunk(b"head\n"), bomb, bomb, bomb]
        db = _FakeDB(_row())

        await _make_handler(fake, db).handle_kernel_log(None, AGENT, _event())

        (stored,) = db.stored_logs
        assert stored == b"head\n" + b"\0" * (1024 - 5) + b"(container log truncated)\n"
        # Reading stops once the budget is spent; the rest goes with the key.
        assert fake.popped == 2
        assert fake.deleted == [KEY]

    async def test_bomb_memory_stays_within_the_cap(self) -> None:
        fake = _FakeLists()
        # About 128 KiB in Valkey that expands to 128 MiB.
        fake.lists[KEY] = [_chunk(b"\0" * (128 * 1024 * 1024))]
        db = _FakeDB(_row())
        handler = _make_handler(fake, db)

        tracemalloc.start()
        try:
            await handler.handle_kernel_log(None, AGENT, _event())
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        (stored,) = db.stored_logs
        assert len(stored) == kernel_handlers.MAX_CONTAINER_LOG_SIZE + len(
            b"(container log truncated)\n"
        )
        assert peak < 4 * kernel_handlers.MAX_CONTAINER_LOG_SIZE

    async def test_chunk_count_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(kernel_handlers, "MAX_CONTAINER_LOG_CHUNKS", 3)
        fake = _FakeLists()
        fake.lists[KEY] = [_chunk(b"")] * 10
        db = _FakeDB(_row())

        await _make_handler(fake, db).handle_kernel_log(None, AGENT, _event())

        assert db.stored_logs == [b"(container log truncated)\n"]
        assert fake.popped == 3
        assert fake.deleted == [KEY]

    async def test_oversized_and_malformed_chunks_are_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(kernel_handlers, "MAX_CONTAINER_LOG_ELEMENT_SIZE", 256)
        fake = _FakeLists()
        poison = b'{"compress_type": "zlib", "content": "%s"}' % b64encode(b"\0\0\0")
        fake.lists[KEY] = [
            _chunk(b"a\n"),
            b"x" * 257,  # rejected by size before parsing
            b"not json",
            poison,
            _chunk(b"b\n"),
        ]
        db = _FakeDB(_row())

        await _make_handler(fake, db).handle_kernel_log(None, AGENT, _event())

        assert db.stored_logs == [b"a\nb\n"]
        assert fake.deleted == [KEY]

    async def test_key_is_deleted_when_the_update_fails(self) -> None:
        fake = _FakeLists()
        fake.lists[KEY] = [_chunk(b"log\n")]
        db = _FakeDB(_row(), fail_update=True)

        await _make_handler(fake, db).handle_kernel_log(None, AGENT, _event())

        assert fake.deleted == [KEY]

    async def test_key_is_deleted_when_reading_fails(self) -> None:
        fake = _FakeLists()
        fake.lists[KEY] = [_chunk(b"log\n")]
        fake.llen = AsyncMock(side_effect=RuntimeError("connection lost"))  # type: ignore[method-assign]
        db = _FakeDB(_row())

        await _make_handler(fake, db).handle_kernel_log(None, AGENT, _event())

        assert db.stored_logs == []
        assert fake.deleted == [KEY]
