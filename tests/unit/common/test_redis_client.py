from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.backend.common.exception import InvalidConfigError
from ai.backend.common.redis_client import RedisClient, RedisConnection
from ai.backend.common.types import HostPortPair, RedisTarget

USERNAME = "backend-user"
PASSWORD = "secret"


@pytest.fixture
def mock_execute() -> Iterator[AsyncMock]:
    writer = MagicMock()
    with (
        patch(
            "ai.backend.common.redis_client.asyncio.open_connection",
            new_callable=AsyncMock,
            return_value=(MagicMock(), writer),
        ),
        patch.object(RedisClient, "execute", new_callable=AsyncMock) as execute,
    ):
        yield execute


def _sent_commands(execute: AsyncMock) -> list[list[object]]:
    return [call.args[0] for call in execute.call_args_list]


class TestRedisConnectionAuth:
    async def test_password_only(self, mock_execute: AsyncMock) -> None:
        target = RedisTarget(addr=HostPortPair("127.0.0.1", 6379), password=PASSWORD)
        await RedisConnection(target).connect()
        assert _sent_commands(mock_execute)[0] == ["AUTH", PASSWORD]

    async def test_username_and_password(self, mock_execute: AsyncMock) -> None:
        target = RedisTarget(
            addr=HostPortPair("127.0.0.1", 6379), username=USERNAME, password=PASSWORD
        )
        await RedisConnection(target).connect()
        assert _sent_commands(mock_execute)[0] == ["AUTH", USERNAME, PASSWORD]

    async def test_username_without_password_is_refused(self, mock_execute: AsyncMock) -> None:
        target = RedisTarget(addr=HostPortPair("127.0.0.1", 6379), username=USERNAME)
        with pytest.raises(InvalidConfigError) as exc_info:
            await RedisConnection(target).connect()
        assert USERNAME not in str(exc_info.value)
        mock_execute.assert_not_awaited()
