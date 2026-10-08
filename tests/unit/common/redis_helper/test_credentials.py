from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, patch

import pytest
from glide import GlideClient, GlideClientConfiguration

from ai.backend.common import redis_helper
from ai.backend.common.exception import InvalidConfigError
from ai.backend.common.types import (
    HostPortPair,
    RedisConnectionInfo,
    RedisTarget,
    ValkeyTarget,
)

USERNAME = "backend-user"
PASSWORD = "secret"

type _RedisObjectFactory = Callable[..., RedisConnectionInfo]


@pytest.fixture
def standalone_redis_target_with_username() -> RedisTarget:
    return RedisTarget(
        addr=HostPortPair("127.0.0.1", 6379),
        redis_helper_config={},
        username=USERNAME,
        password=PASSWORD,
    )


@pytest.fixture
def sentinel_redis_target_with_username() -> RedisTarget:
    return RedisTarget(
        sentinel=[HostPortPair("127.0.0.1", 26379)],
        service_name="mymaster",
        redis_helper_config={},
        username=USERNAME,
        password=PASSWORD,
    )


class TestRedisObjectCredentials:
    def test_url_carries_username(self, standalone_redis_target_with_username: RedisTarget) -> None:
        url = redis_helper._parse_redis_url(standalone_redis_target_with_username, 0)
        assert url.user == USERNAME
        assert url.password == PASSWORD

    def test_url_without_username_is_unchanged(self) -> None:
        target = RedisTarget(addr=HostPortPair("127.0.0.1", 6379), password=PASSWORD)
        url = redis_helper._parse_redis_url(target, 0)
        assert str(url) == f"redis://:{PASSWORD}@127.0.0.1:6379/0"

    def test_standalone_connection_uses_username(
        self, standalone_redis_target_with_username: RedisTarget
    ) -> None:
        conn_info = redis_helper.get_redis_object_for_lock(
            standalone_redis_target_with_username, name="test"
        )
        connection = conn_info.client.connection_pool.make_connection()
        assert connection.username == USERNAME
        assert connection.password == PASSWORD

    @pytest.mark.parametrize(
        "factory", [redis_helper.get_redis_object, redis_helper.get_redis_object_for_lock]
    )
    def test_sentinel_master_uses_username(
        self, sentinel_redis_target_with_username: RedisTarget, factory: _RedisObjectFactory
    ) -> None:
        conn_info = factory(sentinel_redis_target_with_username, name="test")
        assert conn_info.sentinel is not None
        assert conn_info.client.connection_pool.connection_kwargs["username"] == USERNAME
        assert conn_info.client.connection_pool.connection_kwargs["password"] == PASSWORD
        # Sentinel nodes are authenticated with a password only.
        sentinel_conn = conn_info.sentinel.sentinels[0].connection_pool.make_connection()
        assert sentinel_conn.username is None

    @pytest.mark.parametrize(
        "factory", [redis_helper.get_redis_object, redis_helper.get_redis_object_for_lock]
    )
    def test_username_without_password_is_refused(self, factory: _RedisObjectFactory) -> None:
        target = RedisTarget(
            addr=HostPortPair("127.0.0.1", 6379), redis_helper_config={}, username=USERNAME
        )
        with pytest.raises(InvalidConfigError) as exc_info:
            factory(target, name="test")
        assert USERNAME not in str(exc_info.value)


class TestCreateValkeyClientCredentials:
    async def test_passes_username_to_glide(self) -> None:
        target = ValkeyTarget(addr="127.0.0.1:6379", username=USERNAME, password=PASSWORD)
        with patch.object(GlideClient, "create", new_callable=AsyncMock) as mock_create:
            await redis_helper.create_valkey_client(target, name="test")
        config: GlideClientConfiguration = mock_create.call_args.args[0]
        assert config.credentials is not None
        assert config.credentials.username == USERNAME
        assert config.credentials.password == PASSWORD

    async def test_refuses_username_without_password(self) -> None:
        target = ValkeyTarget(addr="127.0.0.1:6379", username=USERNAME)
        with patch.object(GlideClient, "create", new_callable=AsyncMock) as mock_create:
            with pytest.raises(InvalidConfigError):
                await redis_helper.create_valkey_client(target, name="test")
        mock_create.assert_not_awaited()
