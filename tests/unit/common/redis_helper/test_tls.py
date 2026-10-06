from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from glide import GlideClient, GlideClientConfiguration
from redis.asyncio.connection import SSLConnection

from ai.backend.common import redis_helper
from ai.backend.common.types import HostPortPair, RedisTarget, ValkeyTarget

PEM_DATA = b"-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n"


@pytest.fixture
def ca_file(tmp_path: Path) -> Path:
    path = tmp_path / "ca.pem"
    path.write_bytes(PEM_DATA)
    return path


@pytest.fixture
def standalone_redis_target_with_tls(ca_file: Path) -> RedisTarget:
    return RedisTarget(
        addr=HostPortPair("127.0.0.1", 6379),
        redis_helper_config={},
        use_tls=True,
        tls_ca_file=str(ca_file),
    )


@pytest.fixture
def sentinel_redis_target_with_tls(ca_file: Path) -> RedisTarget:
    return RedisTarget(
        sentinel=[HostPortPair("127.0.0.1", 26379)],
        service_name="mymaster",
        redis_helper_config={},
        use_tls=True,
        tls_ca_file=str(ca_file),
    )


class TestRedisObjectTls:
    def test_standalone_connection_verifies_with_ca_file(
        self, standalone_redis_target_with_tls: RedisTarget, ca_file: Path
    ) -> None:
        conn_info = redis_helper.get_redis_object_for_lock(
            standalone_redis_target_with_tls, name="test"
        )
        connection = conn_info.client.connection_pool.make_connection()
        assert isinstance(connection, SSLConnection)
        assert connection.ssl_context.ca_certs == str(ca_file)

    def test_sentinel_connection_verifies_with_ca_file(
        self, sentinel_redis_target_with_tls: RedisTarget, ca_file: Path
    ) -> None:
        conn_info = redis_helper.get_redis_object_for_lock(
            sentinel_redis_target_with_tls, name="test"
        )
        assert conn_info.sentinel is not None
        sentinel_conn = conn_info.sentinel.sentinels[0].connection_pool.make_connection()
        assert isinstance(sentinel_conn, SSLConnection)
        assert sentinel_conn.ssl_context.ca_certs == str(ca_file)


class TestCreateValkeyClientTls:
    async def test_passes_ca_file_contents_to_glide(self, ca_file: Path) -> None:
        target = ValkeyTarget(addr="127.0.0.1:6379", use_tls=True, tls_ca_file=str(ca_file))
        with patch.object(GlideClient, "create", new_callable=AsyncMock) as mock_create:
            await redis_helper.create_valkey_client(target, name="test")
        config: GlideClientConfiguration = mock_create.call_args.args[0]
        assert config.advanced_config is not None
        assert config.advanced_config.tls_config is not None
        assert config.advanced_config.tls_config.root_pem_cacerts == PEM_DATA
