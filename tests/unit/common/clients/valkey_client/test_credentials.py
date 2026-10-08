from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from glide import GlideClientConfiguration, ServerCredentials

from ai.backend.common.clients.valkey_client.client import (
    MonitoringValkeyClient,
    ValkeySentinelClient,
    ValkeySentinelTarget,
    ValkeyStandaloneClient,
    ValkeyStandaloneTarget,
    build_server_credentials,
    create_valkey_client,
)
from ai.backend.common.config import redis_config_iv
from ai.backend.common.configs.redis import RedisConfig, SingleRedisConfig
from ai.backend.common.defs import RedisRole
from ai.backend.common.exception import InvalidConfigError
from ai.backend.common.types import RedisProfileTarget, RedisTarget, ValkeyTarget

BASE_USERNAME = "base-user"
BASE_PASSWORD = "base-secret"
OVERRIDE_USERNAME = "stream-user"
OVERRIDE_PASSWORD = "stream-secret"

type _Target = ValkeyTarget | RedisTarget | ValkeyStandaloneTarget | ValkeySentinelTarget


def _assert_credentials(target: _Target, *, username: str | None, password: str | None) -> None:
    assert target.username == username
    assert target.password == password


def _redis_config_data() -> dict[str, object]:
    return {
        "addr": "127.0.0.1:6379",
        "username": BASE_USERNAME,
        "password": BASE_PASSWORD,
        "override_configs": {
            RedisRole.STREAM.value: {
                "addr": "127.0.0.1:6380",
                "username": OVERRIDE_USERNAME,
                "password": OVERRIDE_PASSWORD,
            },
            # An override without credentials of its own.
            RedisRole.LIVE.value: {"addr": "127.0.0.1:6381"},
        },
    }


@pytest.fixture
def redis_config_with_username() -> RedisConfig:
    return RedisConfig.model_validate(_redis_config_data())


@pytest.fixture
def single_redis_config_with_username() -> SingleRedisConfig:
    return SingleRedisConfig.model_validate({
        "addr": "127.0.0.1:6379",
        "username": BASE_USERNAME,
        "password": BASE_PASSWORD,
    })


@pytest.fixture
def redis_target_with_username() -> RedisTarget:
    return RedisTarget(
        sentinel="127.0.0.1:26379",
        service_name="mymaster",
        username=BASE_USERNAME,
        password=BASE_PASSWORD,
    )


@pytest.fixture
def valkey_sentinel_target_with_username() -> ValkeyTarget:
    return ValkeyTarget(
        sentinel=["127.0.0.1:26379"],
        service_name="mymaster",
        username=BASE_USERNAME,
        password=BASE_PASSWORD,
    )


@pytest.fixture
def valkey_standalone_target_with_username() -> ValkeyTarget:
    return ValkeyTarget(
        addr="127.0.0.1:6379",
        username=BASE_USERNAME,
        password=BASE_PASSWORD,
    )


@pytest.fixture
def valkey_standalone_target_without_password() -> ValkeyTarget:
    return ValkeyTarget(addr="127.0.0.1:6379", username=BASE_USERNAME)


class TestUsernamePropagation:
    """Verify the username survives every conversion between config and targets."""

    def test_single_redis_config_to_valkey_target(
        self, single_redis_config_with_username: SingleRedisConfig
    ) -> None:
        target = single_redis_config_with_username.to_valkey_target()
        _assert_credentials(target, username=BASE_USERNAME, password=BASE_PASSWORD)

    def test_single_redis_config_to_redis_target(
        self, single_redis_config_with_username: SingleRedisConfig
    ) -> None:
        target = single_redis_config_with_username.to_redis_target()
        _assert_credentials(target, username=BASE_USERNAME, password=BASE_PASSWORD)

    def test_valkey_profile_target(self, redis_config_with_username: RedisConfig) -> None:
        profile = redis_config_with_username.to_valkey_profile_target()
        _assert_credentials(
            profile.profile_target(RedisRole.STATISTICS),
            username=BASE_USERNAME,
            password=BASE_PASSWORD,
        )
        _assert_credentials(
            profile.profile_target(RedisRole.STREAM),
            username=OVERRIDE_USERNAME,
            password=OVERRIDE_PASSWORD,
        )

    def test_redis_profile_target(self, redis_config_with_username: RedisConfig) -> None:
        profile = redis_config_with_username.to_redis_profile_target()
        _assert_credentials(
            profile.profile_target(RedisRole.STATISTICS),
            username=BASE_USERNAME,
            password=BASE_PASSWORD,
        )
        _assert_credentials(
            profile.profile_target(RedisRole.STREAM),
            username=OVERRIDE_USERNAME,
            password=OVERRIDE_PASSWORD,
        )
        _assert_credentials(
            profile.profile_target(RedisRole.STREAM).to_valkey_target(),
            username=OVERRIDE_USERNAME,
            password=OVERRIDE_PASSWORD,
        )

    def test_redis_profile_target_from_dict(self) -> None:
        profile = RedisProfileTarget.from_dict(_redis_config_data())
        _assert_credentials(
            profile.profile_target(RedisRole.STATISTICS),
            username=BASE_USERNAME,
            password=BASE_PASSWORD,
        )
        _assert_credentials(
            profile.profile_target(RedisRole.STREAM),
            username=OVERRIDE_USERNAME,
            password=OVERRIDE_PASSWORD,
        )

    def test_redis_target_copy(self, redis_target_with_username: RedisTarget) -> None:
        copied = redis_target_with_username.copy()
        assert copied == redis_target_with_username
        assert copied is not redis_target_with_username
        _assert_credentials(copied, username=BASE_USERNAME, password=BASE_PASSWORD)

    def test_redis_target_to_valkey_target(self, redis_target_with_username: RedisTarget) -> None:
        target = redis_target_with_username.to_valkey_target()
        _assert_credentials(target, username=BASE_USERNAME, password=BASE_PASSWORD)

    def test_standalone_target_from_valkey_target(
        self, valkey_standalone_target_with_username: ValkeyTarget
    ) -> None:
        target = ValkeyStandaloneTarget.from_valkey_target(valkey_standalone_target_with_username)
        _assert_credentials(target, username=BASE_USERNAME, password=BASE_PASSWORD)

    def test_sentinel_target_from_valkey_target(
        self, valkey_sentinel_target_with_username: ValkeyTarget
    ) -> None:
        target = ValkeySentinelTarget.from_valkey_target(valkey_sentinel_target_with_username)
        _assert_credentials(target, username=BASE_USERNAME, password=BASE_PASSWORD)

    def test_create_valkey_client_carries_username_to_both_clients(
        self, valkey_sentinel_target_with_username: ValkeyTarget
    ) -> None:
        with patch("ai.backend.common.clients.valkey_client.client.Sentinel"):
            client = create_valkey_client(
                valkey_sentinel_target_with_username, db_id=0, human_readable_name="test"
            )
        assert isinstance(client, MonitoringValkeyClient)
        for inner in (client._operation_client, client._monitor_client):
            assert isinstance(inner, ValkeySentinelClient)
            _assert_credentials(inner._target, username=BASE_USERNAME, password=BASE_PASSWORD)

    def test_monitor_client_differs_only_in_request_timeout(
        self, valkey_standalone_target_with_username: ValkeyTarget
    ) -> None:
        client = create_valkey_client(
            valkey_standalone_target_with_username, db_id=0, human_readable_name="test"
        )
        assert isinstance(client, MonitoringValkeyClient)
        operation = client._operation_client
        monitor = client._monitor_client
        assert isinstance(operation, ValkeyStandaloneClient)
        assert isinstance(monitor, ValkeyStandaloneClient)
        assert monitor._target.request_timeout != operation._target.request_timeout
        monitor._target.request_timeout = operation._target.request_timeout
        assert monitor._target == operation._target


class TestUsernameDefaults:
    """Without a username, every conversion keeps None, as before the field existed."""

    def test_config_without_username(self) -> None:
        config = RedisConfig.model_validate({
            "addr": "127.0.0.1:6379",
            "password": BASE_PASSWORD,
            "override_configs": {RedisRole.STREAM.value: {"addr": "127.0.0.1:6380"}},
        })
        for profile in (config.to_valkey_profile_target(), config.to_redis_profile_target()):
            _assert_credentials(
                profile.profile_target(RedisRole.STATISTICS), username=None, password=BASE_PASSWORD
            )
            _assert_credentials(
                profile.profile_target(RedisRole.STREAM), username=None, password=None
            )

    def test_from_dict_without_username(self) -> None:
        profile = RedisProfileTarget.from_dict({
            "addr": "127.0.0.1:6379",
            "password": BASE_PASSWORD,
        })
        target = profile.profile_target(RedisRole.STATISTICS).to_valkey_target()
        _assert_credentials(target, username=None, password=BASE_PASSWORD)


class TestOverridesDoNotInheritCredentials:
    """An override is a complete target, so it never borrows the base credentials."""

    def test_valkey_profile_target(self, redis_config_with_username: RedisConfig) -> None:
        target = redis_config_with_username.to_valkey_profile_target().profile_target(
            RedisRole.LIVE
        )
        _assert_credentials(target, username=None, password=None)

    def test_redis_profile_target(self, redis_config_with_username: RedisConfig) -> None:
        target = (
            redis_config_with_username.to_redis_profile_target()
            .profile_target(RedisRole.LIVE)
            .to_valkey_target()
        )
        _assert_credentials(target, username=None, password=None)

    def test_redis_profile_target_from_dict(self) -> None:
        target = RedisProfileTarget.from_dict(_redis_config_data()).profile_target(RedisRole.LIVE)
        _assert_credentials(target, username=None, password=None)

    def test_override_with_password_only_does_not_take_base_username(self) -> None:
        config = RedisConfig.model_validate({
            "addr": "127.0.0.1:6379",
            "username": BASE_USERNAME,
            "password": BASE_PASSWORD,
            "override_configs": {
                RedisRole.STREAM.value: {"addr": "127.0.0.1:6380", "password": OVERRIDE_PASSWORD},
            },
        })
        target = config.to_valkey_profile_target().profile_target(RedisRole.STREAM)
        _assert_credentials(target, username=None, password=OVERRIDE_PASSWORD)

    def test_trafaret_override_does_not_take_base_username(self) -> None:
        checked = redis_config_iv.check(_redis_config_data())
        assert checked["username"] == BASE_USERNAME
        assert checked["override_configs"][RedisRole.STREAM.value]["username"] == OVERRIDE_USERNAME
        assert checked["override_configs"][RedisRole.LIVE.value]["username"] is None
        assert checked["override_configs"][RedisRole.LIVE.value]["password"] is None


class TestScopedEtcdCredentials:
    """
    etcd's get_prefix() returns ChainMap(node, global), so credentials stored under the
    node scope take precedence over the shared, global connection settings.
    """

    @staticmethod
    def _resolve(etcd_value: Mapping[str, object]) -> ValkeyTarget:
        # Mirrors how the agent reads config/redis from etcd.
        checked = dict(redis_config_iv.check(etcd_value))
        addr = checked["addr"]
        checked["addr"] = f"{addr.host}:{addr.port}"
        config = RedisConfig.model_validate(checked)
        return config.to_redis_profile_target().profile_target(RedisRole.STREAM).to_valkey_target()

    def test_node_credentials_over_global_connection_settings(self) -> None:
        etcd_value = ChainMap(
            {"username": BASE_USERNAME, "password": BASE_PASSWORD},
            {"addr": "127.0.0.1:6379", "use_tls": "true", "tls_ca_file": "/etc/redis/ca.pem"},
        )
        target = self._resolve(etcd_value)
        assert target.addr == "127.0.0.1:6379"
        assert target.use_tls is True
        assert target.tls_ca_file == "/etc/redis/ca.pem"
        _assert_credentials(target, username=BASE_USERNAME, password=BASE_PASSWORD)

    def test_node_credentials_win_over_global_password(self) -> None:
        etcd_value = ChainMap(
            {"username": BASE_USERNAME, "password": BASE_PASSWORD},
            {"addr": "127.0.0.1:6379", "password": "shared-secret", "use_tls": "true"},
        )
        target = self._resolve(etcd_value)
        _assert_credentials(target, username=BASE_USERNAME, password=BASE_PASSWORD)

    def test_without_node_scope_falls_back_to_global(self) -> None:
        etcd_value = ChainMap(
            {}, {"addr": "127.0.0.1:6379", "password": "shared-secret", "use_tls": "true"}
        )
        target = self._resolve(etcd_value)
        _assert_credentials(target, username=None, password="shared-secret")


class TestBuildServerCredentials:
    def test_password_only(self) -> None:
        credentials = build_server_credentials(username=None, password=BASE_PASSWORD)
        assert credentials is not None
        assert credentials.password == BASE_PASSWORD
        assert credentials.username is None

    def test_password_only_calls_glide_as_before(self) -> None:
        with patch(
            "ai.backend.common.clients.valkey_client.client.ServerCredentials"
        ) as mock_credentials:
            build_server_credentials(username=None, password=BASE_PASSWORD)
        mock_credentials.assert_called_once_with(password=BASE_PASSWORD)

    def test_username_and_password(self) -> None:
        credentials = build_server_credentials(username=BASE_USERNAME, password=BASE_PASSWORD)
        assert credentials is not None
        assert credentials.password == BASE_PASSWORD
        assert credentials.username == BASE_USERNAME

    def test_empty_username_is_no_username(self) -> None:
        credentials = build_server_credentials(username="", password=BASE_PASSWORD)
        assert credentials is not None
        assert credentials.username is None

    @pytest.mark.parametrize("password", [None, ""])
    def test_no_credentials(self, password: str | None) -> None:
        assert build_server_credentials(username=None, password=password) is None

    @pytest.mark.parametrize("password", [None, ""])
    def test_username_without_password_is_refused(self, password: str | None) -> None:
        with pytest.raises(InvalidConfigError) as exc_info:
            build_server_credentials(username=BASE_USERNAME, password=password)
        assert BASE_USERNAME not in str(exc_info.value)
        assert BASE_USERNAME not in repr(exc_info.value.args)


class TestGlideClientCredentials:
    async def test_standalone_sends_username_and_password(
        self, valkey_standalone_target_with_username: ValkeyTarget
    ) -> None:
        client = ValkeyStandaloneClient(
            ValkeyStandaloneTarget.from_valkey_target(valkey_standalone_target_with_username),
            db_id=0,
            human_readable_name="test",
        )
        with patch(
            "ai.backend.common.clients.valkey_client.client._create_glide_client",
            new_callable=AsyncMock,
        ) as mock_create:
            await client.connect()
        config: GlideClientConfiguration = mock_create.call_args.args[0]
        assert isinstance(config.credentials, ServerCredentials)
        assert config.credentials.username == BASE_USERNAME
        assert config.credentials.password == BASE_PASSWORD

    async def test_standalone_refuses_username_without_password(
        self, valkey_standalone_target_without_password: ValkeyTarget
    ) -> None:
        client = ValkeyStandaloneClient(
            ValkeyStandaloneTarget.from_valkey_target(valkey_standalone_target_without_password),
            db_id=0,
            human_readable_name="test",
        )
        with patch(
            "ai.backend.common.clients.valkey_client.client._create_glide_client",
            new_callable=AsyncMock,
        ) as mock_create:
            with pytest.raises(InvalidConfigError):
                await client.connect()
        mock_create.assert_not_awaited()

    async def test_sentinel_sends_username_to_master_only(
        self, valkey_sentinel_target_with_username: ValkeyTarget
    ) -> None:
        with patch("ai.backend.common.clients.valkey_client.client.Sentinel") as mock_sentinel_cls:
            client = ValkeySentinelClient(
                ValkeySentinelTarget.from_valkey_target(valkey_sentinel_target_with_username),
                db_id=0,
                human_readable_name="test",
            )
        sentinel_kwargs = mock_sentinel_cls.call_args.kwargs["sentinel_kwargs"]
        assert "username" not in sentinel_kwargs
        client._sentinel = MagicMock()
        client._sentinel.discover_master = AsyncMock(return_value=("127.0.0.1", 6379))
        with patch(
            "ai.backend.common.clients.valkey_client.client._create_glide_client",
            new_callable=AsyncMock,
        ) as mock_create:
            await client.connect()
        config: GlideClientConfiguration = mock_create.call_args.args[0]
        assert isinstance(config.credentials, ServerCredentials)
        assert config.credentials.username == BASE_USERNAME
        assert config.credentials.password == BASE_PASSWORD
