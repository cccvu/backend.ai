from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ai.backend.common.clients.valkey_client.client import (
    MonitoringValkeyClient,
    ValkeySentinelClient,
    ValkeySentinelTarget,
    ValkeyStandaloneTarget,
    build_tls_config,
    create_valkey_client,
)
from ai.backend.common.configs.redis import RedisConfig, SingleRedisConfig
from ai.backend.common.defs import RedisRole
from ai.backend.common.exception import InvalidConfigError
from ai.backend.common.types import RedisProfileTarget, RedisTarget, ValkeyTarget

BASE_CA_FILE = "/etc/redis/base-ca.pem"
OVERRIDE_CA_FILE = "/etc/redis/override-ca.pem"
PEM_DATA = b"-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n"

type _TlsTarget = ValkeyTarget | RedisTarget | ValkeyStandaloneTarget | ValkeySentinelTarget


def _assert_tls(target: _TlsTarget, *, skip_verify: bool, ca_file: str) -> None:
    assert target.use_tls is True
    assert target.tls_skip_verify is skip_verify
    assert target.tls_ca_file == ca_file


@pytest.fixture
def redis_config_with_tls() -> RedisConfig:
    # String values mirror what is read from etcd.
    return RedisConfig.model_validate({
        "addr": "127.0.0.1:6379",
        "use-tls": "true",
        "tls-skip-verify": "false",
        "tls-ca-file": BASE_CA_FILE,
        "override_configs": {
            RedisRole.STREAM.value: {
                "addr": "127.0.0.1:6380",
                "use_tls": "true",
                "tls_skip_verify": "true",
                "tls_ca_file": OVERRIDE_CA_FILE,
            },
        },
    })


@pytest.fixture
def single_redis_config_with_tls() -> SingleRedisConfig:
    return SingleRedisConfig.model_validate({
        "addr": "127.0.0.1:6379",
        "use-tls": "true",
        "tls-skip-verify": "false",
        "tls-ca-file": BASE_CA_FILE,
    })


@pytest.fixture
def redis_target_with_tls() -> RedisTarget:
    return RedisTarget(
        sentinel="127.0.0.1:26379",
        service_name="mymaster",
        use_tls=True,
        tls_ca_file=BASE_CA_FILE,
    )


@pytest.fixture
def valkey_sentinel_target_with_tls() -> ValkeyTarget:
    return ValkeyTarget(
        sentinel=["127.0.0.1:26379"],
        service_name="mymaster",
        use_tls=True,
        tls_ca_file=BASE_CA_FILE,
    )


@pytest.fixture
def ca_file(tmp_path: Path) -> Path:
    path = tmp_path / "ca.pem"
    path.write_bytes(PEM_DATA)
    return path


@pytest.fixture
def empty_ca_file(tmp_path: Path) -> Path:
    path = tmp_path / "empty.pem"
    path.write_bytes(b"")
    return path


class TestTlsSettingsPropagation:
    """Verify the TLS settings survive every conversion between config and targets."""

    def test_single_redis_config_to_valkey_target(
        self, single_redis_config_with_tls: SingleRedisConfig
    ) -> None:
        target = single_redis_config_with_tls.to_valkey_target()
        _assert_tls(target, skip_verify=False, ca_file=BASE_CA_FILE)

    def test_single_redis_config_to_redis_target(
        self, single_redis_config_with_tls: SingleRedisConfig
    ) -> None:
        target = single_redis_config_with_tls.to_redis_target()
        _assert_tls(target, skip_verify=False, ca_file=BASE_CA_FILE)

    def test_valkey_profile_target(self, redis_config_with_tls: RedisConfig) -> None:
        profile = redis_config_with_tls.to_valkey_profile_target()
        _assert_tls(profile.profile_target(RedisRole.LIVE), skip_verify=False, ca_file=BASE_CA_FILE)
        _assert_tls(
            profile.profile_target(RedisRole.STREAM), skip_verify=True, ca_file=OVERRIDE_CA_FILE
        )

    def test_redis_profile_target(self, redis_config_with_tls: RedisConfig) -> None:
        profile = redis_config_with_tls.to_redis_profile_target()
        _assert_tls(profile.profile_target(RedisRole.LIVE), skip_verify=False, ca_file=BASE_CA_FILE)
        _assert_tls(
            profile.profile_target(RedisRole.STREAM), skip_verify=True, ca_file=OVERRIDE_CA_FILE
        )

    def test_redis_profile_target_from_dict(self) -> None:
        profile = RedisProfileTarget.from_dict({
            "addr": "127.0.0.1:6379",
            "use_tls": True,
            "tls_skip_verify": False,
            "tls_ca_file": BASE_CA_FILE,
            "override_configs": {
                RedisRole.STREAM.value: {
                    "addr": "127.0.0.1:6380",
                    "use_tls": True,
                    "tls_skip_verify": True,
                    "tls_ca_file": OVERRIDE_CA_FILE,
                },
            },
        })
        _assert_tls(profile.profile_target(RedisRole.LIVE), skip_verify=False, ca_file=BASE_CA_FILE)
        _assert_tls(
            profile.profile_target(RedisRole.STREAM), skip_verify=True, ca_file=OVERRIDE_CA_FILE
        )

    def test_redis_target_copy(self, redis_target_with_tls: RedisTarget) -> None:
        _assert_tls(redis_target_with_tls.copy(), skip_verify=False, ca_file=BASE_CA_FILE)

    def test_redis_target_to_valkey_target(self, redis_target_with_tls: RedisTarget) -> None:
        target = redis_target_with_tls.to_valkey_target()
        _assert_tls(target, skip_verify=False, ca_file=BASE_CA_FILE)

    def test_standalone_target_from_valkey_target(
        self, single_redis_config_with_tls: SingleRedisConfig
    ) -> None:
        target = ValkeyStandaloneTarget.from_valkey_target(
            single_redis_config_with_tls.to_valkey_target()
        )
        _assert_tls(target, skip_verify=False, ca_file=BASE_CA_FILE)

    def test_sentinel_target_from_valkey_target(
        self, valkey_sentinel_target_with_tls: ValkeyTarget
    ) -> None:
        target = ValkeySentinelTarget.from_valkey_target(valkey_sentinel_target_with_tls)
        _assert_tls(target, skip_verify=False, ca_file=BASE_CA_FILE)

    def test_create_valkey_client_carries_tls_to_both_clients(
        self, valkey_sentinel_target_with_tls: ValkeyTarget
    ) -> None:
        with patch("ai.backend.common.clients.valkey_client.client.Sentinel"):
            client = create_valkey_client(
                valkey_sentinel_target_with_tls, db_id=0, human_readable_name="test"
            )
        assert isinstance(client, MonitoringValkeyClient)
        for inner in (client._operation_client, client._monitor_client):
            assert isinstance(inner, ValkeySentinelClient)
            _assert_tls(inner._target, skip_verify=False, ca_file=BASE_CA_FILE)

    def test_sentinel_connections_verify_with_ca_file(
        self, valkey_sentinel_target_with_tls: ValkeyTarget
    ) -> None:
        with patch("ai.backend.common.clients.valkey_client.client.Sentinel") as mock_sentinel_cls:
            ValkeySentinelClient(
                target=ValkeySentinelTarget.from_valkey_target(valkey_sentinel_target_with_tls),
                db_id=0,
                human_readable_name="test",
            )
        sentinel_kwargs = mock_sentinel_cls.call_args.kwargs["sentinel_kwargs"]
        assert sentinel_kwargs["ssl"] is True
        assert sentinel_kwargs["ssl_ca_certs"] == BASE_CA_FILE


class TestBuildTlsConfig:
    def test_reads_ca_file_when_tls_is_enabled(self, ca_file: Path) -> None:
        tls_config = build_tls_config(use_tls=True, tls_skip_verify=False, tls_ca_file=str(ca_file))
        assert tls_config.root_pem_cacerts == PEM_DATA
        assert tls_config.use_insecure_tls is False

    def test_uses_default_trust_store_without_ca_file(self) -> None:
        tls_config = build_tls_config(use_tls=True, tls_skip_verify=False, tls_ca_file=None)
        assert tls_config.root_pem_cacerts is None

    def test_omits_ca_file_without_tls(self, ca_file: Path) -> None:
        tls_config = build_tls_config(
            use_tls=False, tls_skip_verify=False, tls_ca_file=str(ca_file)
        )
        assert tls_config.root_pem_cacerts is None

    def test_unreadable_ca_file_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.pem"
        with pytest.raises(InvalidConfigError, match="Cannot read the TLS CA file"):
            build_tls_config(use_tls=True, tls_skip_verify=False, tls_ca_file=str(missing))

    def test_empty_ca_file_raises(self, empty_ca_file: Path) -> None:
        with pytest.raises(InvalidConfigError, match="is empty"):
            build_tls_config(use_tls=True, tls_skip_verify=False, tls_ca_file=str(empty_ca_file))
