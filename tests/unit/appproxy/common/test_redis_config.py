from __future__ import annotations

import pytest

from ai.backend.appproxy.common.config import RedisConfig
from ai.backend.common.defs import RedisRole
from ai.backend.common.types import RedisProfileTarget

CA_FILE = "/etc/redis/ca.pem"


@pytest.fixture
def redis_config_with_tls() -> RedisConfig:
    return RedisConfig.model_validate({
        "addr": {"host": "127.0.0.1", "port": 6379},
        "use_tls": True,
        "tls_skip_verify": False,
        "tls_ca_file": CA_FILE,
    })


class TestRedisConfigTls:
    def test_tls_disabled_by_default(self) -> None:
        config = RedisConfig.model_validate({"addr": {"host": "127.0.0.1", "port": 6379}})
        target = RedisProfileTarget.from_dict(config.to_dict()).profile_target(RedisRole.LIVE)
        assert target.use_tls is False
        assert target.tls_ca_file is None

    def test_valkey_target_carries_tls(self, redis_config_with_tls: RedisConfig) -> None:
        profile = RedisProfileTarget.from_dict(redis_config_with_tls.to_dict())
        target = profile.profile_target(RedisRole.LIVE).to_valkey_target()
        assert target.use_tls is True
        assert target.tls_skip_verify is False
        assert target.tls_ca_file == CA_FILE
