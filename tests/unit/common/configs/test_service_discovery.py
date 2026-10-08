from __future__ import annotations

import pytest

from ai.backend.common.configs.service_discovery import ServiceDiscoveryConfig
from ai.backend.common.exception import BackendAISchemaValidationFailed
from ai.backend.common.types import ServiceDiscoveryType


class TestServiceDiscoveryEnabled:
    def test_enabled_by_default(self) -> None:
        config = ServiceDiscoveryConfig.model_validate({})
        assert config.enabled is True
        assert config.type == ServiceDiscoveryType.REDIS

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (False, False),
            (True, True),
            ("false", False),
            ("true", True),
        ],
    )
    def test_parses_enabled(self, raw: object, expected: bool) -> None:
        config = ServiceDiscoveryConfig.model_validate({"enabled": raw, "type": "redis"})
        assert config.enabled is expected

    def test_disabled_keeps_other_fields(self) -> None:
        config = ServiceDiscoveryConfig.model_validate({
            "enabled": False,
            "type": "etcd",
            "service-group": "agent",
        })
        assert config.enabled is False
        assert config.type == ServiceDiscoveryType.ETCD
        assert config.service_group == "agent"

    def test_rejects_non_boolean(self) -> None:
        with pytest.raises(BackendAISchemaValidationFailed):
            ServiceDiscoveryConfig.model_validate({"enabled": "sometimes"})
