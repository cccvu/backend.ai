import pytest

from ai.backend.common.exception import BackendAISchemaValidationFailed
from ai.backend.common.typed_validators import HostPortPair
from ai.backend.manager.config.unified import ManagerConfig, MetricConfig


def test_config_validation_supports_field_name_and_alias() -> None:
    config = MetricConfig.model_validate({"address": "127.0.0.1:9090"}, by_name=True)
    assert config.address == HostPortPair(host="127.0.0.1", port=9090)

    config = MetricConfig.model_validate({"addr": "127.0.0.1:9090"}, by_name=True)
    assert config.address == HostPortPair(host="127.0.0.1", port=9090)


class TestExtraEventStreams:
    def test_defaults_are_empty(self) -> None:
        config = ManagerConfig.model_validate({})
        assert config.extra_event_stream_keys == []
        assert config.extra_event_channels == []

    def test_kebab_and_snake_aliases(self) -> None:
        kebab = ManagerConfig.model_validate({
            "extra-event-stream-keys": ["events:agent:a"],
            "extra-event-channels": ["events_all:agent:a"],
        })
        snake = ManagerConfig.model_validate({
            "extra_event_stream_keys": ["events:agent:a"],
            "extra_event_channels": ["events_all:agent:a"],
        })
        for config in (kebab, snake):
            assert config.extra_event_stream_keys == ["events:agent:a"]
            assert config.extra_event_channels == ["events_all:agent:a"]
        dumped = kebab.model_dump(by_alias=True)
        assert dumped["extra-event-stream-keys"] == ["events:agent:a"]
        assert dumped["extra-event-channels"] == ["events_all:agent:a"]

    @pytest.mark.parametrize("key", ["extra-event-stream-keys", "extra-event-channels"])
    def test_empty_name_is_rejected(self, key: str) -> None:
        with pytest.raises(BackendAISchemaValidationFailed):
            ManagerConfig.model_validate({key: ["events:agent:a", ""]})
