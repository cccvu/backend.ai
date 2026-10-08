from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.backend.common.exception import BackendAISchemaValidationFailed
from ai.backend.common.message_queue.redis_queue import RedisMQArgs
from ai.backend.storage.config.unified import StorageProxyConfig
from ai.backend.storage.server import _make_message_queue

_REQUIRED: dict[str, Any] = {
    "node-id": "storage-1",
    "secret": "some-secret",
    "session-expire": "1d",
}


async def _make_args(config: StorageProxyConfig, *, experimental: bool) -> RedisMQArgs:
    config = config.model_copy(update={"use_experimental_redis_event_dispatcher": experimental})
    local_config: Any = SimpleNamespace(storage_proxy=config)
    with (
        patch("ai.backend.storage.server.RedisQueue.create", new=AsyncMock()) as mock_create,
        patch("ai.backend.storage.server.HiRedisQueue", new=MagicMock()) as mock_hiredis,
    ):
        await _make_message_queue(local_config, MagicMock())
    mock = mock_hiredis if experimental else mock_create
    args: RedisMQArgs = mock.call_args.args[1]
    return args


class TestStorageProxyEventConfig:
    def test_defaults_are_unchanged(self) -> None:
        config = StorageProxyConfig.model_validate(_REQUIRED)

        assert config.event_stream_key == "events"
        assert config.event_channel == "events_all"
        assert config.event_subscribe_channels == ["events_all"]

    def test_kebab_and_snake_aliases(self) -> None:
        for keys in (
            ("event-stream-key", "event-channel", "event-subscribe-channels"),
            ("event_stream_key", "event_channel", "event_subscribe_channels"),
        ):
            config = StorageProxyConfig.model_validate({
                **_REQUIRED,
                keys[0]: "events:storage",
                keys[1]: "events_all:storage",
                keys[2]: [],
            })
            assert config.event_stream_key == "events:storage"
            assert config.event_channel == "events_all:storage"
            assert config.event_subscribe_channels == []

    @pytest.mark.parametrize(
        "override",
        [
            {"event-stream-key": ""},
            {"event-channel": ""},
            {"event-subscribe-channels": [""]},
        ],
    )
    def test_empty_names_are_rejected(self, override: dict[str, Any]) -> None:
        with pytest.raises(BackendAISchemaValidationFailed):
            StorageProxyConfig.model_validate({**_REQUIRED, **override})


class TestStorageProxyMessageQueue:
    @pytest.mark.parametrize("experimental", [False, True])
    async def test_defaults_are_unchanged(self, experimental: bool) -> None:
        args = await _make_args(
            StorageProxyConfig.model_validate(_REQUIRED), experimental=experimental
        )

        assert args.anycast_stream_key == "events"
        assert args.broadcast_channel == "events_all"
        assert args.consume_stream_keys is None
        assert args.subscribe_channels == {"events_all"}

    @pytest.mark.parametrize("experimental", [False, True])
    async def test_producer_uses_its_own_stream_and_channel(self, experimental: bool) -> None:
        config = StorageProxyConfig.model_validate({
            **_REQUIRED,
            "event-stream-key": "events:storage",
            "event-channel": "events_all:storage",
            "event-subscribe-channels": [],
        })

        args = await _make_args(config, experimental=experimental)

        assert args.anycast_stream_key == "events:storage"
        assert args.broadcast_channel == "events_all:storage"
        assert args.consume_stream_keys is None
        assert not args.subscribe_channels
