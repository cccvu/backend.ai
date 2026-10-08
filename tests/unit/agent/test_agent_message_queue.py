from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.backend.agent.agent import AbstractAgent
from ai.backend.agent.config.unified import OverridableAgentConfig
from ai.backend.common.message_queue.redis_queue import RedisMQArgs
from ai.backend.common.types import RedisTarget


async def _make_args(agent_config: OverridableAgentConfig, *, experimental: bool) -> RedisMQArgs:
    fake_agent: Any = SimpleNamespace(
        id="agent-1",
        local_config=SimpleNamespace(
            agent=SimpleNamespace(
                event_stream_key=agent_config.event_stream_key,
                event_channel=agent_config.event_channel,
                event_subscribe_channels=agent_config.event_subscribe_channels,
                use_experimental_redis_event_dispatcher=experimental,
            ),
        ),
    )
    with (
        patch("ai.backend.agent.agent.RedisQueue.create", new=AsyncMock()) as mock_create,
        patch("ai.backend.agent.agent.HiRedisQueue", new=MagicMock()) as mock_hiredis,
    ):
        await AbstractAgent._make_message_queue(fake_agent, RedisTarget())
    mock = mock_hiredis if experimental else mock_create
    args: RedisMQArgs = mock.call_args.args[1]
    return args


class TestAgentMessageQueue:
    @pytest.mark.parametrize("experimental", [False, True])
    async def test_defaults_are_unchanged(self, experimental: bool) -> None:
        args = await _make_args(
            OverridableAgentConfig.model_validate({}), experimental=experimental
        )

        assert args.anycast_stream_key == "events"
        assert args.broadcast_channel == "events_all"
        assert args.consume_stream_keys is None
        assert args.subscribe_channels == {"events_all"}

    @pytest.mark.parametrize("experimental", [False, True])
    async def test_producer_uses_its_own_stream_and_channel(self, experimental: bool) -> None:
        config = OverridableAgentConfig.model_validate({
            "event-stream-key": "events:agent:agent-1",
            "event-channel": "events_all:agent:agent-1",
            "event-subscribe-channels": [],
        })

        args = await _make_args(config, experimental=experimental)

        assert args.anycast_stream_key == "events:agent:agent-1"
        assert args.broadcast_channel == "events_all:agent:agent-1"
        # A producer-only agent neither consumes a stream nor subscribes to a channel.
        assert args.consume_stream_keys is None
        assert not args.subscribe_channels
