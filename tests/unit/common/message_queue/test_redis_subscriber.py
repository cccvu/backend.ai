from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.backend.common.message_queue.redis_queue import RedisMQArgs, RedisQueue
from ai.backend.common.message_queue.redis_queue.subscriber import NoopSubscriber, RedisSubscriber
from ai.backend.common.message_queue.types import BroadcastMessage
from ai.backend.common.types import RedisTarget


def _make_client(messages: list[tuple[str, bytes]]) -> MagicMock:
    """A pub/sub client that returns the given messages in order, then blocks."""
    blocker = asyncio.Event()

    async def receive() -> tuple[str, bytes]:
        if messages:
            return messages.pop(0)
        await blocker.wait()
        raise AssertionError("unreachable")

    client = MagicMock()
    client.receive_broadcast_message_with_channel = AsyncMock(side_effect=receive)
    client.close = AsyncMock()
    return client


@pytest.fixture
async def subscribers() -> AsyncIterator[list[RedisSubscriber]]:
    created: list[RedisSubscriber] = []
    yield created
    for subscriber in created:
        await subscriber.close()


async def _first(subscriber: RedisSubscriber) -> BroadcastMessage:
    async for message in subscriber.subscribe_queue():
        return message
    raise AssertionError("subscriber stopped")


class TestRedisSubscriber:
    async def test_message_carries_its_channel(self, subscribers: list[RedisSubscriber]) -> None:
        client = _make_client([("events_all:agent:a", b'{"name": "x"}')])
        subscriber = RedisSubscriber(client, {"events_all", "events_all:agent:a"})
        subscribers.append(subscriber)

        message = await asyncio.wait_for(_first(subscriber), timeout=5)

        assert message == BroadcastMessage({"name": "x"}, channel="events_all:agent:a")

    async def test_undecodable_messages_are_dropped_without_delay(
        self, subscribers: list[RedisSubscriber]
    ) -> None:
        invalid: list[tuple[str, bytes]] = [
            ("events_all:agent:a", payload)
            for payload in (b"not json", b"\xff\xfe", b"[1, 2]", b'"text"', b"null")
        ] * 20
        client = _make_client([*invalid, ("events_all", b'{"name": "valid"}')])
        subscriber = RedisSubscriber(client, {"events_all", "events_all:agent:a"})
        subscribers.append(subscriber)

        # The subscriber used to sleep for 1 second after each undecodable message.
        message = await asyncio.wait_for(_first(subscriber), timeout=0.5)

        assert message == BroadcastMessage({"name": "valid"}, channel="events_all")


class TestNoopSubscriber:
    async def test_yields_nothing_and_stops_on_close(self) -> None:
        subscriber = NoopSubscriber()
        received: list[BroadcastMessage] = []

        async def consume() -> None:
            async for message in subscriber.subscribe_queue():
                received.append(message)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        assert not task.done()

        await subscriber.close()
        await asyncio.wait_for(task, timeout=1)
        assert received == []


class TestRedisQueueSubscriber:
    @pytest.mark.parametrize("channels", [None, set()])
    async def test_no_channels_opens_no_subscriber(self, channels: set[str] | None) -> None:
        args = RedisMQArgs(
            anycast_stream_key="events:agent:a",
            broadcast_channel="events_all:agent:a",
            consume_stream_keys=None,
            subscribe_channels=channels,
            group_name="test-group",
            node_id="test-node",
            db=0,
        )
        module = "ai.backend.common.message_queue.redis_queue.queue"
        with (
            patch(f"{module}.RedisAnycaster.create", new=AsyncMock()),
            patch(f"{module}.RedisBroadcaster.create", new=AsyncMock()),
            patch(f"{module}.RedisConsumer.create", new=AsyncMock()),
            patch(f"{module}.RedisSubscriber.create", new=AsyncMock()) as mock_subscriber,
        ):
            queue = await RedisQueue.create(RedisTarget(), args)

        mock_subscriber.assert_not_awaited()
        assert isinstance(queue._subscriber, NoopSubscriber)

    async def test_channels_open_a_subscriber(self) -> None:
        args = RedisMQArgs(
            anycast_stream_key="events",
            broadcast_channel="events_all",
            consume_stream_keys=None,
            subscribe_channels={"events_all"},
            group_name="test-group",
            node_id="test-node",
            db=0,
        )
        module = "ai.backend.common.message_queue.redis_queue.queue"
        with (
            patch(f"{module}.RedisAnycaster.create", new=AsyncMock()),
            patch(f"{module}.RedisBroadcaster.create", new=AsyncMock()),
            patch(f"{module}.RedisConsumer.create", new=AsyncMock()),
            patch(f"{module}.RedisSubscriber.create", new=AsyncMock()) as mock_subscriber,
        ):
            await RedisQueue.create(RedisTarget(), args)

        mock_subscriber.assert_awaited_once()
        assert mock_subscriber.await_args is not None
        assert mock_subscriber.await_args.args[1] == {"events_all"}
