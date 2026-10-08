from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.backend.common.clients.valkey_client.valkey_stream.client import (
    AutoClaimMessage,
    StreamMessage,
)
from ai.backend.common.message_queue.redis_queue.consumer import RedisConsumer, RedisConsumerArgs
from ai.backend.common.message_queue.types import MQMessage
from ai.backend.common.types import RedisTarget

GROUP = "test-group"

type ConsumerFactory = Callable[
    [set[str], dict[str, list[StreamMessage]]], Coroutine[Any, Any, RedisConsumer]
]


def _make_reader(pending: dict[str, list[StreamMessage]]) -> MagicMock:
    """A reader client that returns the given messages once per stream, then blocks."""
    blocker = asyncio.Event()

    async def read_consumer_group(
        stream_key: str, group_name: str, consumer_name: str, **kwargs: Any
    ) -> list[StreamMessage] | None:
        messages = pending.pop(stream_key, None)
        if messages:
            return messages
        await blocker.wait()
        return None

    reader = MagicMock()
    reader.read_consumer_group = AsyncMock(side_effect=read_consumer_group)
    reader.close = AsyncMock()
    return reader


@pytest.fixture
def client() -> MagicMock:
    client = MagicMock()
    client.done_stream_message = AsyncMock()
    client.reque_stream_message = AsyncMock()
    client.auto_claim_stream_message = AsyncMock(return_value=None)
    client.close = AsyncMock()
    return client


@pytest.fixture
async def make_consumer(client: MagicMock) -> AsyncIterator[ConsumerFactory]:
    consumers: list[RedisConsumer] = []
    patches: list[Any] = []

    async def factory(
        stream_keys: set[str], pending: dict[str, list[StreamMessage]]
    ) -> RedisConsumer:
        p = patch(
            "ai.backend.common.message_queue.redis_queue.consumer.ValkeyStreamClient.create",
            new=AsyncMock(return_value=_make_reader(pending)),
        )
        p.start()
        patches.append(p)
        consumer = RedisConsumer(
            client,
            RedisTarget(),
            RedisConsumerArgs(stream_keys=stream_keys, group_name=GROUP, node_id="test-node"),
        )
        consumers.append(consumer)
        return consumer

    yield factory
    for consumer in consumers:
        await consumer.close()
    for p in patches:
        p.stop()


async def _next_messages(consumer: RedisConsumer, count: int) -> list[MQMessage]:
    messages: list[MQMessage] = []

    async def collect() -> None:
        async for message in consumer.consume_queue():
            messages.append(message)
            if len(messages) == count:
                return

    await asyncio.wait_for(collect(), timeout=5)
    return messages


class TestStreamKey:
    async def test_message_carries_the_stream_it_was_read_from(
        self, make_consumer: ConsumerFactory
    ) -> None:
        consumer = await make_consumer(
            {"events", "events:agent:a"},
            {
                "events": [StreamMessage(b"1-0", {b"name": b"x"})],
                "events:agent:a": [StreamMessage(b"1-0", {b"name": b"y"})],
            },
        )

        messages = await _next_messages(consumer, 2)

        by_name = {m.payload[b"name"]: m.stream_key for m in messages}
        assert by_name == {b"x": "events", b"y": "events:agent:a"}


class TestDone:
    async def test_acks_on_the_given_stream_only(
        self, make_consumer: ConsumerFactory, client: MagicMock
    ) -> None:
        consumer = await make_consumer({"events", "events:agent:a"}, {})

        await consumer.done(b"1-0", stream_key="events:agent:a")

        client.done_stream_message.assert_awaited_once_with("events:agent:a", GROUP, b"1-0")

    async def test_single_stream_is_acked_without_stream_key(
        self, make_consumer: ConsumerFactory, client: MagicMock
    ) -> None:
        consumer = await make_consumer({"events"}, {})

        await consumer.done(b"1-0")

        client.done_stream_message.assert_awaited_once_with("events", GROUP, b"1-0")

    async def test_missing_stream_key_with_several_streams_is_refused(
        self, make_consumer: ConsumerFactory, client: MagicMock
    ) -> None:
        consumer = await make_consumer({"events", "events:agent:a"}, {})

        with pytest.raises(ValueError):
            await consumer.done(b"1-0")

        client.done_stream_message.assert_not_awaited()

    async def test_stream_not_consumed_is_refused(
        self, make_consumer: ConsumerFactory, client: MagicMock
    ) -> None:
        consumer = await make_consumer({"events"}, {})

        with pytest.raises(ValueError):
            await consumer.done(b"1-0", stream_key="events:agent:other")

        client.done_stream_message.assert_not_awaited()

    async def test_ack_failure_is_not_raised(
        self, make_consumer: ConsumerFactory, client: MagicMock
    ) -> None:
        consumer = await make_consumer({"events"}, {})
        client.done_stream_message.side_effect = RuntimeError("connection lost")

        await consumer.done(b"1-0", stream_key="events")

        client.done_stream_message.assert_awaited_once()


class TestAutoClaim:
    async def test_malformed_retry_count_is_discarded_on_its_stream(
        self, make_consumer: ConsumerFactory, client: MagicMock
    ) -> None:
        consumer = await make_consumer(set(), {})
        client.auto_claim_stream_message.return_value = AutoClaimMessage(
            next_start_id=b"0-0",
            messages=[
                StreamMessage(b"1-0", {b"name": b"x", b"_retry_count": b"not-a-number"}),
                StreamMessage(b"2-0", {b"name": b"x", b"_retry_count": b"-5"}),
                StreamMessage(b"3-0", {b"name": b"x", b"_retry_count": b"0"}),
            ],
        )

        _, claimed = await consumer._auto_claim("events:agent:a", "0-0", 1000)

        assert claimed is True
        assert [c.args for c in client.done_stream_message.await_args_list] == [
            ("events:agent:a", GROUP, b"1-0"),
            ("events:agent:a", GROUP, b"2-0"),
        ]
        client.reque_stream_message.assert_awaited_once_with(
            "events:agent:a",
            GROUP,
            b"3-0",
            {b"name": b"x", b"_retry_count": b"1"},
        )
