from __future__ import annotations

import asyncio
import random
import secrets
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, override

import pytest

from ai.backend.common import redis_helper
from ai.backend.common.defs import REDIS_STREAM_DB
from ai.backend.common.events.dispatcher import EventDispatcher, EventProducer
from ai.backend.common.events.types import AbstractAnycastEvent, EventDomain
from ai.backend.common.events.user_event.user_event import UserEvent
from ai.backend.common.message_queue.redis_queue import RedisMQArgs, RedisQueue
from ai.backend.common.message_queue.types import BroadcastMessage, MQMessage
from ai.backend.common.types import (
    AgentId,
    HostPortPair,
    RedisConnectionInfo,
    RedisHelperConfig,
    RedisTarget,
)


@pytest.fixture
async def redis_conn(
    redis_container: tuple[str, HostPortPair],
) -> AsyncGenerator[RedisConnectionInfo, None]:
    # Configure test Redis connection
    addr = redis_container[1]
    conn = redis_helper.get_redis_object(
        RedisTarget(
            addr=addr,
            redis_helper_config=RedisHelperConfig(
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
                reconnect_poll_timeout=1.0,
                max_connections=10,
                connection_ready_timeout=1.0,
            ),
        ),
        name="test-redis",
    )
    yield conn
    # Cleanup after tests
    await conn.client.flushdb()
    await conn.close()


@pytest.fixture
def queue_args() -> RedisMQArgs:
    return RedisMQArgs(
        anycast_stream_key="test-stream",
        broadcast_channel="test-broadcast",
        consume_stream_keys={
            "test-stream",
        },
        subscribe_channels={
            "test-broadcast",
        },
        group_name="test-group",
        node_id="test-node",
        db=REDIS_STREAM_DB,
    )


@pytest.fixture(scope="function")
async def redis_queue(
    redis_container: tuple[str, HostPortPair], queue_args: RedisMQArgs
) -> AsyncGenerator[RedisQueue, None]:
    # Create consumer group if not exists
    addr = redis_container[1]
    redis_target = RedisTarget(
        addr=addr,
        redis_helper_config={
            "socket_timeout": 5.0,
            "socket_connect_timeout": 2.0,
            "reconnect_poll_timeout": 0.3,
        },
    )
    queue = await RedisQueue.create(redis_target, queue_args)
    yield queue
    async with queue._anycaster._client._client.client() as conn:  # type: ignore[attr-defined]
        await conn.flushdb()
    async with queue._broadcaster._client._client.client() as conn:  # type: ignore[attr-defined]
        await conn.flushdb()
    async with queue._consumer._client._client.client() as conn:  # type: ignore[attr-defined]
        await conn.flushdb()
    async with queue._subscriber._client._client.client() as conn:  # type: ignore[attr-defined]
        await conn.flushdb()
    await queue.close()


async def test_send_and_consume(redis_queue: RedisQueue) -> None:
    # Test message sending and consuming
    test_payload = {b"key": b"value", b"key2": b"value2"}

    # Send message
    await redis_queue.send(test_payload)

    # Consume message
    async for message in redis_queue.consume_queue():
        assert isinstance(message, MQMessage)
        assert message.payload == test_payload
        await redis_queue.done(message.msg_id)
        break


async def test_subscribe(redis_queue: RedisQueue) -> None:
    # Test message subscription
    test_payload = {"key": "value", "key2": "value2"}

    # Create task to subscribe
    received_messages: list[BroadcastMessage] = []

    async def subscriber() -> None:
        async for message in redis_queue.subscribe_queue():
            received_messages.append(message)
            if len(received_messages) >= 1:
                break

    subscriber_task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.1)  # Allow subscriber to start

    # Send message
    await redis_queue.broadcast(test_payload)

    # Wait for message to be received
    await asyncio.wait_for(subscriber_task, timeout=5)

    assert len(received_messages) == 1
    assert received_messages[0].payload == test_payload


async def test_broadcast_with_cache(redis_queue: RedisQueue) -> None:
    # Test broadcasting with cache
    test_payload = {"key": "value", "key2": "value2"}
    cache_id = f"test-cache-id-{random.randint(1000, 9999)}"

    received_messages: list[BroadcastMessage] = []

    async def subscriber() -> None:
        async for message in redis_queue.subscribe_queue():
            received_messages.append(message)
            if len(received_messages) >= 1:
                break

    subscriber_task = asyncio.create_task(subscriber())
    await asyncio.sleep(0.1)  # Allow subscriber to start

    # Broadcast message with cache
    await redis_queue.broadcast_with_cache(cache_id, test_payload)

    # Wait for message to be received
    await asyncio.wait_for(subscriber_task, timeout=5)

    assert len(received_messages) == 1
    assert received_messages[0].payload == test_payload

    # Fetch cached message
    cached_message = await redis_queue.fetch_cached_broadcast_message(cache_id)
    assert cached_message is not None
    assert cached_message == test_payload


async def test_done(redis_queue: RedisQueue) -> None:
    # Test message acknowledgment
    test_payload = {b"key": b"value"}

    # Send message
    await redis_queue.send(test_payload)

    # Consume and acknowledge message
    async for message in redis_queue.consume_queue():
        await redis_queue.done(message.msg_id)
        return


@dataclass
class _StreamTestEvent(AbstractAnycastEvent):
    value: int

    @override
    def serialize(self) -> tuple[Any, ...]:
        return (self.value,)

    @classmethod
    @override
    def deserialize(cls, value: tuple[Any, ...]) -> _StreamTestEvent:
        return cls(value[0])

    @classmethod
    @override
    def event_domain(cls) -> EventDomain:
        return EventDomain.AGENT

    @override
    def domain_id(self) -> str | None:
        return None

    @override
    def user_event(self) -> UserEvent | None:
        return None

    @classmethod
    @override
    def event_name(cls) -> str:
        return "test_stream_event"


@pytest.fixture
def redis_target(redis_container: tuple[str, HostPortPair]) -> RedisTarget:
    return RedisTarget(addr=redis_container[1])


@pytest.fixture
async def stream_conn(
    redis_container: tuple[str, HostPortPair],
) -> AsyncGenerator[RedisConnectionInfo, None]:
    conn = redis_helper.get_redis_object(
        RedisTarget(
            addr=redis_container[1],
            redis_helper_config=RedisHelperConfig(
                socket_timeout=5.0,
                socket_connect_timeout=2.0,
                reconnect_poll_timeout=0.3,
                max_connections=10,
                connection_ready_timeout=1.0,
            ),
        ),
        name="test-stream-db",
        db=REDIS_STREAM_DB,
    )
    yield conn
    await conn.close()


@pytest.fixture
async def queues() -> AsyncGenerator[list[RedisQueue], None]:
    created: list[RedisQueue] = []
    yield created
    for queue in created:
        await queue.close()


def _producer_args(stream_key: str, channel: str) -> RedisMQArgs:
    return RedisMQArgs(
        anycast_stream_key=stream_key,
        broadcast_channel=channel,
        consume_stream_keys=None,
        subscribe_channels=None,
        group_name="test-group",
        node_id="test-producer",
        db=REDIS_STREAM_DB,
    )


async def _wait_for(predicate: Callable[[], Awaitable[bool]], timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not await predicate():
            await asyncio.sleep(0.05)


async def test_dispatcher_acks_messages_on_every_consumed_stream(
    redis_target: RedisTarget,
    stream_conn: RedisConnectionInfo,
    queues: list[RedisQueue],
) -> None:
    suffix = secrets.token_hex(4)
    own_stream = f"test-events-{suffix}"
    producer_stream = f"test-events-{suffix}:agent:a"
    group_name = f"test-group-{suffix}"
    consumer_queue = await RedisQueue.create(
        redis_target,
        RedisMQArgs(
            anycast_stream_key=own_stream,
            broadcast_channel=f"test-events_all-{suffix}",
            consume_stream_keys={own_stream, producer_stream},
            subscribe_channels=None,
            group_name=group_name,
            node_id="test-consumer",
            db=REDIS_STREAM_DB,
        ),
    )
    queues.append(consumer_queue)
    producer_queue = await RedisQueue.create(
        redis_target, _producer_args(producer_stream, f"test-events_all-{suffix}:agent:a")
    )
    queues.append(producer_queue)

    received: list[tuple[str, int]] = []

    async def handler(ctx: object, source: AgentId, event: _StreamTestEvent) -> None:
        received.append((str(source), event.value))

    dispatcher = EventDispatcher(consumer_queue)
    dispatcher.consume(_StreamTestEvent, None, handler)
    await dispatcher.start()
    try:
        own_producer = EventProducer(consumer_queue, source=AgentId("manager"))
        other_producer = EventProducer(producer_queue, source=AgentId("agent-a"))
        for value in range(3):
            await own_producer.anycast_event(_StreamTestEvent(value))
            await other_producer.anycast_event(_StreamTestEvent(value))

        async def all_received() -> bool:
            return len(received) == 6

        await _wait_for(all_received)

        # Each producer wrote only to its own stream.
        assert await stream_conn.client.xlen(own_stream) == 3
        assert await stream_conn.client.xlen(producer_stream) == 3

        async def nothing_pending() -> bool:
            for stream_key in (own_stream, producer_stream):
                pending = await stream_conn.client.xpending(stream_key, group_name)  # type: ignore[no-untyped-call]
                if pending["pending"] != 0:
                    return False
            return True

        await _wait_for(nothing_pending)
    finally:
        await dispatcher.close()

    assert sorted(received) == sorted(
        [("manager", value) for value in range(3)] + [("agent-a", value) for value in range(3)]
    )


async def test_consumed_message_carries_its_stream_key(
    redis_target: RedisTarget,
    stream_conn: RedisConnectionInfo,
    queues: list[RedisQueue],
) -> None:
    suffix = secrets.token_hex(4)
    streams = [f"test-events-{suffix}:{name}" for name in ("a", "b")]
    group_name = f"test-group-{suffix}"
    consumer_queue = await RedisQueue.create(
        redis_target,
        RedisMQArgs(
            anycast_stream_key=streams[0],
            broadcast_channel=f"test-events_all-{suffix}",
            consume_stream_keys=set(streams),
            subscribe_channels=None,
            group_name=group_name,
            node_id="test-consumer",
            db=REDIS_STREAM_DB,
        ),
    )
    queues.append(consumer_queue)
    for stream_key in streams:
        producer = await RedisQueue.create(
            redis_target, _producer_args(stream_key, f"test-events_all-{suffix}")
        )
        queues.append(producer)
        await producer.send({b"stream": stream_key.encode()})

    seen: dict[str, str | None] = {}
    async with asyncio.timeout(10):
        async for message in consumer_queue.consume_queue():
            seen[message.payload[b"stream"].decode()] = message.stream_key
            await consumer_queue.done(message.msg_id, stream_key=message.stream_key)
            if len(seen) == len(streams):
                break

    assert seen == {stream_key: stream_key for stream_key in streams}
    for stream_key in streams:
        pending = await stream_conn.client.xpending(stream_key, group_name)  # type: ignore[no-untyped-call]
        assert pending["pending"] == 0


async def test_subscriber_reports_the_channel(
    redis_target: RedisTarget,
    queues: list[RedisQueue],
) -> None:
    suffix = secrets.token_hex(4)
    channels = [f"test-events_all-{suffix}", f"test-events_all-{suffix}:agent:a"]
    subscriber_queue = await RedisQueue.create(
        redis_target,
        RedisMQArgs(
            anycast_stream_key=f"test-events-{suffix}",
            broadcast_channel=channels[0],
            consume_stream_keys=None,
            subscribe_channels=set(channels),
            group_name="test-group",
            node_id="test-subscriber",
            db=REDIS_STREAM_DB,
        ),
    )
    queues.append(subscriber_queue)
    producer_queue = await RedisQueue.create(
        redis_target, _producer_args(f"test-events-{suffix}:agent:a", channels[1])
    )
    queues.append(producer_queue)

    async def first_message() -> BroadcastMessage:
        async for message in subscriber_queue.subscribe_queue():
            return message
        raise AssertionError("subscriber stopped")

    task = asyncio.create_task(first_message())
    await asyncio.sleep(0.2)  # Allow the subscription to settle
    await producer_queue.broadcast({"name": "x"})
    message = await asyncio.wait_for(task, timeout=5)

    assert message == BroadcastMessage({"name": "x"}, channel=channels[1])
