from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Iterable
from typing import Any, Self, override

import glide

from ai.backend.common.clients.valkey_client.valkey_stream.client import ValkeyStreamClient
from ai.backend.common.json import load_json
from ai.backend.common.message_queue.abc import AbstractSubscriber
from ai.backend.common.message_queue.types import BroadcastMessage
from ai.backend.common.types import RedisTarget
from ai.backend.logging.utils import BraceStyleAdapter

log = BraceStyleAdapter(logging.getLogger(__spec__.name))

_DROPPED_MESSAGE_REPORT_INTERVAL = 60.0  # seconds


class RedisSubscriber(AbstractSubscriber):
    """
    Redis-based subscriber implementation for receiving broadcast messages.

    This component handles subscribing to Redis pub/sub channels and receiving
    broadcast messages. Messages are delivered to all active subscribers.
    """

    _client: ValkeyStreamClient
    _subscribe_queue: asyncio.Queue[BroadcastMessage]
    _channels: set[str]
    _closed: bool
    _loop_task: asyncio.Task[Any] | None
    _dropped_count: int
    _last_drop_report: float

    def __init__(self, client: ValkeyStreamClient, channels: Iterable[str]) -> None:
        """
        Initialize the Redis subscriber.

        Args:
            client: ValkeyStreamClient configured with pub/sub channels
            channels: Set of Redis channels to subscribe to
        """
        self._client = client
        self._subscribe_queue = asyncio.Queue()
        self._channels = set(channels)
        self._closed = False
        self._dropped_count = 0
        self._last_drop_report = -_DROPPED_MESSAGE_REPORT_INTERVAL

        # Start the background task to read broadcast messages
        self._loop_task = asyncio.create_task(self._read_broadcast_messages_loop())

    @classmethod
    async def create(cls, redis_target: RedisTarget, channels: set[str], db: int = 0) -> Self:
        """
        Create a new RedisSubscriber instance.

        Args:
            redis_target: Redis connection configuration
            channels: Set of Redis channels to subscribe to
            db: Redis database number (default: 0)

        Returns:
            Configured RedisSubscriber instance
        """
        client = await ValkeyStreamClient.create(
            redis_target.to_valkey_target(),
            human_readable_name="redis_subscriber",
            db_id=db,
            pubsub_channels=channels,
        )
        return cls(client, channels)

    @override
    async def subscribe_queue(self) -> AsyncGenerator[BroadcastMessage, None]:  # type: ignore[override]
        """
        Subscribe to broadcast messages.

        This method blocks until broadcast messages are available and yields them
        as they arrive. Unlike consumer messages, broadcast messages don't require
        acknowledgment.

        Yields:
            BroadcastMessage: Broadcast messages from subscribed channels

        Raises:
            RuntimeError: If the subscriber is closed
        """
        while not self._closed:
            try:
                yield await self._subscribe_queue.get()
            except asyncio.CancelledError:
                break

    @override
    async def close(self) -> None:
        """
        Close the subscriber and cleanup resources.

        This cancels the background task and closes the Redis client connection.
        """
        if self._closed:
            return

        self._closed = True

        # Cancel the background task
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                log.debug("Subscriber loop task cancelled")

        await self._client.close()
        log.debug("RedisSubscriber closed")

    async def _read_broadcast_messages_loop(self) -> None:
        """
        Background task to read broadcast messages from subscribed channels.
        """
        log.debug("Starting read broadcast messages loop for channels {}", self._channels)

        while not self._closed:
            try:
                await self._read_broadcast_messages()
            except glide.ClosingError:
                log.info("Client connection closed, stopping read broadcast messages loop")
                break
            except Exception as e:
                log.error("Error while reading broadcast messages: {}", e)
                # Add a small delay to avoid tight error loops
                await asyncio.sleep(1.0)

    async def _read_broadcast_messages(self) -> None:
        """
        Read broadcast messages and put them in the subscribe queue.

        A message that is not a JSON object is dropped without delaying the next one.
        """
        channel, raw_message = await self._client.receive_broadcast_message_with_channel()
        try:
            payload = load_json(raw_message)
        except (TypeError, ValueError):
            payload = None
        if not isinstance(payload, dict):
            self._report_dropped_message(channel)
            return
        msg = BroadcastMessage(payload, channel=channel)
        await self._subscribe_queue.put(msg)

    def _report_dropped_message(self, channel: str) -> None:
        """
        Count a dropped message and log at most one warning per interval.
        """
        self._dropped_count += 1
        now = time.monotonic()
        if now - self._last_drop_report < _DROPPED_MESSAGE_REPORT_INTERVAL:
            return
        log.warning(
            "Dropped {} undecodable broadcast message(s); the last one was on channel {!r}",
            self._dropped_count,
            channel,
        )
        self._dropped_count = 0
        self._last_drop_report = now


class NoopSubscriber(AbstractSubscriber):
    """
    A subscriber without channels.

    It opens no connection and yields nothing until it is closed.
    """

    _closed: asyncio.Event

    def __init__(self) -> None:
        self._closed = asyncio.Event()

    @override
    async def subscribe_queue(self) -> AsyncGenerator[BroadcastMessage, None]:  # type: ignore[override]
        await self._closed.wait()
        nothing: list[BroadcastMessage] = []
        for message in nothing:
            yield message

    @override
    async def close(self) -> None:
        self._closed.set()
