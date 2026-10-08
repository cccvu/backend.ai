import logging
from typing import (
    Self,
)

from glide import (
    Batch,
)

from ai.backend.common.clients.valkey_client.client import (
    AbstractValkeyClient,
    create_valkey_client,
)
from ai.backend.common.exception import BackendAIError
from ai.backend.common.log.types import ContainerLogData
from ai.backend.common.metrics.metric import DomainType, LayerType
from ai.backend.common.resilience import (
    BackoffStrategy,
    MetricArgs,
    MetricPolicy,
    Resilience,
    RetryArgs,
    RetryPolicy,
)
from ai.backend.common.types import AgentId, ValkeyTarget
from ai.backend.logging.utils import BraceStyleAdapter

log = BraceStyleAdapter(logging.getLogger(__spec__.name))

# Resilience instance for valkey_container_log layer
valkey_container_log_resilience = Resilience(
    policies=[
        MetricPolicy(MetricArgs(domain=DomainType.VALKEY, layer=LayerType.VALKEY_CONTAINER_LOG)),
        RetryPolicy(
            RetryArgs(
                max_retries=3,
                retry_delay=0.1,
                backoff_strategy=BackoffStrategy.FIXED,
                non_retryable_exceptions=(BackendAIError,),
            )
        ),
    ]
)


class ValkeyContainerLogClient:
    """
    Client for interacting with Valkey for container log operations using GlideClient.

    This client intentionally ignores server-side failures or connection failures in
    its log-related action methods so that crashes or failures of the container log
    subsystem would not impact the other parts of the system.
    """

    _client: AbstractValkeyClient
    _closed: bool

    def __init__(self, client: AbstractValkeyClient) -> None:
        self._client = client
        self._closed = False

    @classmethod
    async def create(
        cls,
        valkey_target: ValkeyTarget,
        *,
        db_id: int,
        human_readable_name: str,
        pubsub_channels: set[str] | None = None,
    ) -> Self:
        """
        Create a ValkeyContainerLogClient instance.

        :param valkey_target: The target Valkey server to connect to.
        :param db_id: The database index to use.
        :param human_readable_name: The name of the client.
        :param pubsub_channels: Set of channels to subscribe to for pub/sub functionality.
        :return: An instance of ValkeyContainerLogClient.
        """
        client = create_valkey_client(
            valkey_target=valkey_target,
            db_id=db_id,
            human_readable_name=human_readable_name,
            pubsub_channels=pubsub_channels,
        )
        await client.connect()
        return cls(client=client)

    @valkey_container_log_resilience.apply()
    async def close(self) -> None:
        """
        Close the ValkeyContainerLogClient connection.
        """
        if self._closed:
            log.debug("ValkeyContainerLogClient is already closed.")
            return
        self._closed = True
        await self._client.disconnect()

    async def ping(self) -> None:
        """Ping the Valkey server to check connection health."""
        await self._client.ping()

    def _create_batch(self, is_atomic: bool = False) -> Batch:
        """
        Create a batch object for batch operations.

        :param is_atomic: Whether the batch should be atomic (transaction-like).
        :return: A Batch object.
        """
        return Batch(is_atomic=is_atomic)

    @valkey_container_log_resilience.apply()
    async def enqueue_container_logs(
        self,
        agent_id: AgentId,
        container_id: str,
        logs: ContainerLogData,
    ) -> None:
        """
        Enqueue logs for a specific container.
        TODO: Replace with a more efficient log storage solution.

        :param agent_id: The ID of the agent that hosts the container.
        :param container_id: The ID of the container.
        :param logs: The logs to enqueue.
        :raises: GlideClientError if the logs cannot be enqueued.
        """
        key = self._container_log_key(agent_id, container_id)
        tx = self._create_batch()
        tx.rpush(
            key,
            [logs.serialize()],
        )
        tx.expire(
            key,
            3600,  # 1 hour expiration
        )
        async with self._client.client() as conn:
            await conn.exec(tx, raise_on_error=True)

    @valkey_container_log_resilience.apply()
    async def container_log_len(
        self,
        agent_id: AgentId,
        container_id: str,
    ) -> int:
        """
        Get the length of logs for a specific container.

        :param agent_id: The ID of the agent that hosts the container.
        :param container_id: The ID of the container.
        :return: The number of logs for the container.
        :raises: GlideClientError if the length cannot be retrieved.
        """
        key = self._container_log_key(agent_id, container_id)
        async with self._client.client() as conn:
            return await conn.llen(key)

    async def pop_container_logs(
        self,
        agent_id: AgentId,
        container_id: str,
        count: int = 1,
        *,
        max_element_size: int | None = None,
    ) -> list[ContainerLogData] | None:
        """
        Pop logs for a specific container.

        Popped elements are parsed after the pop, outside the retried call, so that a
        malformed element is reported once instead of popping further elements on retry.

        :param agent_id: The ID of the agent that hosts the container.
        :param container_id: The ID of the container.
        :param max_element_size: If given, reject any element longer than this before parsing it.
        :return: List of logs for the container.
        :raises: GlideClientError if the logs cannot be popped.
        :raises: ContainerLogError if a popped element is too large or malformed.
        """
        raw_logs = await self._pop_raw_container_logs(agent_id, container_id, count)
        if raw_logs is None:
            return None
        return [
            ContainerLogData.deserialize(raw_log, max_size=max_element_size) for raw_log in raw_logs
        ]

    @valkey_container_log_resilience.apply()
    async def _pop_raw_container_logs(
        self,
        agent_id: AgentId,
        container_id: str,
        count: int,
    ) -> list[bytes] | None:
        key = self._container_log_key(agent_id, container_id)
        async with self._client.client() as conn:
            return await conn.lpop_count(key, count)

    @valkey_container_log_resilience.apply()
    async def clear_container_logs(
        self,
        agent_id: AgentId,
        container_id: str,
    ) -> None:
        """
        Clear logs for a specific container.

        :param agent_id: The ID of the agent that hosts the container.
        :param container_id: The ID of the container.
        :raises: GlideClientError if the logs cannot be cleared.
        """
        key = self._container_log_key(agent_id, container_id)
        async with self._client.client() as conn:
            await conn.delete([key])

    def _container_log_key(self, agent_id: AgentId, container_id: str) -> str:
        # The agent ID is part of the key so that each agent writes only its own keys.
        return f"containerlog.{agent_id}.{container_id}"
