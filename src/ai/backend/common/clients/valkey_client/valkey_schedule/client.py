from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Self, cast
from uuid import UUID

from glide import Batch, ExpirySet, ExpiryType

from ai.backend.common.clients.valkey_client.client import (
    AbstractValkeyClient,
    create_valkey_client,
)
from ai.backend.common.clients.valkey_client.valkey_schedule.types import (
    ReplicaHealthResult,
    ReplicaHealthStatus,
    ReplicaProbeTarget,
)
from ai.backend.common.exception import BackendAIError
from ai.backend.common.identifier.replica import ReplicaID
from ai.backend.common.json import dump_json_str, load_json
from ai.backend.common.metrics.metric import DomainType, LayerType
from ai.backend.common.resilience import (
    BackoffStrategy,
    MetricArgs,
    MetricPolicy,
    Resilience,
    RetryArgs,
    RetryPolicy,
)
from ai.backend.common.types import AgentId, KernelId, SessionId, ValkeyTarget

PENDING_QUEUE_EXPIRY_SEC = 600  # 10 minutes
SESSION_FAILED_AGENTS_TTL_SEC = 3600  # 1 hour
ROUTE_HEALTH_TTL_SEC = 600  # 10 minutes
MAX_HEALTH_STALENESS_SEC = 300  # 5 minutes - threshold for health status staleness
KERNEL_HEALTH_TTL_SEC = 300  # 5 minutes - TTL for kernel health status
MAX_KERNEL_HEALTH_STALENESS_SEC = 120  # 2 minutes - threshold for kernel health staleness
AGENT_LAST_CHECK_TTL_SEC = 1200  # 20 minutes - TTL for agent last check timestamp
ORPHAN_KERNEL_THRESHOLD_SEC = 600  # 10 minutes - threshold for orphan kernel detection
# TTL for the manager's per-kernel check timestamp. It must stay well above
# ORPHAN_KERNEL_THRESHOLD_SEC plus the orphan observer interval (300 s), so that a kernel
# the manager stopped checking still has a key old enough to be detected as orphaned.
KERNEL_LAST_CHECK_TTL_SEC = 3600  # 1 hour
_KERNEL_PRESENCE_FIELDS: Final[tuple[str, ...]] = ("presence", "last_presence", "created_at")
FORCE_TERMINATED_CLEANUP_TTL_SEC = 1200  # 20 minutes - TTL for force-terminated cleanup queue
ROUTE_PROBE_TTL_SEC = 3600  # 1 hour - TTL for route probe targets
ROUTE_HEALTH_STATUS_TTL_SEC = 120  # 2 minutes - TTL for route health status (expiry = DEGRADED)


class HealthCheckStatus(enum.StrEnum):
    """Status of an individual health check (readiness or liveness)."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STALE = "stale"


# Resilience instance for valkey_schedule layer
valkey_schedule_resilience = Resilience(
    policies=[
        MetricPolicy(MetricArgs(domain=DomainType.VALKEY, layer=LayerType.VALKEY_SCHEDULE)),
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


@dataclass
class HealthStatus:
    """Health status data for a route."""

    readiness: HealthCheckStatus | None  # None if never checked
    liveness: HealthCheckStatus | None  # None if never checked
    last_check: int | None  # Unix timestamp of last check by manager, None if never checked
    created_at: int  # Unix timestamp when route was initialized

    def get_status(self) -> HealthCheckStatus | None:
        """
        Get the overall health status of the route.

        Returns:
            HealthCheckStatus based on readiness and liveness:
            - None: if readiness is not set
            - STALE: if either readiness or liveness is stale
            - HEALTHY: if readiness is healthy (TODO: and liveness when implemented)
            - UNHEALTHY: if readiness is unhealthy
        """
        # TODO: Use liveness too after applying liveness checks in agent
        return self.readiness


@dataclass
class KernelStatus:
    """Presence status for a kernel."""

    presence: HealthCheckStatus | None  # HEALTHY, UNHEALTHY, STALE, or None if never reported
    last_presence: int | None  # Unix timestamp when Agent last reported presence, None if never
    last_check: int | None  # Unix timestamp when Manager last checked, None if never
    created_at: int  # Unix timestamp when first created  # Unix timestamp when first created, None if not initialized


class ValkeyScheduleClient:
    """
    Client for managing scheduling marks in Valkey.
    Provides simple flag-based coordination between scheduling loops.
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
    ) -> Self:
        """
        Create a ValkeyScheduleClient instance.

        :param valkey_target: The target Valkey server to connect to.
        :param db_id: The database index to use.
        :param human_readable_name: The name of the client.
        :return: An instance of ValkeyScheduleClient.
        """
        client = create_valkey_client(
            valkey_target=valkey_target,
            db_id=db_id,
            human_readable_name=human_readable_name,
        )
        await client.connect()
        return cls(client=client)

    def _get_schedule_key(self, schedule_type: str) -> str:
        """
        Generate the Redis key for the given schedule type.

        :param schedule_type: The type of scheduling
        :return: The formatted key string
        """
        return f"schedule:{schedule_type}"

    def _get_deployment_key(self, lifecycle_type: str, sub_step: str | None = None) -> str:
        """
        Generate the Redis key for the given deployment lifecycle type.

        :param lifecycle_type: The type of deployment lifecycle
        :param sub_step: Optional sub-step for finer-grained marks
        :return: The formatted key string
        """
        if sub_step is not None:
            return f"deployment:{lifecycle_type}:{sub_step}"
        return f"deployment:{lifecycle_type}"

    def _get_route_key(self, lifecycle_type: str) -> str:
        """
        Generate the Redis key for the given route lifecycle type.

        :param lifecycle_type: The type of route lifecycle
        :return: The formatted key string
        """
        return f"route:{lifecycle_type}"

    def _get_route_health_key(self, route_id: str) -> str:
        """
        Generate the Redis key for route health status.

        :param route_id: The route ID
        :return: The formatted key string
        """
        return f"route:health:{route_id}"

    def _get_kernel_presence_key(self, agent_id: AgentId, kernel_id: KernelId) -> str:
        """
        Generate the Redis key for kernel presence status.
        The key is scoped by the ID of the agent that reports the presence.

        :param agent_id: The ID of the agent hosting the kernel
        :param kernel_id: The kernel ID
        :return: The formatted key string
        """
        return f"kernel:presence:{agent_id}:{kernel_id}"

    def _get_kernel_last_check_key(self, agent_id: AgentId, kernel_id: KernelId) -> str:
        """
        Generate the Redis key for the manager's last check timestamp of a kernel.
        Only the manager writes it; the agent hosting the kernel reads it.

        :param agent_id: The ID of the agent hosting the kernel
        :param kernel_id: The kernel ID
        :return: The formatted key string
        """
        return f"kernel:last_check:{agent_id}:{kernel_id}"

    def _get_route_probe_key(self, replica_id: ReplicaID) -> str:
        return f"route_probe:{replica_id}"

    def _get_route_health_status_key(self, replica_id: ReplicaID) -> str:
        return f"route_health:{replica_id}"

    def _get_agent_last_check_key(self, agent_id: AgentId) -> str:
        """
        Generate the Redis key for agent last check timestamp.

        :param agent_id: The agent ID
        :return: The formatted key string
        """
        return f"agent:last_check:{agent_id}"

    def _get_session_failed_agents_key(self, session_id: SessionId) -> str:
        """
        Generate the Redis key for session-scoped failed agent tracking.

        :param session_id: The session ID
        :return: The formatted key string
        """
        return f"session:failed_agents:{session_id}"

    async def _get_redis_time(self) -> int:
        """
        Get current Unix timestamp from Redis server using TIME command.
        This ensures consistent timestamps across distributed systems.

        :return: Current Unix timestamp in seconds
        """
        async with self._client.client() as conn:
            result = await conn.time()
        seconds_bytes, _ = result
        return int(seconds_bytes)

    @valkey_schedule_resilience.apply()
    async def get_redis_time(self) -> int:
        """
        Get current Unix timestamp from Redis server using TIME command.
        This ensures consistent timestamps across distributed systems.

        :return: Current Unix timestamp in seconds
        """
        return await self._get_redis_time()

    async def _validate_health_status(
        self,
        status: str | None,
        timestamp_str: str | None,
        current_time: int | None = None,
        staleness_sec: int = MAX_HEALTH_STALENESS_SEC,
    ) -> HealthCheckStatus | None:
        """
        Validate health status by checking if it's healthy and timestamp is not stale.

        :param status: The status string ("1" for healthy, "0" for unhealthy), or None if missing
        :param timestamp_str: The timestamp string value from Redis, or None if missing
        :param current_time: Optional pre-fetched Redis time (fetches if None)
        :param staleness_sec: Staleness threshold in seconds
        :return: HealthCheckStatus indicating the status:
                 - None: if status or timestamp is missing
                 - HEALTHY: status is "1" and timestamp is fresh
                 - UNHEALTHY: status is "0" and timestamp is fresh
                 - STALE: timestamp is stale or invalid
        """
        # Return None if status or timestamp is missing
        if status is None or timestamp_str is None:
            return None
        try:
            timestamp = int(timestamp_str)
            if current_time is None:
                current_time = await self._get_redis_time()
            is_stale = (current_time - timestamp) > staleness_sec
            if is_stale:
                return HealthCheckStatus.STALE
            return HealthCheckStatus.HEALTHY if status == "1" else HealthCheckStatus.UNHEALTHY
        except (ValueError, TypeError):
            # If timestamp is invalid, return None (not enough info)
            return None

    @valkey_schedule_resilience.apply()
    async def mark_schedules_needed_batch(self, schedule_types: Sequence[str]) -> None:
        """
        Batch mark that scheduling is needed for multiple schedule types.

        :param schedule_types: The types of scheduling to mark
        """
        if not schedule_types:
            return
        key_values: Mapping[str | bytes, str | bytes] = {
            self._get_schedule_key(schedule_type): "1" for schedule_type in schedule_types
        }
        async with self._client.client() as conn:
            await conn.mset(key_values)

    @valkey_schedule_resilience.apply()
    async def load_and_delete_schedule_mark(self, schedule_type: str) -> bool:
        """
        Check if a scheduling mark exists and atomically delete it.
        This ensures that only one scheduler processes the mark.

        :param schedule_type: The type of scheduling to check
        :return: True if a mark existed (and was deleted), False otherwise
        """
        key = self._get_schedule_key(schedule_type)
        # Use Batch for atomic GET and DELETE
        batch = Batch(is_atomic=True)
        batch.get(key)
        batch.delete([key])
        async with self._client.client() as conn:
            results = await conn.exec(batch, raise_on_error=True)

        # Check if results exist and the first element (GET result) is not None
        if results and len(results) > 0:
            return results[0] is not None
        return False

    def _pending_queue_key(self, resource_group_id: str) -> str:
        return f"pending_queue:{resource_group_id}"

    def _queue_position_key(self, session_id: SessionId) -> str:
        return f"queue_position:{session_id}"

    @valkey_schedule_resilience.apply()
    async def set_pending_queue(
        self, resource_group_id: str, session_ids: Sequence[SessionId]
    ) -> None:
        """
        Set up the pending queue for a specific resource group and store the position of sessions in the pending queue.
        """
        batch = Batch(is_atomic=False)
        key = self._pending_queue_key(resource_group_id)
        value = dump_json_str([str(sid) for sid in session_ids])
        batch.set(key, value, expiry=ExpirySet(ExpiryType.SEC, PENDING_QUEUE_EXPIRY_SEC))

        for position, session_id in enumerate(session_ids):
            pos_key = self._queue_position_key(session_id)
            batch.set(
                pos_key, str(position), expiry=ExpirySet(ExpiryType.SEC, PENDING_QUEUE_EXPIRY_SEC)
            )
        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    @valkey_schedule_resilience.apply()
    async def get_pending_queue(self, resource_group_id: str) -> list[SessionId]:
        """
        Get the pending queue for a specific resource group.
        """
        key = self._pending_queue_key(resource_group_id)
        async with self._client.client() as conn:
            result = await conn.get(key)
        if result is None:
            return []
        raw_session_ids = load_json(result)
        return [SessionId(UUID(sid)) for sid in raw_session_ids]

    @valkey_schedule_resilience.apply()
    async def get_queue_positions(self, session_ids: Sequence[SessionId]) -> list[int | None]:
        """
        Get the positions of multiple sessions in their pending queue.
        """
        if not session_ids:
            return []
        batch = Batch(is_atomic=False)
        for session_id in session_ids:
            key = self._queue_position_key(session_id)
            batch.get(key)
        async with self._client.client() as conn:
            batch_result = await conn.exec(batch, raise_on_error=True)
        if batch_result is None:
            return [None for _ in session_ids]

        result: list[int | None] = []
        for pos in batch_result:
            if pos is None:
                result.append(None)
            else:
                try:
                    result.append(int(pos))
                except ValueError:
                    result.append(None)
        return result

    @valkey_schedule_resilience.apply()
    async def record_session_failed_agents(
        self,
        session_id: SessionId,
        agent_ids: Sequence[AgentId],
        ttl_sec: int = SESSION_FAILED_AGENTS_TTL_SEC,
    ) -> None:
        """
        Record agents that failed for a specific session.
        Uses SADD to append to the set, so repeated calls accumulate failed agents.

        :param session_id: The session that experienced the failure
        :param agent_ids: Agent IDs that failed for this session
        :param ttl_sec: TTL in seconds for auto-cleanup (default 1 hour)
        """
        if not agent_ids:
            return
        key = self._get_session_failed_agents_key(session_id)
        members: list[str] = [str(aid) for aid in agent_ids]
        batch = Batch(is_atomic=True)
        batch.sadd(key, members)
        batch.expire(key, ttl_sec)
        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    @valkey_schedule_resilience.apply()
    async def get_multiple_session_failed_agents(
        self,
        session_ids: Sequence[SessionId],
    ) -> list[frozenset[AgentId]]:
        """
        Get the sets of agents that previously failed for multiple sessions in one round-trip.

        Uses a non-atomic Batch to pipeline all SMEMBERS commands, reducing N round-trips to 1.

        :param session_ids: The sessions to look up
        :return: List of frozensets of failed agent IDs, in the same order as session_ids
        """
        if not session_ids:
            return []
        batch = Batch(is_atomic=False)
        for session_id in session_ids:
            batch.smembers(self._get_session_failed_agents_key(session_id))
        async with self._client.client() as conn:
            batch_result = await conn.exec(batch, raise_on_error=True)
        # batch_result is None only for atomic batches (transactions) when a WATCH-guarded key
        # was modified before EXEC — non-atomic pipelines (is_atomic=False) never return None.
        # This guard is kept for defensive typing since exec() is annotated Optional[List[...]].
        if batch_result is None:
            return [frozenset() for _ in session_ids]
        return [
            frozenset(AgentId(member.decode()) for member in members)
            if isinstance(members, set)
            else frozenset()
            for members in batch_result
        ]

    @valkey_schedule_resilience.apply()
    async def mark_deployment_needed(
        self, lifecycle_type: str, sub_step: str | None = None
    ) -> None:
        """
        Mark that a deployment lifecycle operation is needed.
        Simply sets a flag that will be checked in the next scheduling loop.

        :param lifecycle_type: The type of deployment lifecycle to mark
        :param sub_step: Optional sub-step for finer-grained marks
        """
        key = self._get_deployment_key(lifecycle_type, sub_step)
        async with self._client.client() as conn:
            await conn.set(key, b"1")

    @valkey_schedule_resilience.apply()
    async def load_and_delete_deployment_mark(
        self, lifecycle_type: str, sub_step: str | None = None
    ) -> bool:
        """
        Check if a deployment lifecycle mark exists and atomically delete it.
        This ensures that only one scheduler processes the mark.

        :param lifecycle_type: The type of deployment lifecycle to check
        :param sub_step: Optional sub-step for finer-grained marks
        :return: True if a mark existed (and was deleted), False otherwise
        """
        key = self._get_deployment_key(lifecycle_type, sub_step)
        # Use Batch for atomic GET and DELETE
        batch = Batch(is_atomic=True)
        batch.get(key)
        batch.delete([key])
        async with self._client.client() as conn:
            results = await conn.exec(batch, raise_on_error=True)

        # Check if results exist and the first element (GET result) is not None
        if results and len(results) > 0:
            return results[0] is not None
        return False

    @valkey_schedule_resilience.apply()
    async def mark_route_needed(self, lifecycle_type: str) -> None:
        """
        Mark that a route lifecycle operation is needed.
        Simply sets a flag that will be checked in the next scheduling loop.

        :param lifecycle_type: The type of route lifecycle to mark
        """
        key = self._get_route_key(lifecycle_type)
        async with self._client.client() as conn:
            await conn.set(key, b"1")

    @valkey_schedule_resilience.apply()
    async def load_and_delete_route_mark(self, lifecycle_type: str) -> bool:
        """
        Check if a route lifecycle mark exists and atomically delete it.
        This ensures that only one scheduler processes the mark.

        :param lifecycle_type: The type of route lifecycle to check
        :return: True if a mark existed (and was deleted), False otherwise
        """
        key = self._get_route_key(lifecycle_type)
        # Use Batch for atomic GET and DELETE
        batch = Batch(is_atomic=True)
        batch.get(key)
        batch.delete([key])
        async with self._client.client() as conn:
            results = await conn.exec(batch, raise_on_error=True)

        # Check if results exist and the first element (GET result) is not None
        if results and len(results) > 0:
            return results[0] is not None
        return False

    @valkey_schedule_resilience.apply()
    async def get_route_health_status(self, route_id: str) -> HealthStatus | None:
        """
        Get health status for a route from Redis.

        :param route_id: The route ID to check
        :return: HealthStatus object or None if not found
        """
        key = self._get_route_health_key(route_id)
        async with self._client.client() as conn:
            result = await conn.hgetall(key)
        if not result:
            return None

        # Convert bytes to strings and parse
        data = {k.decode(): v.decode() for k, v in result.items()}

        # Parse boolean values using validation helper (checks both status and staleness)
        # Pass None for missing fields instead of defaults
        readiness = await self._validate_health_status(
            data.get("readiness"), data.get("last_readiness")
        )
        liveness = await self._validate_health_status(
            data.get("liveness"), data.get("last_liveness")
        )
        last_check = int(data["last_check"]) if "last_check" in data else None
        created_at = int(data.get("created_at", "0"))

        return HealthStatus(
            readiness=readiness,
            liveness=liveness,
            last_check=last_check,
            created_at=created_at,
        )

    @valkey_schedule_resilience.apply()
    async def initialize_routes_health_status_batch(self, route_ids: list[str]) -> None:
        """
        Batch initialize health status for multiple routes in Redis with TTL.
        This should only be called during initial route creation.
        Always initializes with readiness=False and liveness=False.

        :param route_ids: List of route IDs to initialize
        """
        if not route_ids:
            return

        current_time = str(await self._get_redis_time())
        batch = Batch(is_atomic=False)

        for route_id in route_ids:
            key = self._get_route_health_key(route_id)
            data: Mapping[str | bytes, str | bytes] = {
                "readiness": "0",
                "liveness": "0",
                "last_check": current_time,
                "created_at": current_time,
                # last_readiness and last_liveness are not set until first health check
            }
            batch.hset(key, data)
            batch.expire(key, ROUTE_HEALTH_TTL_SEC)

        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    @valkey_schedule_resilience.apply()
    async def update_route_readiness(self, route_id: str, readiness: bool) -> None:
        """
        Update readiness status for a route in Redis.
        This should be called by app proxy after health check.

        :param route_id: The route ID to update
        :param readiness: Whether the route is ready
        """
        key = self._get_route_health_key(route_id)
        data: Mapping[str | bytes, str | bytes] = {
            "readiness": "1" if readiness else "0",
            "last_readiness": str(await self._get_redis_time()),
        }

        batch = Batch(is_atomic=False)
        batch.hset(key, data)
        batch.expire(key, ROUTE_HEALTH_TTL_SEC)
        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    @valkey_schedule_resilience.apply()
    async def update_route_liveness(self, route_id: str, liveness: bool) -> None:
        """
        Update liveness status for a route in Redis.
        This should be called by agent after liveness check.

        :param route_id: The route ID to update
        :param liveness: Whether the route is alive
        """
        key = self._get_route_health_key(route_id)
        data: Mapping[str | bytes, str | bytes] = {
            "liveness": "1" if liveness else "0",
            "last_liveness": str(await self._get_redis_time()),
        }

        batch = Batch(is_atomic=False)
        batch.hset(key, data)
        batch.expire(key, ROUTE_HEALTH_TTL_SEC)
        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    @valkey_schedule_resilience.apply()
    async def check_route_health_status(
        self, route_ids: list[str]
    ) -> Mapping[str, HealthStatus | None]:
        """
        Check health status for multiple routes and update last_check timestamp.
        This is used by the manager to track when it last checked each route.

        :param route_ids: List of route IDs to check
        :return: Mapping of route ID to HealthStatus object or None if not found
        """
        if not route_ids:
            return {}

        current_time = str(await self._get_redis_time())
        batch = Batch(is_atomic=False)

        # Single batch: update last_check, refresh TTL, and get all data
        for route_id in route_ids:
            key = self._get_route_health_key(route_id)
            batch.hset(key, {"last_check": current_time})
            batch.expire(key, ROUTE_HEALTH_TTL_SEC)
            batch.hgetall(key)

        async with self._client.client() as conn:
            results = await conn.exec(batch, raise_on_error=False)
        if results is None:
            return dict.fromkeys(route_ids)

        # Process results - every 3rd result is the hgetall response
        health_statuses: dict[str, HealthStatus | None] = {}
        for i, route_id in enumerate(route_ids):
            # Results are in groups of 3: hset result, expire result, hgetall result
            hgetall_result = results[i * 3 + 2] if len(results) > i * 3 + 2 else None

            if not hgetall_result:
                health_statuses[route_id] = None
                continue

            result = cast(dict[bytes, bytes], hgetall_result)
            if not result:
                health_statuses[route_id] = None
                continue

            # Parse existing data
            data = {k.decode(): v.decode() for k, v in result.items()}
            # Validate health check statuses - pass None for missing fields
            readiness_status = await self._validate_health_status(
                data.get("readiness"), data.get("last_readiness")
            )
            liveness_status = await self._validate_health_status(
                data.get("liveness"), data.get("last_liveness")
            )
            health_statuses[route_id] = HealthStatus(
                readiness=readiness_status,
                liveness=liveness_status,
                last_check=int(data["last_check"]) if "last_check" in data else None,
                created_at=int(data.get("created_at", "0")),
            )

        return health_statuses

    @valkey_schedule_resilience.apply()
    async def update_routes_readiness_batch(self, route_readiness: Mapping[str, bool]) -> None:
        """
        Batch update readiness status for multiple routes in Redis.
        This should be called by app proxy after health checks.

        :param route_readiness: Mapping of route ID to readiness status
        """
        if not route_readiness:
            return

        current_time = str(await self._get_redis_time())
        batch = Batch(is_atomic=False)
        for route_id, readiness in route_readiness.items():
            key = self._get_route_health_key(route_id)
            data: Mapping[str | bytes, str | bytes] = {
                "readiness": "1" if readiness else "0",
                "last_readiness": current_time,
            }
            batch.hset(key, data)
            batch.expire(key, ROUTE_HEALTH_TTL_SEC)

        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    # ==================== ReplicaProbeTarget / ReplicaHealthStatus Methods ====================

    @valkey_schedule_resilience.apply()
    async def register_route_probe_targets_batch(
        self, targets: Sequence[ReplicaProbeTarget]
    ) -> None:
        """
        Batch register ReplicaProbeTarget entries in Valkey.
        Called by coordinator when route enters WARMING_UP and replica host/port are known.

        :param targets: ReplicaProbeTarget instances to store
        """
        if not targets:
            return

        batch = Batch(is_atomic=False)
        for target in targets:
            key = self._get_route_probe_key(target.replica_id)
            batch.hset(key, target.to_valkey_hash())
            batch.expire(key, ROUTE_PROBE_TTL_SEC)

        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    @valkey_schedule_resilience.apply()
    async def get_route_probe_targets_batch(
        self, replica_ids: Sequence[ReplicaID]
    ) -> Mapping[ReplicaID, ReplicaProbeTarget | None]:
        """
        Batch get ReplicaProbeTargets from Valkey.

        :param replica_ids: Replica IDs to look up
        :return: Mapping of replica_id to ReplicaProbeTarget (None if missing or expired)
        """
        if not replica_ids:
            return {}

        batch = Batch(is_atomic=False)
        for replica_id in replica_ids:
            batch.hgetall(self._get_route_probe_key(replica_id))

        async with self._client.client() as conn:
            results = await conn.exec(batch, raise_on_error=False)
        if results is None:
            return dict.fromkeys(replica_ids)

        targets: dict[ReplicaID, ReplicaProbeTarget | None] = {}
        for i, replica_id in enumerate(replica_ids):
            hgetall_result = results[i] if len(results) > i else None
            if not hgetall_result:
                targets[replica_id] = None
                continue
            raw = cast(dict[bytes, bytes], hgetall_result)
            if not raw or b"replica_id" not in raw:
                targets[replica_id] = None
                continue
            data = {k.decode(): v.decode() for k, v in raw.items()}
            targets[replica_id] = ReplicaProbeTarget.from_valkey_hash(data)

        return targets

    @valkey_schedule_resilience.apply()
    async def record_route_health_statuses_batch(
        self, results: Sequence[ReplicaHealthResult]
    ) -> None:
        """
        Batch record health check results for multiple routes.
        Fetches Redis time once and writes all statuses in a single pipeline.
        Refreshes TTL on every call; key expiry signals DEGRADED.

        :param results: Sequence of ReplicaHealthResult instances
        """
        if not results:
            return

        current_time = str(await self._get_redis_time())
        batch = Batch(is_atomic=False)
        for result in results:
            key = self._get_route_health_status_key(result.replica_id)
            data: Mapping[str | bytes, str | bytes] = {
                "replica_id": str(result.replica_id),
                "healthy": "1" if result.healthy else "0",
                "last_check": current_time,
                "consecutive_failures": str(result.consecutive_failures),
            }
            batch.hset(key, data)
            ttl_sec = result.ttl_sec if result.ttl_sec is not None else ROUTE_HEALTH_STATUS_TTL_SEC
            batch.expire(key, ttl_sec)

        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    @valkey_schedule_resilience.apply()
    async def get_route_health_statuses_batch(
        self, replica_ids: Sequence[ReplicaID]
    ) -> Mapping[ReplicaID, ReplicaHealthStatus | None]:
        """
        Batch get ReplicaHealthStatus from Valkey.
        None means no recent health check (key missing or TTL expired) → DEGRADED.

        :param replica_ids: Replica IDs to look up
        :return: Mapping of replica_id to ReplicaHealthStatus (None if missing or expired)
        """
        if not replica_ids:
            return {}

        batch = Batch(is_atomic=False)
        for replica_id in replica_ids:
            batch.hgetall(self._get_route_health_status_key(replica_id))

        async with self._client.client() as conn:
            results = await conn.exec(batch, raise_on_error=False)
        if results is None:
            return dict.fromkeys(replica_ids)

        statuses: dict[ReplicaID, ReplicaHealthStatus | None] = {}
        for i, replica_id in enumerate(replica_ids):
            hgetall_result = results[i] if len(results) > i else None
            if not hgetall_result:
                statuses[replica_id] = None
                continue
            raw = cast(dict[bytes, bytes], hgetall_result)
            if not raw or b"replica_id" not in raw:
                statuses[replica_id] = None
                continue
            data = {k.decode(): v.decode() for k, v in raw.items()}
            statuses[replica_id] = ReplicaHealthStatus.from_valkey_hash(data)

        return statuses

    @valkey_schedule_resilience.apply()
    async def close(self) -> None:
        """
        Close the ValkeyScheduleClient connection.
        """
        if self._closed:
            return
        self._closed = True
        await self._client.disconnect()

    async def ping(self) -> None:
        """Ping the Valkey server to check connection health."""
        await self._client.ping()

    # ==================== Kernel Presence Methods ====================
    #
    # Every key is scoped by the ID of the agent hosting the kernel, and readers take that
    # ID from their own records, so each agent only ever writes keys under its own ID:
    # - kernel:presence:<agent>:<kernel>: hash written by the agent hosting the kernel
    # - kernel:last_check:<agent>:<kernel>: written only by the manager
    # - agent:last_check:<agent>: written only by the manager

    @staticmethod
    def _parse_str(value: Any) -> str | None:
        """
        Decode a value read from Valkey, returning None for missing or malformed values
        (including errors returned in place of a result by a non-raising batch).
        """
        if isinstance(value, bytes):
            try:
                return value.decode()
            except UnicodeDecodeError:
                return None
        if isinstance(value, str):
            return value
        return None

    @classmethod
    def _parse_timestamp(cls, value: Any) -> int | None:
        """
        Parse a Unix timestamp read from Valkey, returning None for missing or malformed values.
        """
        text = cls._parse_str(value)
        if text is None:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @classmethod
    def _parse_kernel_presence(
        cls,
        fields: Any,
        *,
        last_check: int | None,
        current_time: int,
    ) -> KernelStatus | None:
        """
        Build a kernel status from the HMGET result of the presence fields.

        Returns None when the presence was never reported or a value is malformed,
        which callers treat as "unknown" (the manager confirms with the agent).
        """
        if not isinstance(fields, list) or len(fields) != len(_KERNEL_PRESENCE_FIELDS):
            return None
        presence_raw, last_presence_raw, created_at_raw = fields
        presence = cls._parse_str(presence_raw)
        last_presence = cls._parse_timestamp(last_presence_raw)
        if presence is None or last_presence is None:
            return None
        if (current_time - last_presence) > MAX_KERNEL_HEALTH_STALENESS_SEC:
            health = HealthCheckStatus.STALE
        elif presence == "1":
            health = HealthCheckStatus.HEALTHY
        else:
            health = HealthCheckStatus.UNHEALTHY
        return KernelStatus(
            presence=health,
            last_presence=last_presence,
            last_check=last_check,
            created_at=cls._parse_timestamp(created_at_raw) or 0,
        )

    @valkey_schedule_resilience.apply()
    async def initialize_kernel_presence_batch(
        self,
        kernel_agents: Mapping[KernelId, AgentId],
    ) -> None:
        """
        Batch initialize presence status for multiple kernels in Redis.

        :param kernel_agents: Mapping of kernel ID to the ID of the agent hosting it
        """
        if not kernel_agents:
            return

        current_time = await self._get_redis_time()
        current_time_str = str(current_time)
        batch = Batch(is_atomic=False)
        for kernel_id, agent_id in kernel_agents.items():
            key = self._get_kernel_presence_key(agent_id, kernel_id)
            data: Mapping[str | bytes, str | bytes] = {
                "presence": "0",
                "last_presence": current_time_str,
                "created_at": current_time_str,
            }
            batch.hset(key, data)
            batch.expire(key, KERNEL_HEALTH_TTL_SEC)
            batch.set(
                self._get_kernel_last_check_key(agent_id, kernel_id),
                current_time_str,
                expiry=ExpirySet(ExpiryType.SEC, KERNEL_LAST_CHECK_TTL_SEC),
            )

        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    @valkey_schedule_resilience.apply()
    async def update_kernel_presence_batch(
        self,
        agent_id: AgentId,
        kernel_presences: Mapping[KernelId, bool],
    ) -> None:
        """
        Batch update presence status for multiple kernels in Redis.
        This is the preferred method for Agent to report all kernel presences.

        :param agent_id: The ID of the reporting agent, which hosts the kernels
        :param kernel_presences: Mapping of kernel_id to presence status
        """
        if not kernel_presences:
            return

        current_time = await self._get_redis_time()
        current_time_str = str(current_time)
        batch = Batch(is_atomic=False)
        for kernel_id, presence in kernel_presences.items():
            key = self._get_kernel_presence_key(agent_id, kernel_id)
            data: Mapping[str | bytes, str | bytes] = {
                "presence": "1" if presence else "0",
                "last_presence": current_time_str,
            }
            batch.hset(key, data)
            batch.expire(key, KERNEL_HEALTH_TTL_SEC)

        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    @valkey_schedule_resilience.apply()
    async def delete_kernel_presence_batch(
        self,
        kernel_agents: Mapping[KernelId, AgentId],
    ) -> None:
        """
        Batch delete presence status and check timestamps for multiple kernels from Redis.

        :param kernel_agents: Mapping of kernel ID to the ID of the agent hosting it
        """
        if not kernel_agents:
            return

        keys: list[str | bytes] = []
        for kernel_id, agent_id in kernel_agents.items():
            keys.append(self._get_kernel_presence_key(agent_id, kernel_id))
            keys.append(self._get_kernel_last_check_key(agent_id, kernel_id))
        async with self._client.client() as conn:
            await conn.delete(keys)

    @valkey_schedule_resilience.apply()
    async def check_kernel_presence_status_batch(
        self,
        kernel_agents: Mapping[KernelId, AgentId],
        agent_ids: set[AgentId] | None = None,
    ) -> dict[KernelId, KernelStatus | None]:
        """
        Batch check kernel presence status and update the last check timestamps.
        This should be called by Manager during periodic checks.

        The manager records its check time for every kernel under its own key, reads the
        presence reported by the agent hosting each kernel, and optionally records the
        agent check time, all in a single request.

        :param kernel_agents: Mapping of kernel ID to the ID of the agent hosting it,
            taken from the manager's own records
        :param agent_ids: Optional set of agent IDs to update last_check for
        :return: Mapping of kernel_id to status (None if not found or malformed)
        """
        if not kernel_agents:
            return {}

        current_time = await self._get_redis_time()
        current_time_str = str(current_time)
        batch = Batch(is_atomic=False)
        kernel_ids = list(kernel_agents.keys())

        # Update kernel check times first, then agent last_check
        # This ordering prevents timing issues where agent appears alive
        # but kernels haven't been checked yet
        for kernel_id in kernel_ids:
            agent_id = kernel_agents[kernel_id]
            batch.set(
                self._get_kernel_last_check_key(agent_id, kernel_id),
                current_time_str,
                expiry=ExpirySet(ExpiryType.SEC, KERNEL_LAST_CHECK_TTL_SEC),
            )
            batch.hmget(
                self._get_kernel_presence_key(agent_id, kernel_id),
                list(_KERNEL_PRESENCE_FIELDS),
            )

        # Update agent last_check timestamps after kernel updates
        if agent_ids:
            for agent_id in agent_ids:
                agent_key = self._get_agent_last_check_key(agent_id)
                batch.set(
                    agent_key,
                    current_time_str,
                    expiry=ExpirySet(ExpiryType.SEC, AGENT_LAST_CHECK_TTL_SEC),
                )

        async with self._client.client() as conn:
            results = await conn.exec(batch, raise_on_error=False)
        if results is None:
            return dict.fromkeys(kernel_ids)

        # Kernel results come first (2 ops each: set, hmget), then agent results (1 op each)
        result: dict[KernelId, KernelStatus | None] = {}
        for i, kernel_id in enumerate(kernel_ids):
            idx = i * 2 + 1
            result[kernel_id] = self._parse_kernel_presence(
                results[idx] if len(results) > idx else None,
                last_check=current_time,
                current_time=current_time,
            )
        return result

    # ==================== Agent Last Check Methods ====================

    @valkey_schedule_resilience.apply()
    async def get_agent_last_check(self, agent_id: AgentId) -> int | None:
        """
        Get the last check timestamp for an agent.
        This is used by Agent to determine if Manager has checked it.

        :param agent_id: The agent ID
        :return: Unix timestamp of last check, or None if not found or malformed
        """
        key = self._get_agent_last_check_key(agent_id)
        async with self._client.client() as conn:
            result = await conn.get(key)
        return self._parse_timestamp(result)

    @valkey_schedule_resilience.apply()
    async def get_kernel_last_check_batch(
        self,
        agent_id: AgentId,
        kernel_ids: Sequence[KernelId],
    ) -> dict[KernelId, int | None]:
        """
        Get the manager's last check timestamps of the kernels hosted by an agent.
        This is for Agent to read them without modifying anything.

        :param agent_id: The ID of the agent hosting the kernels
        :param kernel_ids: Sequence of kernel IDs to read
        :return: Mapping of kernel_id to Unix timestamp (None if not found or malformed)
        """
        if not kernel_ids:
            return {}

        keys: list[str | bytes] = [
            self._get_kernel_last_check_key(agent_id, kernel_id) for kernel_id in kernel_ids
        ]
        async with self._client.client() as conn:
            values = await conn.mget(keys)
        return {
            kernel_id: self._parse_timestamp(value)
            for kernel_id, value in zip(kernel_ids, values, strict=True)
        }

    # =========================================================================
    # Force-terminated session cleanup queue
    # =========================================================================

    @staticmethod
    def _get_force_terminated_cleanup_key() -> str:
        return "force_terminated_cleanup"

    @valkey_schedule_resilience.apply()
    async def add_force_terminated_sessions(
        self,
        session_ids: Sequence[SessionId],
        ttl_sec: int = FORCE_TERMINATED_CLEANUP_TTL_SEC,
    ) -> None:
        """
        Add force-terminated session IDs to the cleanup queue.
        Uses SADD to accumulate session IDs for container cleanup.

        :param session_ids: Session IDs that were force-terminated
        :param ttl_sec: TTL in seconds for auto-cleanup (default 20 minutes)
        """
        if not session_ids:
            return
        key = self._get_force_terminated_cleanup_key()
        members: list[str] = [str(sid) for sid in session_ids]
        batch = Batch(is_atomic=True)
        batch.sadd(key, members)
        batch.expire(key, ttl_sec)
        async with self._client.client() as conn:
            await conn.exec(batch, raise_on_error=True)

    @valkey_schedule_resilience.apply()
    async def get_force_terminated_sessions(self) -> list[SessionId]:
        """
        Read all force-terminated session IDs from the cleanup queue (non-destructive).

        :return: List of session IDs that need container cleanup
        """
        key = self._get_force_terminated_cleanup_key()
        async with self._client.client() as conn:
            members = await conn.smembers(key)

        if not members:
            return []

        return [SessionId(UUID(member.decode())) for member in members]

    @valkey_schedule_resilience.apply()
    async def remove_force_terminated_sessions(
        self,
        session_ids: Sequence[SessionId],
    ) -> None:
        """
        Remove specific session IDs from the cleanup queue after successful cleanup.

        :param session_ids: Session IDs to remove
        """
        if not session_ids:
            return
        key = self._get_force_terminated_cleanup_key()
        members: list[str] = [str(sid) for sid in session_ids]
        async with self._client.client() as conn:
            await conn.srem(key, members)
