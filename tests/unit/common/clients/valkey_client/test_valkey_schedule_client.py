"""
Tests for ValkeyScheduleClient health status timestamp tracking.
Tests the client with real Redis operations for route health monitoring.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from time import time
from uuid import UUID, uuid4

import pytest
from glide import ExpirySet, ExpiryType

from ai.backend.common.clients.valkey_client.valkey_schedule.client import (
    AGENT_LAST_CHECK_TTL_SEC,
    KERNEL_HEALTH_TTL_SEC,
    KERNEL_LAST_CHECK_TTL_SEC,
    MAX_HEALTH_STALENESS_SEC,
    MAX_KERNEL_HEALTH_STALENESS_SEC,
    HealthCheckStatus,
    ValkeyScheduleClient,
)
from ai.backend.common.clients.valkey_client.valkey_schedule.types import (
    ReplicaHealthResult,
    ReplicaProbeTarget,
)
from ai.backend.common.defs import REDIS_LIVE_DB
from ai.backend.common.identifier.replica import ReplicaID
from ai.backend.common.typed_validators import HostPortPair as HostPortPairModel
from ai.backend.common.types import AgentId, KernelId, SessionId, ValkeyTarget


class TestValkeyScheduleClient:
    """Test cases for ValkeyScheduleClient health status functionality"""

    @pytest.fixture
    async def valkey_schedule_client(
        self,
        redis_container: tuple[str, HostPortPairModel],
    ) -> AsyncGenerator[ValkeyScheduleClient, None]:
        """Create ValkeyScheduleClient with real Redis container"""
        _, hostport_pair = redis_container

        valkey_target = ValkeyTarget(
            addr=hostport_pair.address,
        )

        client = await ValkeyScheduleClient.create(
            valkey_target=valkey_target,
            db_id=REDIS_LIVE_DB,
            human_readable_name="test-valkey-schedule",
        )

        try:
            yield client
        finally:
            await client.close()

    async def _set_stale_health_data(self, client: ValkeyScheduleClient, route_id: str) -> None:
        """Helper: Manually set stale health data for testing staleness detection"""
        key = client._get_route_health_key(route_id)
        stale_timestamp = str(int(time()) - MAX_HEALTH_STALENESS_SEC - 10)
        async with client._client.client() as conn:
            await conn.hset(
                key,
                {
                    "readiness": "1",
                    "last_readiness": stale_timestamp,
                    "liveness": "1",
                    "last_liveness": stale_timestamp,
                },
            )

    async def test_initialize_routes_health_status_batch(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test that initialized routes have None readiness/liveness until first health check"""
        route_ids = ["route-1", "route-2", "route-3"]

        await valkey_schedule_client.initialize_routes_health_status_batch(route_ids)

        # Verify all initialized routes have None readiness/liveness (no timestamp data yet)
        for route_id in route_ids:
            status = await valkey_schedule_client.get_route_health_status(route_id)
            assert status is not None
            assert status.readiness is None, "Initialized route should have None readiness"
            assert status.liveness is None, "Initialized route should have None liveness"

    async def test_update_route_readiness_healthy(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test that updating readiness to healthy makes route ready"""
        route_id = "test-route-healthy"

        await valkey_schedule_client.update_route_readiness(route_id, True)

        status = await valkey_schedule_client.get_route_health_status(route_id)
        assert status is not None
        assert status.readiness == HealthCheckStatus.HEALTHY, (
            "Route should be ready after healthy update"
        )

    async def test_update_route_readiness_unhealthy(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test that updating readiness to unhealthy makes route not ready"""
        route_id = "test-route-unhealthy"

        await valkey_schedule_client.update_route_readiness(route_id, False)

        status = await valkey_schedule_client.get_route_health_status(route_id)
        assert status is not None
        assert status.readiness == HealthCheckStatus.UNHEALTHY, (
            "Route should not be ready after unhealthy update"
        )

    async def test_update_route_liveness_healthy(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test that updating liveness to healthy makes route alive"""
        route_id = "test-route-live"

        await valkey_schedule_client.update_route_liveness(route_id, True)

        status = await valkey_schedule_client.get_route_health_status(route_id)
        assert status is not None
        assert status.liveness == HealthCheckStatus.HEALTHY, (
            "Route should be alive after healthy update"
        )

    async def test_update_route_liveness_unhealthy(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test that updating liveness to unhealthy makes route not alive"""
        route_id = "test-route-dead"

        await valkey_schedule_client.update_route_liveness(route_id, False)

        status = await valkey_schedule_client.get_route_health_status(route_id)
        assert status is not None
        assert status.liveness == HealthCheckStatus.UNHEALTHY, (
            "Route should not be alive after unhealthy update"
        )

    async def test_update_routes_readiness_batch(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test batch updating readiness for multiple routes with correct statuses"""
        route_readiness = {
            "batch-route-1": True,
            "batch-route-2": False,
            "batch-route-3": True,
        }

        await valkey_schedule_client.update_routes_readiness_batch(route_readiness)

        # Verify all routes have correct readiness status
        for route_id, expected_readiness in route_readiness.items():
            status = await valkey_schedule_client.get_route_health_status(route_id)
            assert status is not None
            expected_status = (
                HealthCheckStatus.HEALTHY if expected_readiness else HealthCheckStatus.UNHEALTHY
            )
            assert status.readiness == expected_status, (
                f"Route {route_id} should have readiness={expected_status}"
            )

    async def test_get_route_health_status_healthy_and_fresh(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test getting health status with healthy and fresh data"""
        route_id = "test-fresh-healthy"

        # Set healthy status with fresh timestamp
        await valkey_schedule_client.update_route_readiness(route_id, True)
        await valkey_schedule_client.update_route_liveness(route_id, True)

        status = await valkey_schedule_client.get_route_health_status(route_id)
        assert status is not None
        assert status.readiness == HealthCheckStatus.HEALTHY
        assert status.liveness == HealthCheckStatus.HEALTHY

    async def test_get_route_health_status_healthy_but_stale(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test that stale health data is considered stale"""
        route_id = "test-stale-healthy"

        # Set stale health data
        await self._set_stale_health_data(valkey_schedule_client, route_id)

        status = await valkey_schedule_client.get_route_health_status(route_id)
        assert status is not None
        # Stale data should be marked as stale
        assert status.readiness == HealthCheckStatus.STALE, (
            "Stale health data should be marked as stale"
        )
        assert status.liveness == HealthCheckStatus.STALE, (
            "Stale health data should be marked as stale"
        )

    async def test_get_route_health_status_unhealthy(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test getting health status with unhealthy status"""
        route_id = "test-unhealthy"

        await valkey_schedule_client.update_route_readiness(route_id, False)
        await valkey_schedule_client.update_route_liveness(route_id, False)

        status = await valkey_schedule_client.get_route_health_status(route_id)
        assert status is not None
        assert status.readiness == HealthCheckStatus.UNHEALTHY
        assert status.liveness == HealthCheckStatus.UNHEALTHY

    async def test_get_route_health_status_missing_fields(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test getting health status when last_readiness/last_liveness are missing"""
        route_id = "test-missing-fields"

        # Initialize route (no last_readiness/last_liveness fields)
        await valkey_schedule_client.initialize_routes_health_status_batch([route_id])

        status = await valkey_schedule_client.get_route_health_status(route_id)
        assert status is not None
        # Should be None without timestamp fields
        assert status.readiness is None
        assert status.liveness is None

    async def test_get_route_health_status_not_found(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test getting health status for non-existent route"""
        status = await valkey_schedule_client.get_route_health_status("nonexistent-route")
        assert status is None

    async def test_check_route_health_status_multiple_routes(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test checking health status for multiple routes with different states"""
        # Setup routes with different states
        healthy_route = "check-healthy"
        unhealthy_route = "check-unhealthy"
        stale_route = "check-stale"

        # Healthy route
        await valkey_schedule_client.update_route_readiness(healthy_route, True)
        await valkey_schedule_client.update_route_liveness(healthy_route, True)

        # Unhealthy route
        await valkey_schedule_client.update_route_readiness(unhealthy_route, False)
        await valkey_schedule_client.update_route_liveness(unhealthy_route, False)

        # Stale route
        await self._set_stale_health_data(valkey_schedule_client, stale_route)

        # Check all routes
        statuses = await valkey_schedule_client.check_route_health_status([
            healthy_route,
            unhealthy_route,
            stale_route,
        ])

        healthy_status = statuses[healthy_route]
        assert healthy_status is not None
        assert healthy_status.readiness == HealthCheckStatus.HEALTHY, (
            "Healthy route should be ready"
        )

        unhealthy_status = statuses[unhealthy_route]
        assert unhealthy_status is not None
        assert unhealthy_status.readiness == HealthCheckStatus.UNHEALTHY, (
            "Unhealthy route should not be ready"
        )

        stale_status = statuses[stale_route]
        assert stale_status is not None
        assert stale_status.readiness == HealthCheckStatus.STALE, (
            "Stale route should be marked as stale"
        )

    async def test_check_route_health_status_updates_last_check(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test that check_route_health_status updates last_check timestamp"""
        route_id = "check-last-check"

        # Set route with old last_check timestamp
        key = valkey_schedule_client._get_route_health_key(route_id)
        old_timestamp = str(int(time()) - 60)  # 60 seconds ago
        async with valkey_schedule_client._client.client() as conn:
            await conn.hset(
                key,
                {"readiness": "1", "last_readiness": old_timestamp, "last_check": old_timestamp},
            )

        # Check health status should update last_check
        await valkey_schedule_client.check_route_health_status([route_id])

        # Verify last_check was updated to current time
        updated_status = await valkey_schedule_client.get_route_health_status(route_id)
        assert updated_status is not None
        assert updated_status.last_check is not None
        assert updated_status.last_check > int(old_timestamp), (
            "last_check should be updated after check"
        )


@dataclass
class KernelPresenceFixture:
    """Container for kernel presence test fixtures."""

    client: ValkeyScheduleClient
    agent_id: AgentId
    kernel_ids: list[KernelId]
    healthy_kernel_id: KernelId
    unhealthy_kernel_id: KernelId
    stale_kernel_id: KernelId
    missing_kernel_id: KernelId


class TestKernelPresenceParsing:
    """Parsing of presence values never raises; malformed values give None."""

    @pytest.mark.parametrize(
        "fields",
        [
            None,
            Exception("WRONGTYPE"),
            [b"1", b"1000"],
            [None, None, None],
            [b"1", None, None],
            [b"1", b"not-a-number", None],
            [b"\xff", b"1000", None],
            [b"1", b"\xff", None],
        ],
        ids=[
            "missing",
            "error",
            "short",
            "absent",
            "no-timestamp",
            "bad-timestamp",
            "bad-presence",
            "bad-timestamp-bytes",
        ],
    )
    def test_malformed_presence_is_none(self, fields: object) -> None:
        assert (
            ValkeyScheduleClient._parse_kernel_presence(fields, last_check=1000, current_time=1000)
            is None
        )

    def test_malformed_created_at_is_ignored(self) -> None:
        status = ValkeyScheduleClient._parse_kernel_presence(
            [b"1", b"1000", b"x"], last_check=1000, current_time=1000
        )
        assert status is not None
        assert status.presence == HealthCheckStatus.HEALTHY
        assert status.created_at == 0

    @pytest.mark.parametrize(
        "value",
        [None, b"", b"abc", b"\xff", b"1.5", Exception("error")],
    )
    def test_malformed_timestamp_is_none(self, value: object) -> None:
        assert ValkeyScheduleClient._parse_timestamp(value) is None

    def test_timestamp(self) -> None:
        assert ValkeyScheduleClient._parse_timestamp(b"1234") == 1234


class TestKernelPresenceStatus:
    """Test cases for kernel presence status functionality"""

    @pytest.fixture
    async def valkey_schedule_client(
        self,
        redis_container: tuple[str, HostPortPairModel],
    ) -> AsyncGenerator[ValkeyScheduleClient, None]:
        """Create ValkeyScheduleClient with real Redis container."""
        _, hostport_pair = redis_container

        valkey_target = ValkeyTarget(
            addr=hostport_pair.address,
        )

        client = await ValkeyScheduleClient.create(
            valkey_target=valkey_target,
            db_id=REDIS_LIVE_DB,
            human_readable_name="test-valkey-schedule-kernel",
        )

        try:
            yield client
        finally:
            await client.close()

    @pytest.fixture
    def agent_id(self) -> AgentId:
        """Generate a unique agent ID."""
        return AgentId(f"agent-{uuid4().hex[:8]}")

    @pytest.fixture
    def other_agent_id(self) -> AgentId:
        """Generate another unique agent ID."""
        return AgentId(f"agent-{uuid4().hex[:8]}")

    @pytest.fixture
    def kernel_ids(self) -> list[KernelId]:
        """Generate multiple kernel IDs."""
        return [KernelId(uuid4()) for _ in range(3)]

    @pytest.fixture
    async def initialized_kernels(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        kernel_ids: list[KernelId],
    ) -> list[KernelId]:
        """Initialize multiple kernels and return their IDs."""
        await valkey_schedule_client.initialize_kernel_presence_batch(
            dict.fromkeys(kernel_ids, agent_id)
        )
        return kernel_ids

    @pytest.fixture
    async def healthy_kernel(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
    ) -> KernelId:
        """Create a kernel with healthy presence status."""
        kernel_id = KernelId(uuid4())
        await valkey_schedule_client.update_kernel_presence_batch(agent_id, {kernel_id: True})
        return kernel_id

    @pytest.fixture
    async def unhealthy_kernel(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
    ) -> KernelId:
        """Create a kernel with unhealthy presence status."""
        kernel_id = KernelId(uuid4())
        await valkey_schedule_client.update_kernel_presence_batch(agent_id, {kernel_id: False})
        return kernel_id

    @pytest.fixture
    async def stale_kernel(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
    ) -> KernelId:
        """Create a kernel with stale presence status."""
        kernel_id = KernelId(uuid4())
        key = valkey_schedule_client._get_kernel_presence_key(agent_id, kernel_id)
        stale_timestamp = str(int(time()) - MAX_KERNEL_HEALTH_STALENESS_SEC - 10)
        async with valkey_schedule_client._client.client() as conn:
            await conn.hset(key, {"presence": "1", "last_presence": stale_timestamp})
            await conn.expire(key, KERNEL_HEALTH_TTL_SEC)
        return kernel_id

    @pytest.fixture
    async def mixed_state_kernels(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        healthy_kernel: KernelId,
        unhealthy_kernel: KernelId,
        stale_kernel: KernelId,
    ) -> KernelPresenceFixture:
        """Create kernels with various states for mixed state testing."""
        missing_kernel_id = KernelId(uuid4())  # Not initialized
        return KernelPresenceFixture(
            client=valkey_schedule_client,
            agent_id=agent_id,
            kernel_ids=[healthy_kernel, unhealthy_kernel, stale_kernel, missing_kernel_id],
            healthy_kernel_id=healthy_kernel,
            unhealthy_kernel_id=unhealthy_kernel,
            stale_kernel_id=stale_kernel,
            missing_kernel_id=missing_kernel_id,
        )

    # ===== Tests =====

    async def test_initialize_kernel_presence_batch(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        initialized_kernels: list[KernelId],
    ) -> None:
        """Test batch initialization of kernel presence status."""
        statuses = await valkey_schedule_client.check_kernel_presence_status_batch(
            dict.fromkeys(initialized_kernels, agent_id)
        )

        assert len(statuses) == len(initialized_kernels)
        for kernel_id in initialized_kernels:
            status = statuses[kernel_id]
            assert status is not None
            assert status.presence == HealthCheckStatus.UNHEALTHY
            assert status.last_presence is not None and status.last_presence > 0
            assert status.last_check is not None and status.last_check > 0
            assert status.created_at > 0

    async def test_initialize_kernel_presence_batch_empty(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test that empty batch initialization does nothing."""
        await valkey_schedule_client.initialize_kernel_presence_batch({})

    async def test_presence_key_is_agent_scoped(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        healthy_kernel: KernelId,
    ) -> None:
        """The agent writes its presence under its own ID only."""
        async with valkey_schedule_client._client.client() as conn:
            scoped = await conn.hgetall(f"kernel:presence:{agent_id}:{healthy_kernel}")
            unscoped = await conn.exists([f"kernel:presence:{healthy_kernel}"])
        assert set(scoped.keys()) == {b"presence", b"last_presence"}
        assert unscoped == 0

    async def test_other_agents_presence_is_never_read(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        other_agent_id: AgentId,
    ) -> None:
        """A presence reported under another agent's ID does not count for the kernel."""
        kernel_id = KernelId(uuid4())
        await valkey_schedule_client.update_kernel_presence_batch(other_agent_id, {kernel_id: True})

        statuses = await valkey_schedule_client.check_kernel_presence_status_batch({
            kernel_id: agent_id
        })

        assert statuses[kernel_id] is None

    async def test_update_kernel_presence_batch_healthy(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        healthy_kernel: KernelId,
    ) -> None:
        """Test batch update of kernel presence to healthy."""
        statuses = await valkey_schedule_client.check_kernel_presence_status_batch({
            healthy_kernel: agent_id
        })

        status = statuses[healthy_kernel]
        assert status is not None
        assert status.presence == HealthCheckStatus.HEALTHY

    async def test_update_kernel_presence_batch_unhealthy(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        unhealthy_kernel: KernelId,
    ) -> None:
        """Test batch update of kernel presence to unhealthy."""
        statuses = await valkey_schedule_client.check_kernel_presence_status_batch({
            unhealthy_kernel: agent_id
        })

        status = statuses[unhealthy_kernel]
        assert status is not None
        assert status.presence == HealthCheckStatus.UNHEALTHY

    async def test_delete_kernel_presence_batch(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        initialized_kernels: list[KernelId],
    ) -> None:
        """Test batch deletion of kernel presence status and check timestamps."""
        kernel_agents = dict.fromkeys(initialized_kernels, agent_id)

        await valkey_schedule_client.delete_kernel_presence_batch(kernel_agents)

        last_checks = await valkey_schedule_client.get_kernel_last_check_batch(
            agent_id, initialized_kernels
        )
        statuses = await valkey_schedule_client.check_kernel_presence_status_batch(kernel_agents)
        for kernel_id in initialized_kernels:
            assert last_checks[kernel_id] is None
            assert statuses[kernel_id] is None

    async def test_delete_kernel_presence_batch_empty(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test that empty batch deletion does nothing."""
        await valkey_schedule_client.delete_kernel_presence_batch({})

    async def test_check_kernel_presence_status_batch_not_found(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        kernel_ids: list[KernelId],
    ) -> None:
        """Test checking status for non-existent kernels."""
        statuses = await valkey_schedule_client.check_kernel_presence_status_batch(
            dict.fromkeys(kernel_ids, agent_id)
        )

        assert len(statuses) == len(kernel_ids)
        for kernel_id in kernel_ids:
            assert statuses[kernel_id] is None
        # Checking a kernel never creates its presence hash.
        async with valkey_schedule_client._client.client() as conn:
            exists = await conn.exists([
                valkey_schedule_client._get_kernel_presence_key(agent_id, kernel_id)
                for kernel_id in kernel_ids
            ])
        assert exists == 0

    async def test_check_kernel_presence_status_batch_empty(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Test checking status with empty input."""
        statuses = await valkey_schedule_client.check_kernel_presence_status_batch({})
        assert statuses == {}

    async def test_check_writes_manager_only_last_check_key(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        healthy_kernel: KernelId,
    ) -> None:
        """The check time goes to its own key with a TTL, not into the agent's hash."""
        statuses = await valkey_schedule_client.check_kernel_presence_status_batch({
            healthy_kernel: agent_id
        })
        status = statuses[healthy_kernel]
        assert status is not None

        key = f"kernel:last_check:{agent_id}:{healthy_kernel}"
        async with valkey_schedule_client._client.client() as conn:
            value = await conn.get(key)
            ttl = await conn.ttl(key)
            hash_last_check = await conn.hget(
                valkey_schedule_client._get_kernel_presence_key(agent_id, healthy_kernel),
                "last_check",
            )
        assert value is not None and int(value) == status.last_check
        assert 0 < ttl <= KERNEL_LAST_CHECK_TTL_SEC
        assert hash_last_check is None

    async def test_check_kernel_presence_status_batch_stale_detection(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        stale_kernel: KernelId,
    ) -> None:
        """Test that stale kernel presence is detected correctly."""
        statuses = await valkey_schedule_client.check_kernel_presence_status_batch({
            stale_kernel: agent_id
        })

        status = statuses[stale_kernel]
        assert status is not None
        assert status.presence == HealthCheckStatus.STALE

    async def test_check_kernel_presence_status_batch_mixed_states(
        self,
        mixed_state_kernels: KernelPresenceFixture,
    ) -> None:
        """Test checking multiple kernels with different states."""
        fixture = mixed_state_kernels
        statuses = await fixture.client.check_kernel_presence_status_batch(
            dict.fromkeys(fixture.kernel_ids, fixture.agent_id)
        )

        assert len(statuses) == len(fixture.kernel_ids)

        healthy_status = statuses[fixture.healthy_kernel_id]
        assert healthy_status is not None
        assert healthy_status.presence == HealthCheckStatus.HEALTHY

        unhealthy_status = statuses[fixture.unhealthy_kernel_id]
        assert unhealthy_status is not None
        assert unhealthy_status.presence == HealthCheckStatus.UNHEALTHY

        stale_status = statuses[fixture.stale_kernel_id]
        assert stale_status is not None
        assert stale_status.presence == HealthCheckStatus.STALE

        assert statuses[fixture.missing_kernel_id] is None

    @pytest.mark.parametrize(
        "fields",
        [
            {"presence": "1", "last_presence": "not-a-number"},
            {"presence": b"\xff", "last_presence": "0"},
            {"presence": "1", "last_presence": b"\xff\xfe"},
            {"last_presence": "0"},
        ],
        ids=["bad-timestamp", "bad-presence", "bad-bytes", "no-presence"],
    )
    async def test_malformed_presence_does_not_break_the_batch(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        healthy_kernel: KernelId,
        fields: dict[str | bytes, str | bytes],
    ) -> None:
        """A malformed hash reads as None and the other kernels are still parsed."""
        bad_kernel = KernelId(uuid4())
        wrong_type_kernel = KernelId(uuid4())
        async with valkey_schedule_client._client.client() as conn:
            await conn.hset(
                valkey_schedule_client._get_kernel_presence_key(agent_id, bad_kernel), fields
            )
            await conn.set(
                valkey_schedule_client._get_kernel_presence_key(agent_id, wrong_type_kernel),
                "not-a-hash",
            )

        statuses = await valkey_schedule_client.check_kernel_presence_status_batch({
            bad_kernel: agent_id,
            wrong_type_kernel: agent_id,
            healthy_kernel: agent_id,
        })

        assert statuses[bad_kernel] is None
        assert statuses[wrong_type_kernel] is None
        healthy_status = statuses[healthy_kernel]
        assert healthy_status is not None
        assert healthy_status.presence == HealthCheckStatus.HEALTHY


class TestAgentLastCheck:
    """Test cases for agent last check functionality.

    These tests verify the bidirectional kernel presence synchronization
    mechanism, specifically the agent_last_check timestamp tracking.
    """

    @pytest.fixture
    async def valkey_schedule_client(
        self,
        redis_container: tuple[str, HostPortPairModel],
    ) -> AsyncGenerator[ValkeyScheduleClient, None]:
        """Create ValkeyScheduleClient with real Redis container."""
        _, hostport_pair = redis_container

        valkey_target = ValkeyTarget(
            addr=hostport_pair.address,
        )

        client = await ValkeyScheduleClient.create(
            valkey_target=valkey_target,
            db_id=REDIS_LIVE_DB,
            human_readable_name="test-valkey-schedule-agent",
        )

        try:
            yield client
        finally:
            await client.close()

    @pytest.fixture
    def agent_id(self) -> AgentId:
        """Generate a unique agent ID."""
        return AgentId(f"test-agent-{uuid4().hex[:8]}")

    @pytest.fixture
    def agent_ids(self) -> set[AgentId]:
        """Generate multiple unique agent IDs."""
        return {AgentId(f"test-agent-{uuid4().hex[:8]}") for _ in range(2)}

    @pytest.fixture
    def kernel_ids(self) -> list[KernelId]:
        """Generate multiple kernel IDs."""
        return [KernelId(uuid4()) for _ in range(3)]

    async def test_get_agent_last_check_not_exists(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
    ) -> None:
        """Test that get_agent_last_check returns None for non-existent agent."""
        result = await valkey_schedule_client.get_agent_last_check(agent_id)
        assert result is None

    async def test_get_agent_last_check_exists(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
    ) -> None:
        """Test that get_agent_last_check returns timestamp when it exists."""
        # Manually set agent_last_check
        key = valkey_schedule_client._get_agent_last_check_key(agent_id)
        expected_timestamp = int(time())
        async with valkey_schedule_client._client.client() as conn:
            await conn.set(
                key,
                str(expected_timestamp),
                expiry=ExpirySet(ExpiryType.SEC, AGENT_LAST_CHECK_TTL_SEC),
            )

        result = await valkey_schedule_client.get_agent_last_check(agent_id)
        assert result == expected_timestamp

    @pytest.mark.parametrize("value", ["abc", "1.5", b"\xff"], ids=["text", "float", "bytes"])
    async def test_get_agent_last_check_malformed(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        value: str | bytes,
    ) -> None:
        """A non-integer agent last check reads as None."""
        async with valkey_schedule_client._client.client() as conn:
            await conn.set(valkey_schedule_client._get_agent_last_check_key(agent_id), value)

        assert await valkey_schedule_client.get_agent_last_check(agent_id) is None

    async def test_check_kernel_presence_updates_agent_last_check(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        kernel_ids: list[KernelId],
    ) -> None:
        """Test that check_kernel_presence_status_batch updates agent_last_check."""
        kernel_agents = dict.fromkeys(kernel_ids, agent_id)
        await valkey_schedule_client.initialize_kernel_presence_batch(kernel_agents)

        # Verify agent_last_check doesn't exist yet
        assert await valkey_schedule_client.get_agent_last_check(agent_id) is None

        await valkey_schedule_client.check_kernel_presence_status_batch(
            kernel_agents, agent_ids={agent_id}
        )

        result = await valkey_schedule_client.get_agent_last_check(agent_id)
        assert result is not None
        assert result > 0

    async def test_check_kernel_presence_updates_multiple_agents_last_check(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_ids: set[AgentId],
        kernel_ids: list[KernelId],
    ) -> None:
        """Test that check_kernel_presence_status_batch updates multiple agent_last_check."""
        agents = sorted(agent_ids)
        kernel_agents = {
            kernel_id: agents[i % len(agents)] for i, kernel_id in enumerate(kernel_ids)
        }
        await valkey_schedule_client.initialize_kernel_presence_batch(kernel_agents)

        await valkey_schedule_client.check_kernel_presence_status_batch(
            kernel_agents, agent_ids=agent_ids
        )

        for agent_id in agent_ids:
            result = await valkey_schedule_client.get_agent_last_check(agent_id)
            assert result is not None
            assert result > 0

    async def test_check_kernel_presence_without_agent_ids_does_not_update(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        kernel_ids: list[KernelId],
    ) -> None:
        """Test that check_kernel_presence_status_batch without agent_ids doesn't update."""
        kernel_agents = dict.fromkeys(kernel_ids, agent_id)
        await valkey_schedule_client.initialize_kernel_presence_batch(kernel_agents)

        await valkey_schedule_client.check_kernel_presence_status_batch(kernel_agents)

        result = await valkey_schedule_client.get_agent_last_check(agent_id)
        assert result is None

    async def test_get_kernel_last_check_batch(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        kernel_ids: list[KernelId],
    ) -> None:
        """The agent reads the manager's check time of its own kernels."""
        statuses = await valkey_schedule_client.check_kernel_presence_status_batch(
            dict.fromkeys(kernel_ids, agent_id)
        )

        last_checks = await valkey_schedule_client.get_kernel_last_check_batch(agent_id, kernel_ids)

        for kernel_id in kernel_ids:
            status = statuses[kernel_id]
            assert status is None  # never reported by the agent
            last_check = last_checks[kernel_id]
            assert last_check is not None
            assert last_check > 0

    async def test_other_agents_last_check_is_never_read(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_ids: set[AgentId],
    ) -> None:
        """A check time recorded under another agent's ID does not count for the kernel."""
        agent_id, other_agent_id = sorted(agent_ids)
        kernel_id = KernelId(uuid4())
        async with valkey_schedule_client._client.client() as conn:
            await conn.set(
                valkey_schedule_client._get_kernel_last_check_key(other_agent_id, kernel_id), "0"
            )

        last_checks = await valkey_schedule_client.get_kernel_last_check_batch(
            agent_id, [kernel_id]
        )

        assert last_checks == {kernel_id: None}

    async def test_agent_presence_report_does_not_write_last_check(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
        kernel_ids: list[KernelId],
    ) -> None:
        """Only the manager's check writes the kernel check time."""
        await valkey_schedule_client.update_kernel_presence_batch(
            agent_id, dict.fromkeys(kernel_ids, True)
        )

        last_checks = await valkey_schedule_client.get_kernel_last_check_batch(agent_id, kernel_ids)

        assert last_checks == dict.fromkeys(kernel_ids)

    async def test_get_kernel_last_check_batch_malformed(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
    ) -> None:
        """A malformed check time reads as None."""
        kernel_id = KernelId(uuid4())
        async with valkey_schedule_client._client.client() as conn:
            await conn.set(
                valkey_schedule_client._get_kernel_last_check_key(agent_id, kernel_id), "x"
            )

        last_checks = await valkey_schedule_client.get_kernel_last_check_batch(
            agent_id, [kernel_id]
        )

        assert last_checks == {kernel_id: None}

    async def test_get_kernel_last_check_batch_empty(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        agent_id: AgentId,
    ) -> None:
        """Test that get_kernel_last_check_batch returns empty dict for empty input."""
        assert await valkey_schedule_client.get_kernel_last_check_batch(agent_id, []) == {}


class TestForceTerminatedCleanupQueue:
    """Test cases for force-terminated session cleanup queue operations."""

    @pytest.fixture
    async def valkey_schedule_client(
        self,
        redis_container: tuple[str, HostPortPairModel],
    ) -> AsyncGenerator[ValkeyScheduleClient, None]:
        _, hostport_pair = redis_container
        valkey_target = ValkeyTarget(addr=hostport_pair.address)
        client = await ValkeyScheduleClient.create(
            valkey_target=valkey_target,
            db_id=REDIS_LIVE_DB,
            human_readable_name="test-force-terminated-cleanup",
        )
        try:
            key = ValkeyScheduleClient._get_force_terminated_cleanup_key()
            async with client._client.client() as conn:
                await conn.delete([key])
            yield client
        finally:
            await client.close()

    async def test_add_and_get_returns_session_ids(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Stored session IDs are returned as list[SessionId] by get."""
        session_ids = [SessionId(uuid4()) for _ in range(3)]

        await valkey_schedule_client.add_force_terminated_sessions(session_ids)
        result = await valkey_schedule_client.get_force_terminated_sessions()

        assert isinstance(result, list)
        assert set(result) == set(session_ids)
        for item in result:
            assert isinstance(item, UUID)

    async def test_get_empty_queue_returns_empty_list(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """Get from empty queue returns empty list."""
        result = await valkey_schedule_client.get_force_terminated_sessions()

        assert isinstance(result, list)
        assert result == []

    async def test_remove_deletes_only_specified_sessions(
        self, valkey_schedule_client: ValkeyScheduleClient
    ) -> None:
        """remove_force_terminated_sessions removes only specified IDs."""
        sid_keep = SessionId(uuid4())
        sid_remove = SessionId(uuid4())

        await valkey_schedule_client.add_force_terminated_sessions([sid_keep, sid_remove])
        await valkey_schedule_client.remove_force_terminated_sessions([sid_remove])
        result = await valkey_schedule_client.get_force_terminated_sessions()

        assert result == [sid_keep]


class TestReplicaProbeTargetClient:
    """Test ValkeyScheduleClient methods for ReplicaProbeTarget."""

    @pytest.fixture
    async def valkey_schedule_client(
        self,
        redis_container: tuple[str, HostPortPairModel],
    ) -> AsyncGenerator[ValkeyScheduleClient, None]:
        _, hostport_pair = redis_container
        client = await ValkeyScheduleClient.create(
            valkey_target=ValkeyTarget(addr=hostport_pair.address),
            db_id=REDIS_LIVE_DB,
            human_readable_name="test-route-probe-target",
        )
        try:
            yield client
        finally:
            await client.close()

    @pytest.fixture
    def replica_id(self) -> ReplicaID:
        return ReplicaID(uuid4())

    def _make_target(self, replica_id: ReplicaID) -> ReplicaProbeTarget:
        return ReplicaProbeTarget(
            replica_id=replica_id,
            health_path="/health",
            inference_port=8080,
            replica_host="10.0.0.1",
        )

    async def test_register_and_get(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        replica_id: ReplicaID,
    ) -> None:
        target = self._make_target(replica_id)
        await valkey_schedule_client.register_route_probe_targets_batch([target])
        results = await valkey_schedule_client.get_route_probe_targets_batch([replica_id])
        assert results[replica_id] == target

    async def test_register_batch_multiple(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
    ) -> None:
        targets = [self._make_target(ReplicaID(uuid4())) for _ in range(3)]
        await valkey_schedule_client.register_route_probe_targets_batch(targets)
        replica_ids = [t.replica_id for t in targets]
        results = await valkey_schedule_client.get_route_probe_targets_batch(replica_ids)
        for target in targets:
            assert results[target.replica_id] == target

    async def test_get_missing_returns_none(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        replica_id: ReplicaID,
    ) -> None:
        results = await valkey_schedule_client.get_route_probe_targets_batch([replica_id])
        assert results[replica_id] is None

    async def test_register_empty_does_nothing(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
    ) -> None:
        await valkey_schedule_client.register_route_probe_targets_batch([])

    async def test_get_empty_returns_empty_dict(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
    ) -> None:
        results = await valkey_schedule_client.get_route_probe_targets_batch([])
        assert results == {}

    async def test_register_overwrites_existing(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        replica_id: ReplicaID,
    ) -> None:
        await valkey_schedule_client.register_route_probe_targets_batch([
            self._make_target(replica_id)
        ])
        updated = ReplicaProbeTarget(
            replica_id=replica_id,
            health_path="/healthz",
            inference_port=9000,
            replica_host="10.0.0.2",
        )
        await valkey_schedule_client.register_route_probe_targets_batch([updated])
        results = await valkey_schedule_client.get_route_probe_targets_batch([replica_id])
        assert results[replica_id] == updated


class TestReplicaHealthStatusClient:
    """Test ValkeyScheduleClient methods for ReplicaHealthStatus."""

    @pytest.fixture
    async def valkey_schedule_client(
        self,
        redis_container: tuple[str, HostPortPairModel],
    ) -> AsyncGenerator[ValkeyScheduleClient, None]:
        _, hostport_pair = redis_container
        client = await ValkeyScheduleClient.create(
            valkey_target=ValkeyTarget(addr=hostport_pair.address),
            db_id=REDIS_LIVE_DB,
            human_readable_name="test-route-health-status",
        )
        try:
            yield client
        finally:
            await client.close()

    @pytest.fixture
    def replica_id(self) -> ReplicaID:
        return ReplicaID(uuid4())

    async def test_record_healthy_and_get(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        replica_id: ReplicaID,
    ) -> None:
        await valkey_schedule_client.record_route_health_statuses_batch([
            ReplicaHealthResult(replica_id=replica_id, healthy=True)
        ])
        results = await valkey_schedule_client.get_route_health_statuses_batch([replica_id])
        status = results[replica_id]
        assert status is not None
        assert status.healthy is True
        assert status.last_check > 0

    async def test_record_unhealthy_and_get(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        replica_id: ReplicaID,
    ) -> None:
        await valkey_schedule_client.record_route_health_statuses_batch([
            ReplicaHealthResult(replica_id=replica_id, healthy=False)
        ])
        results = await valkey_schedule_client.get_route_health_statuses_batch([replica_id])
        status = results[replica_id]
        assert status is not None
        assert status.healthy is False

    async def test_get_missing_returns_none(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        replica_id: ReplicaID,
    ) -> None:
        results = await valkey_schedule_client.get_route_health_statuses_batch([replica_id])
        assert results[replica_id] is None

    async def test_get_empty_returns_empty_dict(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
    ) -> None:
        results = await valkey_schedule_client.get_route_health_statuses_batch([])
        assert results == {}

    async def test_key_deletion_simulates_ttl_expiry(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        replica_id: ReplicaID,
    ) -> None:
        """Deleting the key (simulating TTL expiry) results in None → DEGRADED."""
        await valkey_schedule_client.record_route_health_statuses_batch([
            ReplicaHealthResult(replica_id=replica_id, healthy=True)
        ])
        key = valkey_schedule_client._get_route_health_status_key(replica_id)
        async with valkey_schedule_client._client.client() as conn:
            await conn.delete([key])
        results = await valkey_schedule_client.get_route_health_statuses_batch([replica_id])
        assert results[replica_id] is None

    async def test_record_batch_multiple(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
    ) -> None:
        replica_ids = [ReplicaID(uuid4()) for _ in range(3)]
        for rid in replica_ids:
            await valkey_schedule_client.record_route_health_statuses_batch([
                ReplicaHealthResult(replica_id=rid, healthy=True)
            ])
        results = await valkey_schedule_client.get_route_health_statuses_batch(replica_ids)
        assert len(results) == 3
        for rid in replica_ids:
            status = results[rid]
            assert status is not None
            assert status.healthy is True

    async def test_record_overwrites_previous(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        replica_id: ReplicaID,
    ) -> None:
        await valkey_schedule_client.record_route_health_statuses_batch([
            ReplicaHealthResult(replica_id=replica_id, healthy=True)
        ])
        await valkey_schedule_client.record_route_health_statuses_batch([
            ReplicaHealthResult(replica_id=replica_id, healthy=False)
        ])
        results = await valkey_schedule_client.get_route_health_statuses_batch([replica_id])
        status = results[replica_id]
        assert status is not None
        assert status.healthy is False

    async def test_record_consecutive_failures_round_trip(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        replica_id: ReplicaID,
    ) -> None:
        await valkey_schedule_client.record_route_health_statuses_batch([
            ReplicaHealthResult(replica_id=replica_id, healthy=False, consecutive_failures=4)
        ])
        results = await valkey_schedule_client.get_route_health_statuses_batch([replica_id])
        status = results[replica_id]
        assert status is not None
        assert status.consecutive_failures == 4

    async def test_record_with_custom_ttl(
        self,
        valkey_schedule_client: ValkeyScheduleClient,
        replica_id: ReplicaID,
    ) -> None:
        """Per-route ttl_sec is applied to the health status key."""
        await valkey_schedule_client.record_route_health_statuses_batch([
            ReplicaHealthResult(replica_id=replica_id, healthy=True, ttl_sec=7)
        ])
        key = valkey_schedule_client._get_route_health_status_key(replica_id)
        async with valkey_schedule_client._client.client() as conn:
            ttl = await conn.ttl(key)
        assert 0 < ttl <= 7
