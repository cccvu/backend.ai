from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from decimal import Decimal

import msgpack as plain_msgpack
import pytest

from ai.backend.common import msgpack
from ai.backend.common.clients.valkey_client.valkey_stat.client import ValkeyStatClient
from ai.backend.common.defs import REDIS_STATISTICS_DB
from ai.backend.common.typed_validators import HostPortPair as HostPortPairModel
from ai.backend.common.types import ValkeyTarget


class TestValkeyStatClient:
    @pytest.fixture
    async def test_valkey_stat(
        self, redis_container: tuple[str, HostPortPairModel]
    ) -> AsyncIterator[ValkeyStatClient]:
        hostport_pair: HostPortPairModel = redis_container[1]
        valkey_target = ValkeyTarget(
            addr=hostport_pair.address,
        )
        client = await ValkeyStatClient.create(
            valkey_target,
            human_readable_name="test.stat",
            db_id=REDIS_STATISTICS_DB,
        )
        try:
            yield client
        finally:
            await client.close()

    async def test_valkey_stat_expiration(self, test_valkey_stat: ValkeyStatClient) -> None:
        """Test key expiration functionality."""
        test_key = f"test-key-exp-{uuid.uuid4().hex[:8]}"
        test_value = b"test-value-exp"

        # Set with custom expiration
        await test_valkey_stat.set(test_key, test_value, expire_sec=1)

        # Verify key exists immediately
        result = await test_valkey_stat._get_raw(test_key)
        assert result == test_value

    async def test_valkey_stat_multiple_keys(self, test_valkey_stat: ValkeyStatClient) -> None:
        """Test multiple key operations."""
        test_keys = [f"test-key-{i}-{uuid.uuid4().hex[:8]}" for i in range(3)]
        test_values = [f"test-value-{i}".encode() for i in range(3)]

        # Set multiple keys
        key_value_map = dict(zip(test_keys, test_values, strict=True))
        await test_valkey_stat.set_multiple_keys(key_value_map)

        # Get multiple keys
        results = await test_valkey_stat._get_multiple_keys(test_keys)
        assert len(results) == len(test_keys)
        for i, result in enumerate(results):
            assert result == test_values[i]

        # Clean up
        deleted_count = await test_valkey_stat.delete(test_keys)
        assert deleted_count == len(test_keys)

    async def test_get_computer_metadata_reads_only_requested_slots(
        self, test_valkey_stat: ValkeyStatClient
    ) -> None:
        suffix = uuid.uuid4().hex[:8]
        requested = f"requested-{suffix}.device"
        other = f"other-{suffix}.device"
        missing = f"missing-{suffix}.device"
        await test_valkey_stat.store_computer_metadata({
            requested: b'{"slot_name": "requested"}',
            other: b'{"slot_name": "other"}',
        })

        result = await test_valkey_stat.get_computer_metadata([requested, missing])

        assert result == {requested: b'{"slot_name": "requested"}'}
        assert await test_valkey_stat.get_computer_metadata([]) == {}

class TestAgentScopedKernelKeys:
    """Kernel statistics and commit statuses are stored under the hosting agent's ID."""

    @pytest.fixture
    async def valkey_stat(
        self, redis_container: tuple[str, HostPortPairModel]
    ) -> AsyncIterator[ValkeyStatClient]:
        hostport_pair: HostPortPairModel = redis_container[1]
        client = await ValkeyStatClient.create(
            ValkeyTarget(addr=hostport_pair.address),
            human_readable_name="test.stat.kernel",
            db_id=REDIS_STATISTICS_DB,
        )
        try:
            yield client
        finally:
            await client.close()

    @pytest.fixture
    def kernel_id(self) -> str:
        return str(uuid.uuid4())

    async def test_kernel_statistics_key_is_agent_scoped(
        self, valkey_stat: ValkeyStatClient, kernel_id: str
    ) -> None:
        stat = {"cpu_util": {"current": "12.5", "pct": "12.50"}}
        await valkey_stat.set_kernel_statistics_batch("agent-a", {kernel_id: msgpack.packb(stat)})

        raw = await valkey_stat._get_raw(f"kstat.agent-a.{kernel_id}")
        assert raw is not None
        assert await valkey_stat._get_raw(kernel_id) is None
        assert await valkey_stat.get_kernel_statistics("agent-a", kernel_id) == stat
        assert await valkey_stat.get_user_kernel_statistics_batch([("agent-a", kernel_id)]) == [
            stat
        ]

    async def test_other_agents_kernel_statistics_are_never_read(
        self, valkey_stat: ValkeyStatClient, kernel_id: str
    ) -> None:
        # A value planted under another agent's ID for the same kernel is ignored.
        planted = {"cpu_util": {"current": "99", "pct": "99.00"}}
        await valkey_stat.set_kernel_statistics_batch(
            "agent-b", {kernel_id: msgpack.packb(planted)}
        )

        assert await valkey_stat.get_kernel_statistics("agent-a", kernel_id) is None
        assert await valkey_stat.get_user_kernel_statistics_batch([("agent-a", kernel_id)]) == [
            None
        ]

    async def test_kernel_without_agent_has_no_statistics(
        self, valkey_stat: ValkeyStatClient, kernel_id: str
    ) -> None:
        assert await valkey_stat.get_user_kernel_statistics_batch([(None, kernel_id)]) == [None]
        assert await valkey_stat.get_user_kernel_statistics_batch([]) == []

    @pytest.mark.parametrize(
        "raw",
        [
            b"\xc1",  # never used in msgpack
            b"\x93\x01\x02",  # truncated array
            msgpack.packb([1, 2, 3]),  # not a mapping
            msgpack.packb("text"),
        ],
        ids=["invalid", "truncated", "list", "str"],
    )
    async def test_malformed_kernel_statistics_read_as_none(
        self, valkey_stat: ValkeyStatClient, kernel_id: str, raw: bytes
    ) -> None:
        good_kernel_id = str(uuid.uuid4())
        good = {"mem": {"current": "1024"}}
        await valkey_stat.set_kernel_statistics_batch(
            "agent-a", {kernel_id: raw, good_kernel_id: msgpack.packb(good)}
        )

        assert await valkey_stat.get_kernel_statistics("agent-a", kernel_id) is None
        assert await valkey_stat.get_user_kernel_statistics_batch([
            ("agent-a", kernel_id),
            (None, str(uuid.uuid4())),
            ("agent-a", good_kernel_id),
        ]) == [None, None, good]

    async def test_kernel_statistics_decode_no_extension_types(
        self, valkey_stat: ValkeyStatClient, kernel_id: str
    ) -> None:
        # Extension types are left undecoded instead of being deserialized.
        await valkey_stat.set_kernel_statistics_batch(
            "agent-a", {kernel_id: msgpack.packb({"cpu_util": {"current": Decimal("1.5")}})}
        )

        [stat] = await valkey_stat.get_user_kernel_statistics_batch([("agent-a", kernel_id)])

        assert stat is not None
        assert isinstance(stat["cpu_util"]["current"], plain_msgpack.ExtType)

    async def test_delete_kernel_statistics(
        self, valkey_stat: ValkeyStatClient, kernel_id: str
    ) -> None:
        await valkey_stat.set_kernel_statistics_batch("agent-a", {kernel_id: msgpack.packb({})})

        assert await valkey_stat.delete_kernel_statistics([(None, kernel_id)]) == 0
        assert await valkey_stat.delete_kernel_statistics([("agent-a", kernel_id)]) == 1
        assert await valkey_stat._get_raw(f"kstat.agent-a.{kernel_id}") is None

    async def test_kernel_commit_status_key_is_agent_scoped(
        self, valkey_stat: ValkeyStatClient, kernel_id: str
    ) -> None:
        await valkey_stat.update_kernel_commit_statuses("agent-a", [kernel_id], 60)

        assert await valkey_stat._get_raw(f"kernel_commit.agent-a.{kernel_id}") == b"ongoing"
        assert await valkey_stat.get_kernel_commit_statuses([
            ("agent-a", kernel_id),
            ("agent-b", kernel_id),
            (None, kernel_id),
        ]) == [b"ongoing", None, None]
