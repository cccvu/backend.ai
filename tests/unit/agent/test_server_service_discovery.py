from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.backend.agent.server import service_discovery_ctx
from ai.backend.common.configs.service_discovery import ServiceDiscoveryConfig
from ai.backend.common.service_discovery.service_discovery import ServiceMetadata
from ai.backend.common.typed_validators import HostPortPair

_SERVER = "ai.backend.agent.server"


@dataclass
class _Mocks:
    redis_create: AsyncMock
    etcd_discovery: MagicMock
    sd_loop: MagicMock
    event_publisher: MagicMock
    apply_otel: MagicMock


@pytest.fixture
def mocks() -> Iterator[_Mocks]:
    event_publisher = MagicMock()
    event_publisher.return_value.start = AsyncMock()
    event_publisher.return_value.stop = AsyncMock()
    with (
        patch(f"{_SERVER}.RedisServiceDiscovery.create", new_callable=AsyncMock) as redis_create,
        patch(f"{_SERVER}.ETCDServiceDiscovery") as etcd_discovery,
        patch(f"{_SERVER}.ServiceDiscoveryLoop") as sd_loop,
        patch(f"{_SERVER}.ServiceDiscoveryEventPublisher", event_publisher),
        patch(f"{_SERVER}.BraceStyleAdapter.apply_otel") as apply_otel,
    ):
        yield _Mocks(
            redis_create=redis_create,
            etcd_discovery=etcd_discovery,
            sd_loop=sd_loop,
            event_publisher=event_publisher,
            apply_otel=apply_otel,
        )


def _make_agent_server(sd_config: ServiceDiscoveryConfig, *, otel_enabled: bool) -> MagicMock:
    local_config = SimpleNamespace(
        service_discovery=sd_config,
        agent_common=SimpleNamespace(
            announce_internal_addr=HostPortPair(host="127.0.0.1", port=6003),
        ),
        agent_default=SimpleNamespace(defaulted_id="i-test"),
        otel=SimpleNamespace(
            enabled=otel_enabled,
            log_level="INFO",
            endpoint="http://127.0.0.1:4317",
            max_queue_size=2048,
            max_export_batch_size=512,
        ),
        redis=MagicMock(),
    )
    agent_server = MagicMock()
    agent_server.local_config = local_config
    agent_server.read_agent_config = AsyncMock()
    return agent_server


class TestServiceDiscoveryDisabled:
    @pytest.mark.parametrize("sd_type", ["redis", "etcd"])
    async def test_creates_no_discovery_client_or_loop(self, mocks: _Mocks, sd_type: str) -> None:
        sd_config = ServiceDiscoveryConfig.model_validate({
            "enabled": False,
            "type": sd_type,
            "service-group": "agent",
        })
        agent_server = _make_agent_server(sd_config, otel_enabled=False)
        etcd = MagicMock()

        async with service_discovery_ctx(etcd, agent_server):
            pass

        mocks.redis_create.assert_not_awaited()
        mocks.etcd_discovery.assert_not_called()
        mocks.sd_loop.assert_not_called()
        mocks.event_publisher.assert_not_called()
        agent_server.read_agent_config.assert_not_awaited()
        assert etcd.mock_calls == []

    async def test_still_configures_otel(self, mocks: _Mocks) -> None:
        sd_config = ServiceDiscoveryConfig.model_validate({"enabled": False})
        agent_server = _make_agent_server(sd_config, otel_enabled=True)

        async with service_discovery_ctx(MagicMock(), agent_server):
            pass

        mocks.sd_loop.assert_not_called()
        mocks.apply_otel.assert_called_once()
        otel_spec = mocks.apply_otel.call_args.args[0]
        assert otel_spec.service_name == "agent"
        assert otel_spec.service_instance_name == "agent-i-test"


class TestServiceDiscoveryEnabled:
    async def test_redis_starts_and_closes_loop(self, mocks: _Mocks) -> None:
        sd_config = ServiceDiscoveryConfig.model_validate({})
        agent_server = _make_agent_server(sd_config, otel_enabled=False)

        async with service_discovery_ctx(MagicMock(), agent_server):
            mocks.redis_create.assert_awaited_once()
            mocks.sd_loop.assert_called_once()
            sd_type, client, metadata = mocks.sd_loop.call_args.args
            assert sd_type == sd_config.type
            assert client is mocks.redis_create.return_value
            assert isinstance(metadata, ServiceMetadata)
            assert metadata.service_group == "agent"
            assert metadata.display_name == "agent-i-test"
            mocks.sd_loop.return_value.close.assert_not_called()

        agent_server.read_agent_config.assert_awaited_once()
        mocks.etcd_discovery.assert_not_called()
        mocks.event_publisher.assert_not_called()
        mocks.sd_loop.return_value.close.assert_called_once()

    async def test_publishes_events_when_service_group_set(self, mocks: _Mocks) -> None:
        sd_config = ServiceDiscoveryConfig.model_validate({"service-group": "agent"})
        agent_server = _make_agent_server(sd_config, otel_enabled=False)

        async with service_discovery_ctx(MagicMock(), agent_server):
            pass

        mocks.sd_loop.assert_called_once()
        mocks.event_publisher.assert_called_once()
        mocks.event_publisher.return_value.start.assert_awaited_once()
        mocks.event_publisher.return_value.stop.assert_awaited_once()
