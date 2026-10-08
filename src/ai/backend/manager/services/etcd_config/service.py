from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from pydantic import TypeAdapter

from ai.backend.common.types import AcceleratorMetadata
from ai.backend.logging.utils import BraceStyleAdapter
from ai.backend.manager.errors.api import InvalidAPIParameters

from .actions.delete_config import DeleteConfigAction, DeleteConfigActionResult
from .actions.get_config import GetConfigAction, GetConfigActionResult
from .actions.get_resource_metadata import (
    GetResourceMetadataAction,
    GetResourceMetadataActionResult,
)
from .actions.get_resource_slots import GetResourceSlotsAction, GetResourceSlotsActionResult
from .actions.get_vfolder_types import GetVfolderTypesAction, GetVfolderTypesActionResult
from .actions.set_config import SetConfigAction, SetConfigActionResult

if TYPE_CHECKING:
    from ai.backend.common.clients.valkey_client.valkey_stat.client import ValkeyStatClient
    from ai.backend.common.etcd import AsyncEtcd
    from ai.backend.manager.config.provider import ManagerConfigProvider
    from ai.backend.manager.repositories.etcd_config import EtcdConfigRepository

__all__ = ("EtcdConfigService",)

log = BraceStyleAdapter(logging.getLogger(__spec__.name))

_ACCELERATOR_METADATA_ADAPTER: Final[TypeAdapter[AcceleratorMetadata]] = TypeAdapter(
    AcceleratorMetadata
)

KNOWN_SLOT_METADATA: dict[str, AcceleratorMetadata] = {
    "cpu": {
        "slot_name": "cpu",
        "description": "CPU",
        "human_readable_name": "CPU",
        "display_unit": "Core",
        "number_format": {"binary": False, "round_length": 0},
        "display_icon": "cpu",
    },
    "mem": {
        "slot_name": "ram",
        "description": "Memory",
        "human_readable_name": "RAM",
        "display_unit": "GiB",
        "number_format": {"binary": True, "round_length": 0},
        "display_icon": "cpu",
    },
    "cuda.device": {
        "slot_name": "cuda.device",
        "human_readable_name": "GPU",
        "description": "CUDA-capable GPU",
        "display_unit": "GPU",
        "number_format": {"binary": False, "round_length": 0},
        "display_icon": "gpu1",
    },
    "cuda.shares": {
        "slot_name": "cuda.shares",
        "human_readable_name": "fGPU",
        "description": "CUDA-capable GPU (fractional)",
        "display_unit": "fGPU",
        "number_format": {"binary": False, "round_length": 2},
        "display_icon": "gpu1",
    },
    "rocm.device": {
        "slot_name": "rocm.device",
        "human_readable_name": "GPU",
        "description": "ROCm-capable GPU",
        "display_unit": "GPU",
        "number_format": {"binary": False, "round_length": 0},
        "display_icon": "gpu2",
    },
    "tpu.device": {
        "slot_name": "tpu.device",
        "human_readable_name": "TPU",
        "description": "TPU device",
        "display_unit": "GPU",
        "number_format": {"binary": False, "round_length": 0},
        "display_icon": "tpu",
    },
}


@dataclass
class EtcdConfigService:
    """Service for etcd configuration operations."""

    _repository: EtcdConfigRepository
    _config_provider: ManagerConfigProvider
    _etcd: AsyncEtcd
    _valkey_stat: ValkeyStatClient

    def __init__(
        self,
        *,
        repository: EtcdConfigRepository,
        config_provider: ManagerConfigProvider,
        etcd: AsyncEtcd,
        valkey_stat: ValkeyStatClient,
    ) -> None:
        self._repository = repository
        self._config_provider = config_provider
        self._etcd = etcd
        self._valkey_stat = valkey_stat

    async def get_resource_slots(
        self, action: GetResourceSlotsAction
    ) -> GetResourceSlotsActionResult:
        """Get system-wide known resource slots."""
        known_slots = await self._config_provider.legacy_etcd_config_loader.get_resource_slots()
        return GetResourceSlotsActionResult(
            slots={str(k): v for k, v in known_slots.items()},
        )

    async def get_resource_metadata(
        self, action: GetResourceMetadataAction
    ) -> GetResourceMetadataActionResult:
        """Get resource metadata with optional scaling group filter."""
        known_slots = await self._config_provider.legacy_etcd_config_loader.get_resource_slots()

        # Preconfigured metadata always wins; agents report metadata only for other slots.
        accelerator_metadata: dict[str, AcceleratorMetadata] = {
            slot_name: metadata
            for slot_name, metadata in KNOWN_SLOT_METADATA.items()
            if slot_name in known_slots
        }
        unknown_slots = [
            str(slot_name) for slot_name in known_slots if slot_name not in KNOWN_SLOT_METADATA
        ]
        if unknown_slots:
            accelerator_metadata.update(await self._get_reported_metadata(unknown_slots))

        # Optionally filter by the slots reported by the given resource group's agents
        if action.sgroup is not None:
            available_slot_keys = await self._repository.get_available_agent_slots(action.sgroup)
            accelerator_metadata = {
                str(k): v
                for k, v in accelerator_metadata.items()
                if k in {"cpu", "mem", *available_slot_keys}
            }

        return GetResourceMetadataActionResult(metadata=accelerator_metadata)

    async def _get_reported_metadata(
        self, slot_names: Sequence[str]
    ) -> dict[str, AcceleratorMetadata]:
        """Read agent-reported metadata of the given slots, skipping malformed entries."""
        reported = await self._valkey_stat.get_computer_metadata(slot_names)
        metadata: dict[str, AcceleratorMetadata] = {}
        skipped: list[str] = []
        for slot_name, raw in reported.items():
            try:
                metadata[slot_name] = _ACCELERATOR_METADATA_ADAPTER.validate_json(raw)
            except ValueError:
                skipped.append(slot_name)
        if skipped:
            log.warning("Skipped malformed reported metadata of slots: {}", ", ".join(skipped))
        return metadata

    async def get_vfolder_types(self, action: GetVfolderTypesAction) -> GetVfolderTypesActionResult:
        """Get available vfolder types."""
        vfolder_types = await self._config_provider.legacy_etcd_config_loader.get_vfolder_types()
        return GetVfolderTypesActionResult(types=list(vfolder_types))

    async def get_config(self, action: GetConfigAction) -> GetConfigActionResult:
        """Get raw etcd key-value."""
        if action.prefix:
            tree_value = dict(await self._etcd.get_prefix_dict(action.key))
            return GetConfigActionResult(result=tree_value)
        scalar_value = await self._etcd.get(action.key)
        return GetConfigActionResult(result=scalar_value)

    async def set_config(self, action: SetConfigAction) -> SetConfigActionResult:
        """Set raw etcd key-value."""
        if isinstance(action.value, Mapping):
            updates: dict[str, Any] = {}

            def flatten(prefix: str, o: Mapping[str, Any]) -> None:
                for k, v in o.items():
                    inner_prefix = prefix if k == "" else f"{prefix}/{k}"
                    if isinstance(v, Mapping):
                        flatten(inner_prefix, v)
                    else:
                        updates[inner_prefix] = v

            flatten(action.key, action.value)
            if len(updates) > 16:
                raise InvalidAPIParameters(
                    "Too large update! Split into smaller key-value pair sets."
                )
            await self._etcd.put_dict(updates)
        else:
            await self._etcd.put(action.key, action.value)
        return SetConfigActionResult()

    async def delete_config(self, action: DeleteConfigAction) -> DeleteConfigActionResult:
        """Delete raw etcd key-value."""
        if action.prefix:
            await self._etcd.delete_prefix(action.key)
        else:
            await self._etcd.delete(action.key)
        return DeleteConfigActionResult()
