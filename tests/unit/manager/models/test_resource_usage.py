from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

import pytest

from ai.backend.common.types import ResourceSlot
from ai.backend.manager.models.kernel import KernelRow
from ai.backend.manager.models.resource_usage import (
    KernelStatUsage,
    parse_kernel_stat_usage,
    parse_resource_usage,
)


def _kernel(agent: str | None = "agent-a") -> KernelRow:
    return KernelRow(
        id=uuid.uuid4(),
        agent=agent,
        occupied_slots=ResourceSlot({"cpu": Decimal(2), "mem": Decimal(1024)}),
        attached_devices={},
        vfolder_mounts=[],
        mounts=[],
        resource_opts={},
    )


class TestParseKernelStatUsage:
    def test_figures(self) -> None:
        stat = {
            "cpu_used": {"current": "1.5"},
            "mem": {"capacity": "2048"},
            "io_scratch_size": {"stats.max": "10"},
            "io_read": {"current": "3"},
            "io_write": {"current": "4"},
        }
        assert parse_kernel_stat_usage("k", stat) == KernelStatUsage(
            cpu_used=1.5, mem_used=2048, disk_used=10, io_read=3, io_write=4
        )

    @pytest.mark.parametrize("stat", [None, {}])
    def test_missing_statistics_give_zeros(self, stat: dict[str, Any] | None) -> None:
        assert parse_kernel_stat_usage("k", stat) == KernelStatUsage()

    @pytest.mark.parametrize(
        "stat",
        [
            {"cpu_used": {"current": "not-a-number"}},
            {"cpu_used": "flat"},
            {"mem": {"capacity": [1, 2]}},
            {"io_read": 5},
        ],
        ids=["bad-number", "flat-metric", "bad-type", "int-metric"],
    )
    def test_malformed_statistics_give_zeros(self, stat: dict[str, Any]) -> None:
        assert parse_kernel_stat_usage("k", stat) == KernelStatUsage()


class TestParseResourceUsage:
    def test_kernel_without_statistics_keeps_its_allocation(self) -> None:
        usage = parse_resource_usage(_kernel(), None)

        assert usage.cpu_allocated == 2.0
        assert usage.mem_allocated == 1024
        assert usage.cpu_used == 0.0
        assert usage.agent_ids == {"agent-a"}
