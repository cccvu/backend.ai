"""Shared fixtures and helpers for scheduler repository (db_source) tests.

These cover the BA-6134 ``agent_resources.reserved`` feature. The fixtures
build the FK-complete set of rows required to exercise ``ScheduleDBSource``
against a real database via ``with_tables``, and the module-level helpers seed
agent capacity, create PENDING sessions with pending ``resource_allocations``,
and assemble ``SessionAllocation`` values.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from dateutil.tz import tzutc

from ai.backend.common.auth import PublicKey
from ai.backend.common.clients.valkey_client.valkey_schedule.client import ValkeyScheduleClient
from ai.backend.common.data.agent.types import AgentInfo
from ai.backend.common.data.user.types import UserRole
from ai.backend.common.identifier.domain import DomainID
from ai.backend.common.identifier.resource_group import ResourceGroupID
from ai.backend.common.identifier.session_group import SessionGroupID
from ai.backend.common.types import (
    AccessKey,
    AgentId,
    ClusterMode,
    DefaultForUnspecified,
    DeviceName,
    KernelId,
    ResourceSlot,
    SecretKey,
    SessionId,
    SessionResult,
    SessionTypes,
    SlotName,
    SlotTypes,
)
from ai.backend.manager.data.agent.types import AgentHeartbeatUpsert, AgentStatus
from ai.backend.manager.data.kernel.types import KernelStatus
from ai.backend.manager.data.session.types import SessionStatus
from ai.backend.manager.data.user.types import UserStatus
from ai.backend.manager.models.agent import AgentRow
from ai.backend.manager.models.container_registry import ContainerRegistryRow
from ai.backend.manager.models.domain import DomainRow
from ai.backend.manager.models.group import GroupRow
from ai.backend.manager.models.image import ImageRow
from ai.backend.manager.models.kernel import KernelRow
from ai.backend.manager.models.keypair import KeyPairRow
from ai.backend.manager.models.rbac_models import (
    AssociationScopesEntitiesRow,
    EntityFieldRow,
    RoleRow,
    UserRoleRow,
)
from ai.backend.manager.models.resource_policy import (
    KeyPairResourcePolicyRow,
    ProjectResourcePolicyRow,
    UserResourcePolicyRow,
)
from ai.backend.manager.models.resource_slot import AgentResourceRow, ResourceAllocationRow
from ai.backend.manager.models.resource_slot.row import ResourceSlotTypeRow
from ai.backend.manager.models.scaling_group import ScalingGroupOpts, ScalingGroupRow
from ai.backend.manager.models.scheduling_history.row import SessionSchedulingHistoryRow
from ai.backend.manager.models.session import SessionDependencyRow, SessionRow
from ai.backend.manager.models.session_group.row import SessionGroupRow
from ai.backend.manager.models.user import UserRow
from ai.backend.manager.models.utils import ExtendedAsyncSAEngine
from ai.backend.manager.views.sokovan.allocation import (
    KernelAllocation,
    SessionAllocation,
)
from ai.backend.manager.views.sokovan.lifecycle import KernelCreationInfo
from ai.backend.testutils.db import with_tables

# Tables required to satisfy FK constraints for ScheduleDBSource, in dependency order.
_SCHEDULER_ROWS: list[type] = [
    DomainRow,
    ScalingGroupRow,
    UserResourcePolicyRow,
    ProjectResourcePolicyRow,
    KeyPairResourcePolicyRow,
    RoleRow,
    UserRoleRow,
    UserRow,
    KeyPairRow,
    GroupRow,
    AssociationScopesEntitiesRow,
    EntityFieldRow,
    AgentRow,
    ContainerRegistryRow,
    ImageRow,
    SessionGroupRow,
    SessionRow,
    KernelRow,
    ResourceSlotTypeRow,
    ResourceAllocationRow,
    AgentResourceRow,
    SessionDependencyRow,
    SessionSchedulingHistoryRow,
]

_AGENT_ADDR = "127.0.0.1:6001"


@pytest.fixture
async def db_with_cleanup(
    database_connection: ExtendedAsyncSAEngine,
) -> AsyncGenerator[ExtendedAsyncSAEngine, None]:
    """Database connection with the scheduler table set created (TRUNCATE CASCADE cleanup)."""
    async with with_tables(database_connection, _SCHEDULER_ROWS):
        yield database_connection


@pytest.fixture
def test_domain_id() -> DomainID:
    return DomainID(uuid.uuid4())


@pytest.fixture
def test_scaling_group_id() -> ResourceGroupID:
    return ResourceGroupID(uuid.uuid4())


@pytest.fixture
async def test_domain_name(
    db_with_cleanup: ExtendedAsyncSAEngine,
    test_domain_id: DomainID,
) -> AsyncGenerator[str, None]:
    domain_name = f"test-domain-{uuid.uuid4().hex[:8]}"
    async with db_with_cleanup.begin_session() as db_sess:
        db_sess.add(
            DomainRow(
                id=test_domain_id,
                name=domain_name,
                total_resource_slots=ResourceSlot({
                    "cpu": Decimal("1000"),
                    "mem": Decimal("1048576"),
                }),
            )
        )
        await db_sess.flush()
    yield domain_name


@pytest.fixture
async def test_scaling_group_name(
    db_with_cleanup: ExtendedAsyncSAEngine,
    test_scaling_group_id: ResourceGroupID,
) -> AsyncGenerator[str, None]:
    sg_name = f"test-sgroup-{uuid.uuid4().hex[:8]}"
    async with db_with_cleanup.begin_session() as db_sess:
        db_sess.add(
            ScalingGroupRow(
                id=test_scaling_group_id,
                name=sg_name,
                driver="static",
                scheduler="fifo",
                scheduler_opts=ScalingGroupOpts(
                    allowed_session_types=[],
                    pending_timeout=timedelta(hours=1),
                    config={},
                ),
                driver_opts={},
                is_active=True,
            )
        )
        await db_sess.flush()
    yield sg_name


@pytest.fixture
async def test_resource_policy_name(
    db_with_cleanup: ExtendedAsyncSAEngine,
) -> AsyncGenerator[str, None]:
    policy_name = f"test-policy-{uuid.uuid4().hex[:8]}"
    async with db_with_cleanup.begin_session() as db_sess:
        db_sess.add(
            ProjectResourcePolicyRow(
                name=policy_name,
                max_vfolder_count=10,
                max_quota_scope_size=-1,
                max_network_count=10,
            )
        )
        await db_sess.flush()
    yield policy_name


@pytest.fixture
async def test_user_resource_policy_name(
    db_with_cleanup: ExtendedAsyncSAEngine,
) -> AsyncGenerator[str, None]:
    policy_name = f"test-user-policy-{uuid.uuid4().hex[:8]}"
    async with db_with_cleanup.begin_session() as db_sess:
        db_sess.add(
            UserResourcePolicyRow(
                name=policy_name,
                max_vfolder_count=10,
                max_quota_scope_size=-1,
                max_session_count_per_model_session=10,
                max_customized_image_count=3,
            )
        )
        await db_sess.flush()
    yield policy_name


@pytest.fixture
async def test_keypair_resource_policy_name(
    db_with_cleanup: ExtendedAsyncSAEngine,
) -> AsyncGenerator[str, None]:
    policy_name = f"test-keypair-policy-{uuid.uuid4().hex[:8]}"
    async with db_with_cleanup.begin_session() as db_sess:
        db_sess.add(
            KeyPairResourcePolicyRow(
                name=policy_name,
                default_for_unspecified=DefaultForUnspecified.LIMITED,
                total_resource_slots=ResourceSlot({
                    "cpu": Decimal("100"),
                    "mem": Decimal("102400"),
                }),
                max_concurrent_sessions=10,
                max_containers_per_session=4,
                idle_timeout=600,
                max_session_lifetime=0,
                allowed_vfolder_hosts={},
            )
        )
        await db_sess.flush()
    yield policy_name


@pytest.fixture
async def test_user_uuid(
    db_with_cleanup: ExtendedAsyncSAEngine,
    test_domain_name: str,
    test_user_resource_policy_name: str,
) -> AsyncGenerator[uuid.UUID, None]:
    user_uuid = uuid.uuid4()
    async with db_with_cleanup.begin_session() as db_sess:
        db_sess.add(
            UserRow(
                uuid=user_uuid,
                email=f"test-user-{uuid.uuid4().hex[:8]}@test.com",
                username=f"test-user-{uuid.uuid4().hex[:8]}",
                role=UserRole.USER,
                status=UserStatus.ACTIVE,
                domain_name=test_domain_name,
                resource_policy=test_user_resource_policy_name,
            )
        )
        await db_sess.flush()
    yield user_uuid


@pytest.fixture
async def test_access_key(
    db_with_cleanup: ExtendedAsyncSAEngine,
    test_user_uuid: uuid.UUID,
    test_keypair_resource_policy_name: str,
) -> AsyncGenerator[AccessKey, None]:
    access_key = AccessKey(f"AKIA{uuid.uuid4().hex[:16].upper()}")
    async with db_with_cleanup.begin_session() as db_sess:
        db_sess.add(
            KeyPairRow(
                user_id=f"test-user-{uuid.uuid4().hex[:8]}@test.com",
                access_key=access_key,
                secret_key=SecretKey(f"SK{uuid.uuid4().hex}"),
                is_active=True,
                is_admin=False,
                resource_policy=test_keypair_resource_policy_name,
                rate_limit=1000,
                num_queries=0,
                user=test_user_uuid,
            )
        )
        await db_sess.flush()
    yield access_key


@pytest.fixture
async def test_group_id(
    db_with_cleanup: ExtendedAsyncSAEngine,
    test_domain_name: str,
    test_resource_policy_name: str,
) -> AsyncGenerator[uuid.UUID, None]:
    group_id = uuid.uuid4()
    async with db_with_cleanup.begin_session() as db_sess:
        db_sess.add(
            GroupRow(
                id=group_id,
                name=f"test-group-{uuid.uuid4().hex[:8]}",
                description="Test group",
                is_active=True,
                domain_name=test_domain_name,
                total_resource_slots=ResourceSlot(),
                allowed_vfolder_hosts={},
                resource_policy=test_resource_policy_name,
            )
        )
        await db_sess.flush()
    yield group_id


@pytest.fixture
async def test_agent_id(
    db_with_cleanup: ExtendedAsyncSAEngine,
    test_scaling_group_name: str,
    test_scaling_group_id: ResourceGroupID,
) -> AsyncGenerator[str, None]:
    agent_id = f"test-agent-{uuid.uuid4().hex[:8]}"
    async with db_with_cleanup.begin_session() as db_sess:
        db_sess.add(
            AgentRow(
                id=agent_id,
                status=AgentStatus.ALIVE,
                region="local",
                scaling_group=test_scaling_group_name,
                resource_group_id=test_scaling_group_id,
                available_slots=ResourceSlot({"cpu": Decimal("10"), "mem": Decimal("10240")}),
                occupied_slots=ResourceSlot(),
                addr=_AGENT_ADDR,
                version="1.0.0",
                architecture="x86_64",
            )
        )
        await db_sess.flush()
    yield agent_id


@pytest.fixture
async def resource_slot_types(
    db_with_cleanup: ExtendedAsyncSAEngine,
) -> AsyncGenerator[None, None]:
    """Seed resource slot types required by the resource_allocations FK."""
    async with db_with_cleanup.begin_session() as db_sess:
        for slot_name, slot_type in [("cpu", "count"), ("mem", "bytes")]:
            db_sess.add(ResourceSlotTypeRow(slot_name=slot_name, slot_type=slot_type))
        await db_sess.flush()
    yield


# --- Module-level helpers (shared across test files) ---


def make_creation_info(cpu: str = "2", mem: str = "4096") -> KernelCreationInfo:
    """Build a KernelCreationInfo whose get_resource_allocations() returns the given slots."""
    return KernelCreationInfo(
        container_id=f"container-{uuid.uuid4().hex[:8]}",
        resource_spec={
            "allocations": {
                "cpu": {"cpu": {"0": cpu}},
                "mem": {"mem": {"0": mem}},
            },
        },
        repl_in_port=2001,
        repl_out_port=2002,
        stdin_port=2003,
        stdout_port=2004,
    )


async def seed_agent_resources(
    db: ExtendedAsyncSAEngine,
    agent_id: str,
    *,
    cpu_capacity: Decimal,
    mem_capacity: Decimal,
    cpu_used: Decimal = Decimal("0"),
    mem_used: Decimal = Decimal("0"),
    cpu_reserved: Decimal = Decimal("0"),
    mem_reserved: Decimal = Decimal("0"),
) -> None:
    """Seed agent_resources rows (capacity/used/reserved) for cpu and mem slots."""
    async with db.begin_session() as db_sess:
        for slot_name, capacity, used, reserved in [
            ("cpu", cpu_capacity, cpu_used, cpu_reserved),
            ("mem", mem_capacity, mem_used, mem_reserved),
        ]:
            db_sess.add(
                AgentResourceRow(
                    agent_id=agent_id,
                    slot_name=slot_name,
                    capacity=capacity,
                    used=used,
                    reserved=reserved,
                )
            )
        await db_sess.flush()


async def create_pending_session_with_kernels(
    db: ExtendedAsyncSAEngine,
    *,
    domain_id: DomainID,
    domain_name: str,
    resource_group_id: ResourceGroupID,
    scaling_group_name: str,
    group_id: uuid.UUID,
    user_uuid: uuid.UUID,
    access_key: AccessKey,
    agent_assignments: list[tuple[str, Decimal, Decimal]],
    job_priority: int = 0,
    is_preemptible: bool = True,
    session_type: SessionTypes = SessionTypes.INTERACTIVE,
    session_status: SessionStatus = SessionStatus.PENDING,
    kernel_status: KernelStatus = KernelStatus.PENDING,
    assign_agents: bool = False,
    session_group_id: SessionGroupID | None = None,
) -> tuple[SessionId, list[KernelId]]:
    """Create a session with one kernel per agent assignment.

    The defaults create the PENDING shape: unassigned kernels with pending
    ``resource_allocations`` rows (``used``/``used_at``/``free_at`` all
    NULL). With ``assign_agents`` each kernel is placed on its assigned
    agent, RUNNING kernels get usage-reported allocation rows, and
    ``starts_at`` is set only for RUNNING sessions, mirroring the real
    transitions. Each entry in ``agent_assignments`` is ``(agent_id, cpu,
    mem)``. Returns the session id and the kernel ids in assignment order.
    """
    session_id = SessionId(uuid.uuid4())
    kernel_ids: list[KernelId] = []
    now = datetime.now(tzutc())

    total_cpu = sum((cpu for _, cpu, _ in agent_assignments), Decimal("0"))
    total_mem = sum((mem for _, _, mem in agent_assignments), Decimal("0"))
    cluster_mode = (
        ClusterMode.SINGLE_NODE if len(agent_assignments) == 1 else ClusterMode.MULTI_NODE
    )

    async with db.begin_session() as db_sess:
        db_sess.add(
            SessionRow(
                id=session_id,
                name=f"test-session-{uuid.uuid4().hex[:8]}",
                session_type=session_type,
                domain_id=domain_id,
                domain_name=domain_name,
                group_id=group_id,
                user_uuid=user_uuid,
                access_key=access_key,
                resource_group_id=resource_group_id,
                scaling_group_name=scaling_group_name,
                status=session_status,
                status_info="test",
                job_priority=job_priority,
                is_preemptible=is_preemptible,
                cluster_mode=cluster_mode,
                requested_slots=ResourceSlot({"cpu": total_cpu, "mem": total_mem}),
                created_at=now,
                starts_at=now if session_status == SessionStatus.RUNNING else None,
                session_group_id=session_group_id,
                images=["python:3.8"],
                vfolder_mounts=[],
                environ={},
                result=SessionResult.UNDEFINED,
            )
        )
        await db_sess.flush()

        usage_reported = kernel_status == KernelStatus.RUNNING
        for idx, (agent_id, cpu_requested, mem_requested) in enumerate(agent_assignments):
            kernel_id = KernelId(uuid.uuid4())
            kernel_ids.append(kernel_id)
            db_sess.add(
                KernelRow(
                    id=kernel_id,
                    session_id=session_id,
                    agent=agent_id if assign_agents else None,
                    agent_addr="127.0.0.1:6001" if assign_agents else None,
                    scaling_group=scaling_group_name,
                    resource_group_id=resource_group_id,
                    cluster_idx=idx,
                    cluster_role="main" if idx == 0 else "sub",
                    cluster_hostname=f"kernel-{uuid.uuid4().hex[:8]}",
                    image="python:3.8",
                    architecture="x86_64",
                    registry="docker.io",
                    status=kernel_status,
                    status_changed=now,
                    occupied_slots=(
                        ResourceSlot({"cpu": cpu_requested, "mem": mem_requested})
                        if usage_reported
                        else ResourceSlot()
                    ),
                    requested_slots=ResourceSlot({"cpu": cpu_requested, "mem": mem_requested}),
                    domain_name=domain_name,
                    group_id=group_id,
                    user_uuid=user_uuid,
                    access_key=access_key,
                    mounts=[],
                    environ={},
                    vfolder_mounts=[],
                    preopen_ports=[],
                    repl_in_port=2001,
                    repl_out_port=2002,
                    stdin_port=2003,
                    stdout_port=2004,
                )
            )
            await db_sess.flush()

            # Mirror the production ledger: the row's bucket matches the
            # kernel's lifecycle interval (RESERVED -> prereserved,
            # SCHEDULED..CREATING -> reserved, RUNNING -> used).
            prereserving = kernel_status == KernelStatus.RESERVED
            reserving = (
                not prereserving
                and not usage_reported
                and kernel_status in KernelStatus.resource_requested_statuses()
            )
            for slot_name, requested in [("cpu", cpu_requested), ("mem", mem_requested)]:
                db_sess.add(
                    ResourceAllocationRow(
                        kernel_id=kernel_id,
                        slot_name=slot_name,
                        requested=requested,
                        prereserved=requested if prereserving else Decimal(0),
                        reserved=requested if reserving else Decimal(0),
                        prereserved_at=now if prereserving else None,
                        reserved_at=now if (reserving or usage_reported) else None,
                        used=requested if usage_reported else None,
                        used_at=now if usage_reported else None,
                    )
                )
            await db_sess.flush()

    return session_id, kernel_ids


def make_session_allocations(
    *,
    session_id: SessionId,
    kernel_assignments: list[tuple[KernelId, str]],
) -> list[SessionAllocation]:
    """Assemble a single-session allocation batch.

    Each entry in ``kernel_assignments`` is ``(kernel_id, agent_id)``. The
    reservation amount is taken by the db_source from the kernel's pending
    ``resource_allocations`` rows.
    """
    kernel_allocations = [
        KernelAllocation(
            kernel_id=kernel_id,
            agent_id=AgentId(agent_id),
            agent_addr=_AGENT_ADDR,
        )
        for kernel_id, agent_id in kernel_assignments
    ]
    return [
        SessionAllocation(
            session_id=session_id,
            kernel_allocations=kernel_allocations,
            preempting_session_ids=(),
        )
    ]


async def fetch_agent_resources(
    db: ExtendedAsyncSAEngine,
    agent_id: str,
) -> dict[str, AgentResourceRow]:
    """Return agent_resources rows for the agent keyed by slot name."""
    async with db.begin_readonly_session() as db_sess:
        rows = (
            (
                await db_sess.execute(
                    sa.select(AgentResourceRow).where(AgentResourceRow.agent_id == agent_id)
                )
            )
            .scalars()
            .all()
        )
    return {row.slot_name: row for row in rows}


# --- Unavailable agents (fence, release, cleanup requeue) ---


@dataclass(frozen=True)
class ResourceGroupRef:
    id: ResourceGroupID
    name: str


async def add_resource_group(db: ExtendedAsyncSAEngine) -> ResourceGroupRef:
    group = ResourceGroupRef(ResourceGroupID(uuid.uuid4()), f"test-sgroup-{uuid.uuid4().hex[:8]}")
    async with db.begin_session() as db_sess:
        db_sess.add(
            ScalingGroupRow(
                id=group.id,
                name=group.name,
                driver="static",
                scheduler="fifo",
                scheduler_opts=ScalingGroupOpts(
                    allowed_session_types=[],
                    pending_timeout=timedelta(hours=1),
                    config={},
                ),
                driver_opts={},
                is_active=True,
            )
        )
    return group


async def add_agent(
    db: ExtendedAsyncSAEngine,
    group: ResourceGroupRef,
    status: AgentStatus,
    *,
    unavailable_for_sec: float | None = None,
    schedulable: bool = True,
) -> AgentId:
    """Add an agent; ``unavailable_for_sec`` backdates ``lost_at`` on the database clock."""
    agent_id = AgentId(f"test-agent-{uuid.uuid4().hex[:8]}")
    async with db.begin_session() as db_sess:
        db_sess.add(
            AgentRow(
                id=agent_id,
                status=status,
                region="local",
                scaling_group=group.name,
                resource_group_id=group.id,
                schedulable=schedulable,
                available_slots=ResourceSlot({"cpu": Decimal("10"), "mem": Decimal("10240")}),
                occupied_slots=ResourceSlot(),
                addr="127.0.0.1:6001",
                version="1.0.0",
                architecture="x86_64",
            )
        )
    if unavailable_for_sec is not None:
        await set_unavailable_for(db, agent_id, unavailable_for_sec)
    return agent_id


async def set_unavailable_for(db: ExtendedAsyncSAEngine, agent_id: AgentId, seconds: float) -> None:
    async with db.begin() as conn:
        await conn.execute(
            sa.update(AgentRow)
            .where(AgentRow.id == agent_id)
            .values(lost_at=sa.func.now() - sa.literal(timedelta(seconds=seconds), sa.Interval))
        )


async def fetch_agent_row(db: ExtendedAsyncSAEngine, agent_id: AgentId) -> dict[str, Any]:
    async with db.begin_readonly() as conn:
        row = (
            await conn.execute(sa.select(AgentRow.__table__).where(AgentRow.id == agent_id))
        ).one()
    return dict(row._mapping)


async def fetch_schedulable(db: ExtendedAsyncSAEngine, agent_id: AgentId) -> bool:
    return bool((await fetch_agent_row(db, agent_id))["schedulable"])


def heartbeat_upsert(agent_id: AgentId, group_name: str) -> AgentHeartbeatUpsert:
    return AgentHeartbeatUpsert.from_agent_info(
        agent_id=agent_id,
        agent_info=AgentInfo(
            ip="127.0.0.1",
            version="1.0.0",
            scaling_group=group_name,
            available_resource_slots=ResourceSlot({
                SlotName("cpu"): "10",
                SlotName("mem"): "10240",
            }),
            slot_key_and_units={
                SlotName("cpu"): SlotTypes.COUNT,
                SlotName("mem"): SlotTypes.BYTES,
            },
            compute_plugins={DeviceName("cpu"): {}},
            addr="127.0.0.1:6001",
            public_key=PublicKey(b"test-public-key"),
            public_host="127.0.0.1",
            images=b"\x82\xc4\x00\x00",
            region="local",
            architecture="x86_64",
            auto_terminate_abusing_kernel=False,
        ),
        heartbeat_received=datetime.now(tzutc()),
    )


async def delete_force_terminated_cleanup_key(client: ValkeyScheduleClient) -> None:
    async with client._client.client() as conn:
        await conn.delete([client._get_force_terminated_cleanup_key()])
