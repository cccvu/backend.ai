"""The START pass releases the CREATING lock without waiting for agents.

A real ``ScheduleCoordinator`` drives a real ``StartSessionsLifecycleHandler``
and ``SessionLauncher`` under an ``asyncio.Lock``-backed lock that records how
long each pass holds it. Agents are per-agent stubs; the repository is mocked.

Proven here:

- A hung agent's ``create_kernels`` (real 600 s bound) neither stretches the
  lock hold nor stops other sessions, in this pass or the next.
- Each agent's request is handed off before the PREPARED -> CREATING commit, so
  a failed commit loses no start: the next pass sends the session again and the
  agent's in-flight guard rejects the duplicate (at-least-once).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, override
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from dateutil.tz import tzutc

from ai.backend.common.identifier.architecture import ArchName
from ai.backend.common.identifier.resource_group import ResourceGroupID
from ai.backend.common.lock import AbstractDistributedLock
from ai.backend.common.types import (
    AccessKey,
    AgentId,
    ClusterMode,
    KernelId,
    ResourceSlot,
    SessionId,
    SessionResult,
    SessionTypes,
)
from ai.backend.manager.data.session.options import SessionHandlerOptions
from ai.backend.manager.data.session.types import (
    ImageSpec,
    MountSpec,
    ResourceSpec,
    SessionExecution,
    SessionIdentity,
    SessionInfo,
    SessionLifecycle,
    SessionMetadata,
    SessionMetrics,
    SessionNetwork,
    SessionStatus,
)
from ai.backend.manager.defs import LockID
from ai.backend.manager.models.network import NetworkType
from ai.backend.manager.repositories.scheduling_history.creators import (
    SessionSchedulingHistoryCreatorSpec,
)
from ai.backend.manager.sokovan.scheduler.coordinator import ScheduleCoordinator
from ai.backend.manager.sokovan.scheduler.factory import CoordinatorHandlers
from ai.backend.manager.sokovan.scheduler.handlers.lifecycle.start_sessions import (
    StartSessionsLifecycleHandler,
)
from ai.backend.manager.sokovan.scheduler.launcher.launcher import (
    SessionLauncher,
    SessionLauncherArgs,
)
from ai.backend.manager.sokovan.scheduler.scheduler import SchedulerComponents
from ai.backend.manager.sokovan.scheduler.types import ScheduleType
from ai.backend.manager.views.sokovan.image import ImageConfigData
from ai.backend.manager.views.sokovan.lifecycle import (
    KernelBindingData,
    SessionDataForStart,
    SessionWithKernels,
)
from ai.backend.manager.views.sokovan.search import SessionWithKernelsAndUserSearchResult

# Every pass must finish (and release the lock) well inside this, while the hung
# agent's create_kernels sits under the real 600 s AGENT_CREATE_KERNELS_TIMEOUT_SEC.
PASS_BOUND_SEC = 2.0
# How long to wait for a stub to observe a call the pass has already handed off.
OBSERVE_SEC = 1.0
# How long the stub pool's acquire suspends for a first contact.
FIRST_CONTACT_SEC = 0.05

HUNG_AGENT = AgentId("agent-hung")
OK_AGENT = AgentId("agent-ok")
IMAGE_ID = UUID("00000000-0000-0000-0000-000000000185")
IMAGE = "cr.backend.ai/stable/python:3.13-ubuntu24.04"
RESOURCE_GROUP = ResourceGroupID(uuid4())


# =============================================================================
# Data builders
# =============================================================================


@dataclass
class _PreparedSession:
    """One single-kernel VOLATILE session, as the coordinator and launcher see it."""

    session_id: SessionId
    kernel_id: KernelId
    agent_id: AgentId
    with_kernels: SessionWithKernels
    for_start: SessionDataForStart


def _prepared_session(agent_id: AgentId) -> _PreparedSession:
    session_id = SessionId(uuid4())
    kernel_id = KernelId(uuid4())
    creation_id = str(uuid4())
    user_uuid = uuid4()
    access_key = "test-access-key"
    name = f"session-{session_id}"
    now = datetime.now(tzutc())
    session_info = SessionInfo(
        identity=SessionIdentity(
            id=session_id,
            creation_id=creation_id,
            name=name,
            session_type=SessionTypes.INTERACTIVE,
            priority=0,
        ),
        metadata=SessionMetadata(
            name=name,
            domain_name="default",
            group_id=uuid4(),
            user_uuid=user_uuid,
            access_key=access_key,
            session_type=SessionTypes.INTERACTIVE,
            priority=0,
            created_at=now,
            tag=None,
        ),
        resource=ResourceSpec(
            cluster_mode=ClusterMode.SINGLE_NODE.value,
            cluster_size=1,
            occupying_slots=ResourceSlot(),
            requested_slots=ResourceSlot(),
            scaling_group_name="default",
            target_sgroup_names=None,
            agent_ids=[agent_id],
        ),
        image=ImageSpec(images=[IMAGE], tag=None),
        mounts=MountSpec(vfolder_mounts=None),
        execution=SessionExecution(
            environ=None,
            bootstrap_script=None,
            startup_command=None,
            use_host_network=False,
            callback_url=None,
        ),
        lifecycle=SessionLifecycle(
            status=SessionStatus.PREPARED,
            result=SessionResult.UNDEFINED,
            created_at=now,
            terminated_at=None,
            starts_at=None,
            status_changed=now,
            batch_timeout=None,
            status_info=None,
            status_data=None,
            status_history=None,
        ),
        metrics=SessionMetrics(num_queries=0, last_stat=None),
        network=SessionNetwork(network_type=NetworkType.VOLATILE, network_id=None),
        handler_options=SessionHandlerOptions(),
    )
    for_start = SessionDataForStart(
        session_id=session_id,
        creation_id=creation_id,
        access_key=AccessKey(access_key),
        session_type=SessionTypes.INTERACTIVE,
        name=name,
        cluster_mode=ClusterMode.SINGLE_NODE,
        kernels=[
            KernelBindingData(
                kernel_id=kernel_id,
                agent_id=agent_id,
                agent_addr=f"tcp://{agent_id}:6001",
                scaling_group="default",
                image=IMAGE,
                image_id=IMAGE_ID,
                architecture=ArchName("x86_64"),
            )
        ],
        user_uuid=user_uuid,
        user_email="test@example.com",
        user_name="test-user",
        environ={},
        network_type=NetworkType.VOLATILE,
    )
    # The START path reads only session_info from the coordinator's view.
    return _PreparedSession(
        session_id=session_id,
        kernel_id=kernel_id,
        agent_id=agent_id,
        with_kernels=SessionWithKernels(session_info=session_info, kernel_infos=[]),
        for_start=for_start,
    )


def _image_configs() -> dict[UUID, ImageConfigData]:
    return {
        IMAGE_ID: ImageConfigData(
            id=IMAGE_ID,
            canonical=IMAGE,
            architecture=ArchName("x86_64"),
            project="stable",
            is_local=False,
            digest="sha256:abc123",
            labels={},
            registry_name="cr.backend.ai",
            registry_url="https://cr.backend.ai",
            registry_username=None,
            registry_password=None,
        )
    }


def _search_result(sessions: Sequence[_PreparedSession]) -> SessionWithKernelsAndUserSearchResult:
    return SessionWithKernelsAndUserSearchResult(
        sessions=[s.for_start for s in sessions],
        image_configs=_image_configs(),
    )


def _committed_session_ids(update_with_history: AsyncMock) -> list[set[SessionId]]:
    """Session ids each ``update_with_history`` call moved to CREATING."""
    committed: list[set[SessionId]] = []
    for call in update_with_history.await_args_list:
        bulk_creator = call.args[1]
        ids: set[SessionId] = set()
        for spec in bulk_creator.specs:
            assert isinstance(spec, SessionSchedulingHistoryCreatorSpec)
            assert spec.from_status == SessionStatus.PREPARED
            assert spec.to_status == SessionStatus.CREATING
            ids.add(spec.session_id)
        committed.append(ids)
    return committed


# =============================================================================
# Lock and agent stand-ins
# =============================================================================


class _RecordingLock(AbstractDistributedLock):
    """A process-local lock that records how long each holder kept it."""

    _lock: asyncio.Lock
    _holds: list[float]
    _acquired_at: float

    def __init__(self, lock: asyncio.Lock, holds: list[float], lifetime: float) -> None:
        super().__init__(lifetime=lifetime)
        self._lock = lock
        self._holds = holds
        self._acquired_at = 0.0

    @override
    async def __aenter__(self) -> None:
        await self._lock.acquire()
        self._acquired_at = time.monotonic()

    @override
    async def __aexit__(self, *exc_info: Any) -> None:
        self._holds.append(time.monotonic() - self._acquired_at)
        self._lock.release()


class _RecordingLockFactory:
    """``DistributedLockFactory`` over one ``asyncio.Lock`` per lock id."""

    holds: dict[LockID, list[float]]
    _locks: dict[LockID, asyncio.Lock]

    def __init__(self) -> None:
        self.holds = {}
        self._locks = {}

    def __call__(self, lock_id: LockID, lifetime_hint: float) -> AbstractDistributedLock:
        lock = self._locks.setdefault(lock_id, asyncio.Lock())
        return _RecordingLock(lock, self.holds.setdefault(lock_id, []), lifetime_hint)


class _KernelCreationInProgress(Exception):
    """Stands in for the agent's ``ResourceError("Kernel creation already in progress")``."""


class _AgentStub:
    """An agent's ``create_kernels``, with the agent's in-flight guard.

    Mirrors ``AbstractAgent.track_create``: a request for a kernel id that is
    still being created is rejected; otherwise the creation is accepted and
    stays in flight until ``finish`` is set (never, unless a test sets it), as
    while a container's bootstrap runs or Docker hangs.
    """

    in_flight: set[KernelId]
    accepted: list[KernelId]
    rejected: list[KernelId]
    entered: asyncio.Event
    finish: asyncio.Event
    cancelled: asyncio.Event

    def __init__(self) -> None:
        self.in_flight = set()
        self.accepted = []
        self.rejected = []
        self.entered = asyncio.Event()
        self.finish = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def create_kernels(
        self,
        session_id: SessionId,
        kernel_ids: Sequence[KernelId],
        *args: Any,
    ) -> None:
        self.entered.set()
        duplicates = [k for k in kernel_ids if k in self.in_flight]
        if duplicates:
            self.rejected.extend(duplicates)
            raise _KernelCreationInProgress("Kernel creation already in progress")
        self.in_flight.update(kernel_ids)
        self.accepted.extend(kernel_ids)
        try:
            await self.finish.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        finally:
            self.in_flight.difference_update(kernel_ids)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def repository() -> AsyncMock:
    repository = AsyncMock()
    repository.get_all_resource_groups = AsyncMock(return_value=[RESOURCE_GROUP])
    repository.get_last_session_histories = AsyncMock(return_value={})
    repository.get_db_now = AsyncMock(side_effect=lambda: datetime.now(tzutc()))
    repository.update_with_history = AsyncMock(return_value=1)
    repository.update_session_network_id = AsyncMock(return_value=None)
    repository.update_session_error_info = AsyncMock(return_value=None)
    repository.invalidate_kernel_related_cache = AsyncMock(return_value=None)
    return repository


@pytest.fixture
def valkey_schedule() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def config_provider() -> MagicMock:
    provider = MagicMock()
    provider.config.manager.session_schedule_lock_lifetime = 30.0
    provider.config.debug.enabled = False
    provider.config.network.inter_container.default_driver = None
    return provider


@pytest.fixture
def hung_agent(per_agent_client_pool: MagicMock) -> _AgentStub:
    stub = _AgentStub()
    per_agent_client_pool.client(HUNG_AGENT).create_kernels.side_effect = stub.create_kernels
    return stub


@pytest.fixture
def ok_agent(per_agent_client_pool: MagicMock) -> AsyncMock:
    create_kernels: AsyncMock = per_agent_client_pool.client(OK_AGENT).create_kernels
    create_kernels.return_value = None
    return create_kernels


@pytest.fixture
def guarded_agent(per_agent_client_pool: MagicMock) -> _AgentStub:
    stub = _AgentStub()
    per_agent_client_pool.client(OK_AGENT).create_kernels.side_effect = stub.create_kernels
    return stub


@pytest.fixture
def slow_first_contact(per_agent_client_pool: MagicMock) -> None:
    """``acquire`` suspends, as a first contact's DB/Valkey lookup does.

    Without it the spawned task would reach the agent before the commit anyway,
    and the test could not tell whether START waited for the handoff.
    """

    @asynccontextmanager
    async def acquire(agent_id: AgentId) -> AsyncIterator[AsyncMock]:
        await asyncio.sleep(FIRST_CONTACT_SEC)
        yield per_agent_client_pool.client(agent_id)

    per_agent_client_pool.acquire.side_effect = acquire


@pytest.fixture
async def launcher(
    repository: AsyncMock,
    per_agent_client_pool: MagicMock,
    config_provider: MagicMock,
    valkey_schedule: AsyncMock,
) -> AsyncIterator[SessionLauncher]:
    launcher = SessionLauncher(
        SessionLauncherArgs(
            repository=repository,
            agent_client_pool=per_agent_client_pool,
            network_plugin_ctx=MagicMock(),
            config_provider=config_provider,
            valkey_schedule=valkey_schedule,
        )
    )
    yield launcher
    # Safety net; each test closes the launcher itself and checks the result.
    await launcher.close()


@pytest.fixture
def lock_factory() -> _RecordingLockFactory:
    return _RecordingLockFactory()


@pytest.fixture
def coordinator(
    repository: AsyncMock,
    valkey_schedule: AsyncMock,
    config_provider: MagicMock,
    launcher: SessionLauncher,
    lock_factory: _RecordingLockFactory,
) -> ScheduleCoordinator:
    handler = StartSessionsLifecycleHandler(launcher=launcher, repository=repository)
    return ScheduleCoordinator(
        valkey_schedule=valkey_schedule,
        components=SchedulerComponents(
            provisioner=MagicMock(),
            launcher=launcher,
            terminator=MagicMock(),
            repository=repository,
            config_provider=config_provider,
            hook_registry=MagicMock(),
        ),
        handlers=CoordinatorHandlers(
            lifecycle_handlers={ScheduleType.START: handler},
            promotion_specs={},
            kernel_handlers={},
            kernel_observers={},
            cleanup_handlers={},
        ),
        scheduling_controller=AsyncMock(),
        event_producer=AsyncMock(),
        lock_factory=lock_factory,
    )


async def _run_pass(coordinator: ScheduleCoordinator) -> bool:
    async with asyncio.timeout(PASS_BOUND_SEC):
        return await coordinator.process_lifecycle_schedule(ScheduleType.START)


async def _close_and_check(launcher: SessionLauncher) -> None:
    await launcher.close()
    assert not launcher._kernel_creations


# =============================================================================
# Tests
# =============================================================================


class TestStartLockHold:
    async def test_hung_agent_does_not_hold_the_lock_or_block_other_sessions(
        self,
        coordinator: ScheduleCoordinator,
        launcher: SessionLauncher,
        repository: AsyncMock,
        lock_factory: _RecordingLockFactory,
        hung_agent: _AgentStub,
        ok_agent: AsyncMock,
        slow_first_contact: None,
    ) -> None:
        session_a = _prepared_session(HUNG_AGENT)
        session_b = _prepared_session(OK_AGENT)
        session_c = _prepared_session(OK_AGENT)
        repository.get_sessions_for_handler.side_effect = [
            [session_a.with_kernels, session_b.with_kernels],
            [session_c.with_kernels],
        ]
        repository.search_sessions_with_kernels_and_user.side_effect = [
            _search_result([session_a, session_b]),
            _search_result([session_c]),
        ]
        sent_at_commit: list[tuple[set[KernelId], int]] = []

        async def commit(*args: Any) -> int:
            sent_at_commit.append((set(hung_agent.in_flight), ok_agent.await_count))
            return 1

        repository.update_with_history.side_effect = commit
        try:
            # Pass 1: A on the hung agent, B on the healthy one. Both requests are
            # handed off before the commit.
            assert await _run_pass(coordinator)
            assert sent_at_commit == [({session_a.kernel_id}, 1)]
            await asyncio.wait_for(hung_agent.entered.wait(), OBSERVE_SEC)
            assert hung_agent.accepted == [session_a.kernel_id]
            assert ok_agent.await_count == 1
            assert ok_agent.await_args_list[0].args[0] == session_b.session_id
            assert _committed_session_ids(repository.update_with_history) == [
                {session_a.session_id, session_b.session_id}
            ]

            # Pass 2: C on the healthy agent while A's create_kernels is still pending.
            assert await _run_pass(coordinator)
            assert sent_at_commit[1] == ({session_a.kernel_id}, 2)
            assert hung_agent.in_flight == {session_a.kernel_id}
            assert not hung_agent.cancelled.is_set()
            assert ok_agent.await_count == 2
            assert ok_agent.await_args_list[1].args[0] == session_c.session_id
            assert _committed_session_ids(repository.update_with_history)[1] == {
                session_c.session_id
            }

            holds = lock_factory.holds[LockID.LOCKID_SOKOVAN_TARGET_CREATING]
            assert len(holds) == 2
            assert all(hold < PASS_BOUND_SEC for hold in holds)
            repository.update_session_error_info.assert_not_awaited()
        finally:
            await _close_and_check(launcher)
        # close() cancelled the hung call rather than leaving it to the 600 s bound.
        assert hung_agent.cancelled.is_set()
        assert not hung_agent.in_flight

    async def test_failed_commit_resends_and_agent_rejects_the_duplicate(
        self,
        coordinator: ScheduleCoordinator,
        launcher: SessionLauncher,
        repository: AsyncMock,
        valkey_schedule: AsyncMock,
        lock_factory: _RecordingLockFactory,
        guarded_agent: _AgentStub,
        slow_first_contact: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        session = _prepared_session(OK_AGENT)
        # The commit fails on pass 1, so the session is still PREPARED on pass 2.
        repository.get_sessions_for_handler.side_effect = [
            [session.with_kernels],
            [session.with_kernels],
        ]
        repository.search_sessions_with_kernels_and_user.side_effect = [
            _search_result([session]),
            _search_result([session]),
        ]
        in_flight_at_commit: list[set[KernelId]] = []

        async def commit(*args: Any) -> int:
            in_flight_at_commit.append(set(guarded_agent.in_flight))
            if len(in_flight_at_commit) == 1:
                raise ConnectionError("database unavailable")
            return 1

        repository.update_with_history.side_effect = commit
        failure_recorded = asyncio.Event()
        valkey_schedule.record_session_failed_agents.side_effect = (
            lambda *args, **kwargs: failure_recorded.set()
        )
        try:
            # Pass 1: the request is handed off before the commit, which then fails.
            with caplog.at_level(logging.ERROR):
                assert await _run_pass(coordinator)
            assert in_flight_at_commit == [{session.kernel_id}]
            assert "database unavailable" in caplog.text

            # Pass 2: the still-PREPARED session is sent again; the agent rejects
            # the duplicate while the first creation is still in flight.
            assert await _run_pass(coordinator)
            await asyncio.wait_for(failure_recorded.wait(), OBSERVE_SEC)
            assert guarded_agent.accepted == [session.kernel_id]
            assert guarded_agent.rejected == [session.kernel_id]
            assert guarded_agent.in_flight == {session.kernel_id}
            valkey_schedule.record_session_failed_agents.assert_awaited_once_with(
                session.session_id, [OK_AGENT]
            )
            assert _committed_session_ids(repository.update_with_history) == [
                {session.session_id},
                {session.session_id},
            ]

            holds = lock_factory.holds[LockID.LOCKID_SOKOVAN_TARGET_CREATING]
            assert len(holds) == 2
            assert all(hold < PASS_BOUND_SEC for hold in holds)
        finally:
            await _close_and_check(launcher)
