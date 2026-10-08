"""Tests for CleanupForceTerminatedHandler."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ai.backend.common.types import (
    AccessKey,
    AgentId,
    KernelId,
    ResourceSlot,
    SessionId,
    SessionTypes,
)
from ai.backend.manager.data.kernel.types import KernelStatus
from ai.backend.manager.data.session.types import SessionStatus
from ai.backend.manager.sokovan.recorder.context import RecorderContext
from ai.backend.manager.sokovan.scheduler.handlers.cleanup.force_terminated import (
    CleanupForceTerminatedHandler,
)
from ai.backend.manager.sokovan.scheduler.terminator import terminator as terminator_module
from ai.backend.manager.sokovan.scheduler.terminator.terminator import (
    SessionTerminator,
    SessionTerminatorArgs,
)
from ai.backend.manager.views.sokovan.session import (
    TerminatingKernelData,
    TerminatingSessionData,
)


@pytest.fixture
def mock_terminator() -> AsyncMock:
    terminator = AsyncMock()
    terminator.terminate_sessions_for_handler = AsyncMock(return_value=[])
    return terminator


@pytest.fixture
def mock_repository() -> AsyncMock:
    repository = AsyncMock()
    repository.get_terminating_sessions_by_ids = AsyncMock(return_value=[])
    return repository


@pytest.fixture
def mock_valkey_schedule() -> AsyncMock:
    valkey = AsyncMock()
    valkey.get_force_terminated_sessions = AsyncMock(return_value=[])
    valkey.remove_force_terminated_sessions = AsyncMock(return_value=None)
    return valkey


@pytest.fixture
def handler(
    mock_terminator: AsyncMock,
    mock_repository: AsyncMock,
    mock_valkey_schedule: AsyncMock,
) -> CleanupForceTerminatedHandler:
    return CleanupForceTerminatedHandler(
        terminator=mock_terminator,
        repository=mock_repository,
        valkey_schedule=mock_valkey_schedule,
    )


def _make_terminating_session_data(session_id: SessionId) -> TerminatingSessionData:
    return TerminatingSessionData(
        session_id=session_id,
        access_key=AccessKey("test-access-key"),
        creation_id="test-creation-id",
        status=SessionStatus.TERMINATED,
        status_info="FORCE_TERMINATED",
        session_type=SessionTypes.INTERACTIVE,
        kernels=[
            TerminatingKernelData(
                kernel_id=KernelId(uuid4()),
                status=KernelStatus.TERMINATED,
                container_id="container-1",
                agent_id=AgentId("agent-1"),
                agent_addr="tcp://agent-1:6001",
                occupied_slots=ResourceSlot({}),
            ),
        ],
    )


class TestCleanupForceTerminatedHandler:
    def test_name(self) -> None:
        assert CleanupForceTerminatedHandler.name() == "cleanup-force-terminated"

    async def test_fetch_session_ids_delegates_to_valkey(
        self,
        handler: CleanupForceTerminatedHandler,
        mock_valkey_schedule: AsyncMock,
    ) -> None:
        session_ids = [SessionId(uuid4())]
        mock_valkey_schedule.get_force_terminated_sessions.return_value = session_ids

        result = await handler.fetch_session_ids()

        assert list(result) == session_ids

    async def test_execute_sends_rpc_and_removes_succeeded(
        self,
        handler: CleanupForceTerminatedHandler,
        mock_terminator: AsyncMock,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
    ) -> None:
        session_id = SessionId(uuid4())
        session_data = _make_terminating_session_data(session_id)
        mock_repository.get_terminating_sessions_by_ids.return_value = [session_data]
        mock_terminator.terminate_sessions_for_handler.return_value = [session_id]

        await handler.execute([session_id])

        mock_terminator.terminate_sessions_for_handler.assert_awaited_once_with([session_data])
        mock_valkey_schedule.remove_force_terminated_sessions.assert_awaited_once_with([session_id])

    async def test_execute_no_db_data_removes_stale_ids(
        self,
        handler: CleanupForceTerminatedHandler,
        mock_terminator: AsyncMock,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
    ) -> None:
        """Sessions no longer in DB are removed from Valkey to avoid infinite retry."""
        session_id = SessionId(uuid4())
        mock_repository.get_terminating_sessions_by_ids.return_value = []

        await handler.execute([session_id])

        mock_terminator.terminate_sessions_for_handler.assert_not_awaited()
        mock_valkey_schedule.remove_force_terminated_sessions.assert_awaited_once_with([session_id])

    async def test_execute_partial_failure_removes_only_succeeded(
        self,
        handler: CleanupForceTerminatedHandler,
        mock_terminator: AsyncMock,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
    ) -> None:
        """Only successfully cleaned sessions are removed from Valkey."""
        sid_ok = SessionId(uuid4())
        sid_fail = SessionId(uuid4())
        data_ok = _make_terminating_session_data(sid_ok)
        data_fail = _make_terminating_session_data(sid_fail)
        mock_repository.get_terminating_sessions_by_ids.return_value = [data_ok, data_fail]

        # The terminator reports only the first session as fully cleaned up
        mock_terminator.terminate_sessions_for_handler.return_value = [sid_ok]

        await handler.execute([sid_ok, sid_fail])

        # All sessions go out in one call; only the succeeded session ID is removed
        mock_terminator.terminate_sessions_for_handler.assert_awaited_once_with([
            data_ok,
            data_fail,
        ])
        mock_valkey_schedule.remove_force_terminated_sessions.assert_awaited_once_with([sid_ok])

    async def test_execute_all_fail_removes_nothing(
        self,
        handler: CleanupForceTerminatedHandler,
        mock_terminator: AsyncMock,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
    ) -> None:
        session_id = SessionId(uuid4())
        session_data = _make_terminating_session_data(session_id)
        mock_repository.get_terminating_sessions_by_ids.return_value = [session_data]
        mock_terminator.terminate_sessions_for_handler.side_effect = RuntimeError("Agent down")

        await handler.execute([session_id])

        mock_valkey_schedule.remove_force_terminated_sessions.assert_not_awaited()

    async def test_execute_none_succeeded_removes_nothing(
        self,
        handler: CleanupForceTerminatedHandler,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
    ) -> None:
        session_id = SessionId(uuid4())
        mock_repository.get_terminating_sessions_by_ids.return_value = [
            _make_terminating_session_data(session_id)
        ]

        await handler.execute([session_id])

        mock_valkey_schedule.remove_force_terminated_sessions.assert_not_awaited()


HEALTHY_AGENT = AgentId("agent-healthy")
HUNG_AGENT_COUNT = 10
RPC_TIMEOUT = 0.1


def _hung_agent(index: int) -> AgentId:
    return AgentId(f"agent-hung-{index}")


async def _hang(*args: Any, **kwargs: Any) -> Any:
    await asyncio.Event().wait()


def _make_session_on(agent_id: AgentId) -> TerminatingSessionData:
    session = _make_terminating_session_data(SessionId(uuid4()))
    session.kernels[0].agent_id = agent_id
    return session


class TestCleanupForceTerminatedWithUnresponsiveAgents:
    """The handler with a real SessionTerminator against agents that never reply."""

    @pytest.fixture
    def real_handler(
        self,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> CleanupForceTerminatedHandler:
        monkeypatch.setattr(terminator_module, "AGENT_DESTROY_KERNEL_TIMEOUT_SEC", RPC_TIMEOUT)
        per_agent_client_pool.client(HEALTHY_AGENT).destroy_kernel.return_value = None
        for i in range(HUNG_AGENT_COUNT):
            hung_client = per_agent_client_pool.client(_hung_agent(i))
            hung_client.destroy_kernel.side_effect = _hang
        terminator = SessionTerminator(
            SessionTerminatorArgs(
                repository=AsyncMock(),
                agent_client_pool=per_agent_client_pool,
                valkey_schedule=mock_valkey_schedule,
            )
        )
        return CleanupForceTerminatedHandler(
            terminator=terminator,
            repository=mock_repository,
            valkey_schedule=mock_valkey_schedule,
        )

    async def test_only_succeeded_sessions_are_removed(
        self,
        real_handler: CleanupForceTerminatedHandler,
        per_agent_client_pool: MagicMock,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
    ) -> None:
        hung = _make_session_on(_hung_agent(0))
        healthy = _make_session_on(HEALTHY_AGENT)
        mock_repository.get_terminating_sessions_by_ids.return_value = [hung, healthy]

        session_ids = [hung.session_id, healthy.session_id]
        with RecorderContext[SessionId].scope("cleanup", entity_ids=session_ids):
            await real_handler.execute(session_ids)

        per_agent_client_pool.client(HEALTHY_AGENT).destroy_kernel.assert_awaited_once()
        mock_valkey_schedule.remove_force_terminated_sessions.assert_awaited_once_with([
            healthy.session_id
        ])

    async def test_hung_agents_do_not_serialise_cleanup(
        self,
        real_handler: CleanupForceTerminatedHandler,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
    ) -> None:
        sessions = [_make_session_on(_hung_agent(i)) for i in range(HUNG_AGENT_COUNT)]
        mock_repository.get_terminating_sessions_by_ids.return_value = sessions

        session_ids = [s.session_id for s in sessions]
        started = time.monotonic()
        with RecorderContext[SessionId].scope("cleanup", entity_ids=session_ids):
            await real_handler.execute(session_ids)
        elapsed = time.monotonic() - started

        # One session after another would take 10 * RPC_TIMEOUT = 1.0s.
        assert elapsed < 0.6
        mock_valkey_schedule.remove_force_terminated_sessions.assert_not_awaited()
