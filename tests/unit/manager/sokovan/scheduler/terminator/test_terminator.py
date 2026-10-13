"""Unit tests for Sokovan scheduler SessionTerminator.

Based on BEP-1033 test scenarios for terminator testing.

Test Scenarios:
- SC-TE-001 ~ SC-TE-007: Session Termination
- SC-TE-008 ~ SC-TE-014: Stale Kernel Detection
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from ai.backend.common.types import AgentId, KernelId, SessionId
from ai.backend.manager.data.kernel.types import KernelInfo, KernelStatus
from ai.backend.manager.data.session.types import SessionStatus
from ai.backend.manager.sokovan.recorder import RecorderContext
from ai.backend.manager.sokovan.scheduler.terminator.terminator import SessionTerminator
from ai.backend.manager.sokovan.scheduler.types import ScheduleType
from ai.backend.manager.views.sokovan.session import TerminatingKernelData, TerminatingSessionData

# =============================================================================
# TestSessionTerminatorTermination (SC-TE-001 ~ SC-TE-007)
# =============================================================================


class TestSessionTerminatorTermination:
    """Tests for session termination functionality.

    Verifies the terminator correctly sends termination requests to agents.
    """

    async def test_all_kernels_terminated_successfully(
        self,
        terminator: SessionTerminator,
        mock_agent_client_pool: MagicMock,
        terminating_session_single_kernel: TerminatingSessionData,
    ) -> None:
        """SC-TE-001: All kernels terminated successfully.

        Given: Session with one kernel
        When: Terminate session
        Then: destroy_kernel called for the kernel
        """
        session_ids = [terminating_session_single_kernel.session_id]
        with RecorderContext.scope("test", entity_ids=session_ids):
            await terminator.terminate_sessions_for_handler([terminating_session_single_kernel])

        # Assert
        mock_client = mock_agent_client_pool._mock_client
        mock_client.destroy_kernel.assert_awaited_once()

        # Verify correct kernel was terminated
        call_args = mock_client.destroy_kernel.call_args
        kernel = terminating_session_single_kernel.kernels[0]
        assert call_args[0][0] == kernel.kernel_id

    async def test_kernel_without_agent_skipped(
        self,
        terminator: SessionTerminator,
        mock_agent_client_pool: MagicMock,
        terminating_session_kernel_no_agent: TerminatingSessionData,
    ) -> None:
        """SC-TE-002: Kernel without agent is skipped.

        Given: Session with kernel that has no agent
        When: Terminate session
        Then: destroy_kernel not called (kernel skipped)
        """
        session_ids = [terminating_session_kernel_no_agent.session_id]
        with RecorderContext.scope("test", entity_ids=session_ids):
            await terminator.terminate_sessions_for_handler([terminating_session_kernel_no_agent])

        # Assert - destroy_kernel should not be called
        mock_client = mock_agent_client_pool._mock_client
        mock_client.destroy_kernel.assert_not_awaited()

    async def test_empty_session_list_returns_immediately(
        self,
        terminator: SessionTerminator,
        mock_agent_client_pool: MagicMock,
    ) -> None:
        """SC-TE-003: Empty session list returns immediately.

        Given: Empty session list
        When: Terminate sessions
        Then: No agent calls made
        """
        with RecorderContext.scope("test", entity_ids=[]):
            await terminator.terminate_sessions_for_handler([])

        # Assert
        mock_client = mock_agent_client_pool._mock_client
        mock_client.destroy_kernel.assert_not_awaited()

    async def test_partial_termination_failure_logged(
        self,
        terminator: SessionTerminator,
        mock_agent_client_pool: MagicMock,
        terminating_session_multi_kernel: TerminatingSessionData,
    ) -> None:
        """SC-TE-004: Partial termination failure is logged.

        Given: Session with multiple kernels, one fails
        When: Terminate session
        Then: Both kernels attempted, failure logged (not raised)
        """
        # Arrange - First call fails
        mock_client = mock_agent_client_pool._mock_client
        mock_client.destroy_kernel.side_effect = [
            RuntimeError("Agent 1 unreachable"),
            None,  # Success
        ]

        session_ids = [terminating_session_multi_kernel.session_id]
        with RecorderContext.scope("test", entity_ids=session_ids):
            # Act - Should not raise
            await terminator.terminate_sessions_for_handler([terminating_session_multi_kernel])

        # Assert - Both kernels attempted
        assert mock_client.destroy_kernel.await_count == 2

    async def test_multiple_sessions_terminated_parallel(
        self,
        terminator: SessionTerminator,
        mock_agent_client_pool: MagicMock,
        terminating_sessions_multiple: list[TerminatingSessionData],
    ) -> None:
        """SC-TE-005: Multiple sessions terminated in parallel.

        Given: Multiple sessions to terminate
        When: Terminate sessions
        Then: All kernels terminated concurrently
        """
        session_ids = [s.session_id for s in terminating_sessions_multiple]
        with RecorderContext.scope("test", entity_ids=session_ids):
            await terminator.terminate_sessions_for_handler(terminating_sessions_multiple)

        # Assert - destroy_kernel called for each session's kernels
        mock_client = mock_agent_client_pool._mock_client
        assert mock_client.destroy_kernel.await_count == 2

    async def test_termination_passes_correct_parameters(
        self,
        terminator: SessionTerminator,
        mock_agent_client_pool: MagicMock,
        terminating_session_single_kernel: TerminatingSessionData,
    ) -> None:
        """SC-TE-006: Termination passes correct parameters to agent.

        Given: Session with specific termination reason
        When: Terminate session
        Then: destroy_kernel called with correct kernel_id, session_id, reason
        """
        session_ids = [terminating_session_single_kernel.session_id]
        with RecorderContext.scope("test", entity_ids=session_ids):
            await terminator.terminate_sessions_for_handler([terminating_session_single_kernel])

        # Assert - Verify call parameters
        mock_client = mock_agent_client_pool._mock_client
        call_args = mock_client.destroy_kernel.call_args
        kernel = terminating_session_single_kernel.kernels[0]

        # destroy_kernel(kernel_id, session_id, reason, suppress_events=False)
        assert call_args[0][0] == kernel.kernel_id
        assert call_args[0][1] == terminating_session_single_kernel.session_id
        assert call_args[0][2] == terminating_session_single_kernel.status_info

    async def test_mixed_agent_assignment_handles_both(
        self,
        terminator: SessionTerminator,
        mock_agent_client_pool: MagicMock,
        terminating_session_mixed_agents: TerminatingSessionData,
    ) -> None:
        """SC-TE-007: Mixed agent assignment handled correctly.

        Given: Session with some kernels having agents, some without
        When: Terminate session
        Then: Only kernels with agents are terminated
        """
        session_ids = [terminating_session_mixed_agents.session_id]
        with RecorderContext.scope("test", entity_ids=session_ids):
            await terminator.terminate_sessions_for_handler([terminating_session_mixed_agents])

        # Assert - Only kernel with agent was terminated
        mock_client = mock_agent_client_pool._mock_client
        mock_client.destroy_kernel.assert_awaited_once()

        # Verify correct kernel (the one with agent) was terminated
        call_args = mock_client.destroy_kernel.call_args
        kernel_with_agent = terminating_session_mixed_agents.kernels[0]  # First kernel has agent
        assert call_args[0][0] == kernel_with_agent.kernel_id


class TestCancelUnplacedKernels:
    """TERMINATE cancels the PENDING kernels that were never placed on an agent."""

    async def test_only_unplaced_pending_kernels_of_terminating_sessions_are_cancelled(
        self,
        terminator: SessionTerminator,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
        terminating_kernel_factory: Callable[..., TerminatingKernelData],
        terminating_session_factory: Callable[..., TerminatingSessionData],
    ) -> None:
        unplaced = terminating_kernel_factory()
        unplaced.agent_id = None
        unplaced.status = KernelStatus.PENDING
        unplaced_not_pending = terminating_kernel_factory()
        unplaced_not_pending.agent_id = None
        unplaced_not_pending.status = KernelStatus.CANCELLED
        placed = terminating_kernel_factory()
        placed.status = KernelStatus.PENDING
        session = terminating_session_factory(
            kernels=[unplaced, unplaced_not_pending, placed], status_info="user-requested"
        )
        mock_repository.update_kernel_status_cancelled = AsyncMock(return_value=True)

        cancelled = await terminator.cancel_unplaced_kernels([session])

        assert cancelled == [unplaced.kernel_id]
        mock_repository.update_kernel_status_cancelled.assert_awaited_once_with(
            unplaced.kernel_id, "user-requested"
        )
        mock_valkey_schedule.mark_schedules_needed_batch.assert_awaited_once_with([
            ScheduleType.CHECK_TERMINATING_PROGRESS.value
        ])

    async def test_a_session_that_is_not_terminating_is_left_alone(
        self,
        terminator: SessionTerminator,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
        terminating_kernel_factory: Callable[..., TerminatingKernelData],
        terminating_session_factory: Callable[..., TerminatingSessionData],
    ) -> None:
        unplaced = terminating_kernel_factory()
        unplaced.agent_id = None
        unplaced.status = KernelStatus.PENDING
        session = terminating_session_factory(kernels=[unplaced])
        session.status = SessionStatus.RESCHEDULING

        assert await terminator.cancel_unplaced_kernels([session]) == []
        mock_repository.update_kernel_status_cancelled.assert_not_awaited()
        mock_valkey_schedule.mark_schedules_needed_batch.assert_not_awaited()

    async def test_a_kernel_the_guard_refused_is_not_reported(
        self,
        terminator: SessionTerminator,
        mock_repository: AsyncMock,
        mock_valkey_schedule: AsyncMock,
        terminating_kernel_factory: Callable[..., TerminatingKernelData],
        terminating_session_factory: Callable[..., TerminatingSessionData],
    ) -> None:
        unplaced = terminating_kernel_factory()
        unplaced.agent_id = None
        unplaced.status = KernelStatus.PENDING
        mock_repository.update_kernel_status_cancelled = AsyncMock(return_value=False)

        assert (
            await terminator.cancel_unplaced_kernels([
                terminating_session_factory(kernels=[unplaced])
            ])
            == []
        )
        mock_valkey_schedule.mark_schedules_needed_batch.assert_not_awaited()


class TestDestroyKernel:
    async def test_destroy_reaches_the_agent(
        self,
        terminator: SessionTerminator,
        mock_agent_client_pool: MagicMock,
    ) -> None:
        kernel_id = KernelId(uuid4())
        session_id = SessionId(uuid4())

        result = await terminator.destroy_kernel(
            AgentId("agent-1"), kernel_id, session_id, "already-terminated"
        )

        assert result.success
        mock_agent_client_pool.acquire.assert_called_once_with(AgentId("agent-1"))
        mock_agent_client_pool._mock_client.destroy_kernel.assert_awaited_once_with(
            kernel_id, session_id, "already-terminated", suppress_events=False
        )

    async def test_a_failed_destroy_is_returned_not_raised(
        self,
        terminator: SessionTerminator,
        mock_agent_client_pool: MagicMock,
    ) -> None:
        mock_agent_client_pool._mock_client.destroy_kernel.side_effect = ConnectionError(
            "agent unreachable"
        )

        result = await terminator.destroy_kernel(
            AgentId("agent-1"), KernelId(uuid4()), SessionId(uuid4()), "already-terminated"
        )

        assert not result.success
        assert result.error == "agent unreachable"


# =============================================================================
# TestSessionTerminatorStaleDetection (SC-TE-008 ~ SC-TE-014)
# =============================================================================


class TestSessionTerminatorStaleDetection:
    """Tests for stale kernel detection functionality.

    Verifies the terminator correctly identifies dead kernels via Valkey and agent checks.
    """

    async def test_no_stale_kernels_all_alive(
        self,
        terminator: SessionTerminator,
        mock_valkey_schedule: AsyncMock,
        mock_agent_client_pool: MagicMock,
        running_kernels_multiple: list[KernelInfo],
        valkey_all_alive_response: dict[KernelId, MagicMock],
    ) -> None:
        """SC-TE-008: No stale kernels when all are alive.

        Given: Multiple RUNNING kernels
        When: Valkey shows all as ALIVE
        Then: Empty list returned (no stale kernels)
        """
        # Arrange
        mock_valkey_schedule.check_kernel_presence_status_batch.return_value = (
            valkey_all_alive_response
        )

        # Act
        result = await terminator.check_stale_kernels(running_kernels_multiple)

        # Assert
        assert len(result) == 0
        # Agent check should not be called since all are alive
        mock_client = mock_agent_client_pool._mock_client
        mock_client.check_running.assert_not_awaited()

    async def test_stale_from_valkey_agent_confirms_dead(
        self,
        terminator: SessionTerminator,
        mock_valkey_schedule: AsyncMock,
        mock_agent_client_pool: MagicMock,
        running_kernels_multiple: list[KernelInfo],
        valkey_one_stale_response: dict[KernelId, MagicMock],
    ) -> None:
        """SC-TE-009: Stale from Valkey confirmed dead by agent.

        Given: One kernel marked STALE in Valkey
        When: Agent confirms kernel is not running
        Then: Kernel ID returned in stale list
        """
        # Arrange
        mock_valkey_schedule.check_kernel_presence_status_batch.return_value = (
            valkey_one_stale_response
        )
        mock_client = mock_agent_client_pool._mock_client
        mock_client.check_running.return_value = False  # Confirmed dead

        # Act
        result = await terminator.check_stale_kernels(running_kernels_multiple)

        # Assert
        assert len(result) == 1
        assert result[0] == KernelId(running_kernels_multiple[0].id)

    async def test_stale_from_valkey_but_agent_shows_alive(
        self,
        terminator: SessionTerminator,
        mock_valkey_schedule: AsyncMock,
        mock_agent_client_pool: MagicMock,
        running_kernels_multiple: list[KernelInfo],
        valkey_one_stale_response: dict[KernelId, MagicMock],
    ) -> None:
        """SC-TE-010: Stale from Valkey but agent shows alive.

        Given: One kernel marked STALE in Valkey
        When: Agent confirms kernel IS running
        Then: Kernel NOT returned in stale list (false positive)
        """
        # Arrange
        mock_valkey_schedule.check_kernel_presence_status_batch.return_value = (
            valkey_one_stale_response
        )
        mock_client = mock_agent_client_pool._mock_client
        mock_client.check_running.return_value = True  # Still alive

        # Act
        result = await terminator.check_stale_kernels(running_kernels_multiple)

        # Assert - Not marked dead because agent says it's running
        assert len(result) == 0

    async def test_no_presence_info_treated_as_stale(
        self,
        terminator: SessionTerminator,
        mock_valkey_schedule: AsyncMock,
        mock_agent_client_pool: MagicMock,
        running_kernel: KernelInfo,
    ) -> None:
        """SC-TE-011: No presence info treated as potentially stale.

        Given: Kernel with no Valkey presence record
        When: Agent confirms kernel is not running
        Then: Kernel returned in stale list
        """
        # Arrange - Empty response means no presence info
        mock_valkey_schedule.check_kernel_presence_status_batch.return_value = {}
        mock_client = mock_agent_client_pool._mock_client
        mock_client.check_running.return_value = False

        # Act
        result = await terminator.check_stale_kernels([running_kernel])

        # Assert
        assert len(result) == 1
        assert result[0] == KernelId(running_kernel.id)

    async def test_agent_verification_fails_kernel_skipped(
        self,
        terminator: SessionTerminator,
        mock_valkey_schedule: AsyncMock,
        mock_agent_client_pool: MagicMock,
        running_kernel: KernelInfo,
    ) -> None:
        """SC-TE-012: Agent verification failure skips kernel.

        Given: Kernel marked STALE in Valkey
        When: Agent check raises exception
        Then: Kernel skipped (not marked dead)
        """
        # Arrange
        mock_status = MagicMock()
        mock_status.presence = MagicMock()  # STALE by treating None as stale
        mock_valkey_schedule.check_kernel_presence_status_batch.return_value = {}

        mock_client = mock_agent_client_pool._mock_client
        mock_client.check_running.side_effect = RuntimeError("Agent unreachable")

        # Act
        result = await terminator.check_stale_kernels([running_kernel])

        # Assert - Kernel skipped due to verification failure
        assert len(result) == 0

    async def test_kernel_without_agent_skipped_in_stale_check(
        self,
        terminator: SessionTerminator,
        mock_valkey_schedule: AsyncMock,
        mock_agent_client_pool: MagicMock,
        running_kernel_no_agent: KernelInfo,
    ) -> None:
        """SC-TE-013: Kernel without agent skipped in stale check.

        Given: Kernel with no agent assigned
        When: Check for stale kernels
        Then: Kernel skipped (can't verify without agent)
        """
        # Arrange - No presence info to trigger stale check
        mock_valkey_schedule.check_kernel_presence_status_batch.return_value = {}

        # Act
        result = await terminator.check_stale_kernels([running_kernel_no_agent])

        # Assert - Kernel skipped
        assert len(result) == 0
        # Agent check should not be called
        mock_client = mock_agent_client_pool._mock_client
        mock_client.check_running.assert_not_awaited()

    async def test_presence_read_with_agents_from_kernel_records(
        self,
        terminator: SessionTerminator,
        mock_valkey_schedule: AsyncMock,
        running_kernels_multiple: list[KernelInfo],
        running_kernel_no_agent: KernelInfo,
    ) -> None:
        """Presence keys are built from each kernel's recorded agent; kernels without
        an agent are not looked up."""
        mock_valkey_schedule.check_kernel_presence_status_batch.return_value = {}

        await terminator.check_stale_kernels([*running_kernels_multiple, running_kernel_no_agent])

        expected = {
            KernelId(k.id): AgentId(k.resource.agent)
            for k in running_kernels_multiple
            if k.resource.agent
        }
        mock_valkey_schedule.check_kernel_presence_status_batch.assert_awaited_once_with(
            expected,
            agent_ids={AgentId("agent-1"), AgentId("agent-2")},
        )

    async def test_empty_kernel_list_returns_empty(
        self,
        terminator: SessionTerminator,
        mock_valkey_schedule: AsyncMock,
    ) -> None:
        """SC-TE-014: Empty kernel list returns empty result.

        Given: Empty kernel list
        When: Check for stale kernels
        Then: Empty list returned without calling Valkey
        """
        # Act
        result = await terminator.check_stale_kernels([])

        # Assert
        assert len(result) == 0
        mock_valkey_schedule.check_kernel_presence_status_batch.assert_not_awaited()
