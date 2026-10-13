"""Handler for starting sessions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, override

from ai.backend.common.identifier.resource_group import ResourceGroupID
from ai.backend.common.types import AccessKey, SessionId
from ai.backend.logging import BraceStyleAdapter
from ai.backend.manager.data.kernel.types import KernelStatus
from ai.backend.manager.data.session.types import SessionStatus, StatusTransitions, TransitionStatus
from ai.backend.manager.defs import LockID
from ai.backend.manager.models.session.conditions import SessionConditions
from ai.backend.manager.repositories.base import BatchQuerier, NoPagination
from ai.backend.manager.repositories.scheduler import SchedulerRepository
from ai.backend.manager.sokovan.scheduler.handlers.base import SessionLifecycleHandler
from ai.backend.manager.sokovan.scheduler.results import (
    SessionExecutionResult,
    SessionTransitionInfo,
)
from ai.backend.manager.views.sokovan.lifecycle import SessionDataForStart, SessionWithKernels

if TYPE_CHECKING:
    from ai.backend.manager.sokovan.scheduler.launcher.launcher import SessionLauncher

log = BraceStyleAdapter(logging.getLogger(__name__))

# Kernel statuses up to PREPARED: a kernel in any of them has not been sent to
# an agent for creation yet. A kernel without a status counts as PREPARED.
_NOT_YET_STARTED_STATUSES: frozenset[KernelStatus] = KernelStatus.pre_prepared_statuses() | {
    KernelStatus.PREPARED
}


class StartSessionsLifecycleHandler(SessionLifecycleHandler):
    """Handler for starting sessions that passed precondition checks.

    Following the DeploymentCoordinator pattern:
    - Coordinator queries sessions with PREPARED status (provides HandlerSessionData)
    - Handler queries additional data (SessionDataForStart + ImageConfigData) via Repository
    - Handler starts kernels on agents via Launcher
    - Coordinator updates sessions to CREATING status and broadcasts events
    """

    def __init__(
        self,
        launcher: SessionLauncher,
        repository: SchedulerRepository,
    ) -> None:
        self._launcher = launcher
        self._repository = repository

    @classmethod
    @override
    def name(cls) -> str:
        """Get the name of the handler."""
        return "start-sessions"

    @classmethod
    @override
    def target_statuses(cls) -> list[SessionStatus]:
        """Sessions in PREPARED state."""
        return [SessionStatus.PREPARED]

    @classmethod
    @override
    def target_kernel_statuses(cls) -> list[KernelStatus] | None:
        """Include sessions where kernels are in PREPARED status."""
        return [KernelStatus.PREPARED]

    @classmethod
    @override
    def status_transitions(cls) -> StatusTransitions:
        """Define state transitions for start sessions handler (BEP-1030).

        - success: Session/kernel → CREATING
        - need_retry: None (stays PREPARED)
        - expired: Session/kernel → PENDING (re-scheduling after timeout —
          a different agent or a later slot may succeed)
        - give_up: Session/kernel → TERMINATING (retry budget exhausted —
          repeatedly failing to start a kernel signals a persistent
          problem with the kernel spec or environment; we surface the
          failure rather than rescheduling indefinitely)
        """
        return StatusTransitions(
            success=TransitionStatus(
                session=SessionStatus.CREATING,
                kernel=KernelStatus.CREATING,
            ),
            need_retry=None,
            expired=TransitionStatus(
                session=SessionStatus.PENDING,
                kernel=KernelStatus.PENDING,
            ),
            give_up=TransitionStatus(
                session=SessionStatus.TERMINATING,
                kernel=KernelStatus.TERMINATING,
            ),
        )

    @property
    @override
    def lock_id(self) -> LockID | None:
        """Lock for operations targeting PREPARED sessions transitioning to CREATING."""
        return LockID.LOCKID_SOKOVAN_TARGET_CREATING

    @override
    async def execute(
        self,
        _resource_group_id: ResourceGroupID,
        sessions: Sequence[SessionWithKernels],
    ) -> SessionExecutionResult:
        """Start kernels on agents for PREPARED sessions.

        The coordinator provides SessionWithKernels data.
        This handler:
        1. Extracts session IDs from SessionWithKernels
        2. Queries Repository for additional data (SessionDataForStart + ImageConfigData),
           keeping only the sessions that are still PREPARED
        3. Skips a session any of whose kernels is already past PREPARED
        4. Starts kernels on agents via Launcher
        """
        result = SessionExecutionResult()

        if not sessions:
            return result

        # Extract session IDs from SessionWithKernels
        session_ids = [s.session_info.identity.id for s in sessions]

        # Query Repository for additional data needed by Launcher
        # Use search_sessions_with_kernels_and_user to get user info for session start.
        # A session that left PREPARED since the coordinator read it (for example,
        # one being terminated) is neither started nor reported.
        querier = BatchQuerier(
            pagination=NoPagination(),
            conditions=[
                SessionConditions.by_ids(session_ids),
                SessionConditions.by_statuses([SessionStatus.PREPARED]),
            ],
        )
        sessions_data = await self._repository.search_sessions_with_kernels_and_user(querier)

        # A session with a kernel already past PREPARED was sent before. Sending it
        # again could re-create a kernel that is running or build a split cluster,
        # so it is skipped and left for the session expiry to end.
        to_start: list[SessionDataForStart] = []
        already_sent_ids: set[SessionId] = set()
        for session_data in sessions_data.sessions:
            started_kernels = [
                kernel
                for kernel in session_data.kernels
                if kernel.status is not None and kernel.status not in _NOT_YET_STARTED_STATUSES
            ]
            if started_kernels:
                already_sent_ids.add(session_data.session_id)
                log.warning(
                    "start-session(s:{}): not sent again, {} of its kernels are past PREPARED ({})",
                    session_data.session_id,
                    len(started_kernels),
                    ", ".join(sorted({str(kernel.status) for kernel in started_kernels})),
                )
                continue
            to_start.append(session_data)
        started_ids = {session_data.session_id for session_data in to_start}

        # Start kernels on agents via Launcher
        # Note: RecorderContext is handled inside Launcher
        await self._launcher.start_sessions_for_handler(
            to_start,
            sessions_data.image_configs,
        )

        # Mark the started sessions as success for status transition
        for session in sessions:
            session_info = session.session_info
            session_id = session_info.identity.id
            if session_id in already_sent_ids:
                result.skipped.append(
                    SessionTransitionInfo(
                        session_id=session_id,
                        from_status=session_info.lifecycle.status,
                        reason="kernels-already-started",
                        creation_id=session_info.identity.creation_id,
                        access_key=AccessKey(session_info.metadata.access_key),
                    )
                )
                continue
            if session_id not in started_ids:
                continue
            result.successes.append(
                SessionTransitionInfo(
                    session_id=session_id,
                    from_status=session_info.lifecycle.status,
                    reason="triggered-by-scheduler",
                    creation_id=session_info.identity.creation_id,
                    access_key=AccessKey(session_info.metadata.access_key),
                )
            )

        return result
