"""Handler for expiring sessions stuck while starting."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Final, override

from ai.backend.common.events.event_types.kernel.types import KernelLifecycleEventReason
from ai.backend.common.identifier.resource_group import ResourceGroupID
from ai.backend.common.types import AccessKey
from ai.backend.logging import BraceStyleAdapter
from ai.backend.manager.data.kernel.types import KernelStatus
from ai.backend.manager.data.session.types import SessionStatus, StatusTransitions, TransitionStatus
from ai.backend.manager.defs import (
    SESSION_CREATING_EXPIRY_SEC,
    SESSION_PULLING_EXPIRY_SEC,
    LockID,
)
from ai.backend.manager.repositories.scheduler.repository import SchedulerRepository
from ai.backend.manager.sokovan.scheduler.handlers.base import SessionLifecycleHandler
from ai.backend.manager.sokovan.scheduler.results import (
    SessionExecutionResult,
    SessionTransitionInfo,
)
from ai.backend.manager.views.sokovan.lifecycle import SessionWithKernels

log = BraceStyleAdapter(logging.getLogger(__name__))

# The session's status_info once it expires. TERMINATE sends it to the agents as the destroy
# reason, so it is a kernel lifecycle event reason.
EXPIRY_REASON: Final[str] = KernelLifecycleEventReason.BOOTSTRAP_TIMEOUT.value

_STARTING_STATUSES: Final[tuple[SessionStatus, ...]] = (
    SessionStatus.SCHEDULED,
    SessionStatus.PREPARING,
    SessionStatus.PULLING,
    SessionStatus.PREPARED,
    SessionStatus.CREATING,
)


class ExpireStuckStartsLifecycleHandler(SessionLifecycleHandler):
    """Moves sessions stuck while starting to TERMINATING (maintenance operation).

    Agents report a start's progress through events, and some failures report nothing: an
    agent that drops a request, a create that fails before its first event, a session the
    launcher never dispatched. Without an expiry such a session holds its resources for good.

    A session expires when its age in its current status (its ``status_history`` entry for that
    status, on the database clock) is past the bound for the status:

    - SCHEDULED, PREPARED: ``SESSION_CREATING_EXPIRY_SEC``, whatever its kernels are doing.
    - CREATING: ``SESSION_CREATING_EXPIRY_SEC``, while some kernel has not started yet.
    - PREPARING, PULLING: ``SESSION_PULLING_EXPIRY_SEC`` while some kernel is PULLING,
      otherwise ``SESSION_CREATING_EXPIRY_SEC``, while some kernel has not finished preparing.

    Expired sessions are reported as successes; the coordinator moves them to TERMINATING with
    ``EXPIRY_REASON`` as their status_info, and TERMINATE then destroys their kernels and frees
    their resources. The transition moves only sessions still in the status they were read in,
    so the handler needs no lock and concurrent runs are safe.

    ``execute`` dispatches each session to the branch for its status family. Another family
    (for example TERMINATING) joins by adding its statuses to ``target_statuses`` and a branch
    to ``execute``; a branch that releases sessions by its own means reports nothing in the
    result.
    """

    def __init__(
        self,
        repository: SchedulerRepository,
    ) -> None:
        self._repository = repository

    @classmethod
    @override
    def name(cls) -> str:
        """Get the name of the handler."""
        return "expire-stuck-starts"

    @classmethod
    @override
    def target_statuses(cls) -> list[SessionStatus]:
        """Sessions that are starting."""
        return list(_STARTING_STATUSES)

    @classmethod
    @override
    def target_kernel_statuses(cls) -> list[KernelStatus] | None:
        """No kernel filtering, so sessions whose kernels are all PENDING are seen too."""
        return None

    @classmethod
    @override
    def status_transitions(cls) -> StatusTransitions:
        """Define state transitions for the stuck-start expiry (BEP-1030).

        - success: Session → TERMINATING with ``EXPIRY_REASON`` (the start expired);
          kernels are left to TERMINATE
        - need_retry, expired, give_up: None (the handler reports no failures)
        """
        return StatusTransitions(
            success=TransitionStatus(
                session=SessionStatus.TERMINATING,
                kernel=None,
                status_info=EXPIRY_REASON,
            ),
            need_retry=None,
            expired=None,
            give_up=None,
        )

    @property
    @override
    def lock_id(self) -> LockID | None:
        """No lock: the transition is guarded by each session's status as read."""
        return None

    @override
    async def execute(
        self,
        resource_group_id: ResourceGroupID,
        sessions: Sequence[SessionWithKernels],
    ) -> SessionExecutionResult:
        """Report the sessions whose start has expired as successes."""
        result = SessionExecutionResult()
        starting = [s for s in sessions if s.session_info.lifecycle.status in _STARTING_STATUSES]
        if starting:
            now = await self._repository.get_db_now()
            result.successes.extend(self._expire_stuck_starts(resource_group_id, starting, now))
        return result

    def _expire_stuck_starts(
        self,
        resource_group_id: ResourceGroupID,
        sessions: Sequence[SessionWithKernels],
        now: datetime,
    ) -> list[SessionTransitionInfo]:
        expired: list[SessionTransitionInfo] = []
        for session in sessions:
            session_info = session.session_info
            status = session_info.lifecycle.status
            entered_at = _status_entered_at(session_info.lifecycle.status_history, status)
            if entered_at is None:
                log.warning(
                    "expire-stuck-starts: session {} in {} has no status_history entry for its"
                    " status; its start cannot expire",
                    session_info.identity.id,
                    status,
                )
                continue
            age = (now - entered_at).total_seconds()
            bound = expiry_bound(session)
            if bound is None or age <= bound:
                continue
            agents = sorted({
                str(k.resource.agent) for k in session.kernel_infos if k.resource.agent
            })
            log.warning(
                "expire-stuck-starts: session {} expired in {} after {:.0f}s (bound {:.0f}s,"
                " resource group {}, agents {})",
                session_info.identity.id,
                status,
                age,
                bound,
                resource_group_id,
                agents,
            )
            expired.append(
                SessionTransitionInfo(
                    session_id=session_info.identity.id,
                    from_status=status,
                    reason=EXPIRY_REASON,
                    creation_id=session_info.identity.creation_id,
                    access_key=AccessKey(session_info.metadata.access_key),
                )
            )
        return expired


def expiry_bound(session: SessionWithKernels) -> float | None:
    """Return the session's start bound in seconds, or None when it cannot be stuck starting."""
    status = session.session_info.lifecycle.status
    kernel_statuses = {k.lifecycle.status for k in session.kernel_infos}
    match status:
        case SessionStatus.SCHEDULED | SessionStatus.PREPARED:
            return SESSION_CREATING_EXPIRY_SEC
        case SessionStatus.CREATING:
            if kernel_statuses & KernelStatus.pre_running_statuses():
                return SESSION_CREATING_EXPIRY_SEC
            return None
        case SessionStatus.PREPARING | SessionStatus.PULLING:
            if not kernel_statuses & KernelStatus.pre_prepared_statuses():
                return None
            if KernelStatus.PULLING in kernel_statuses:
                return SESSION_PULLING_EXPIRY_SEC
            return SESSION_CREATING_EXPIRY_SEC
        case _:
            return None


def _status_entered_at(
    status_history: Mapping[str, object] | None,
    status: SessionStatus,
) -> datetime | None:
    if not isinstance(status_history, Mapping):
        return None
    value = status_history.get(status.name)
    if not isinstance(value, str):
        return None
    try:
        entered_at = datetime.fromisoformat(value)
    except ValueError:
        return None
    if entered_at.tzinfo is None:
        entered_at = entered_at.replace(tzinfo=UTC)
    return entered_at
