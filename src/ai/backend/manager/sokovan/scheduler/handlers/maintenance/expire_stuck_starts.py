"""Handler for expiring sessions stuck while starting."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, override

from ai.backend.common.events.event_types.kernel.types import KernelLifecycleEventReason
from ai.backend.common.identifier.resource_group import ResourceGroupID
from ai.backend.common.types import AccessKey, AgentId, SessionId
from ai.backend.logging import BraceStyleAdapter
from ai.backend.manager.data.kernel.types import KernelInfo, KernelStatus
from ai.backend.manager.data.session.types import SessionStatus, StatusTransitions, TransitionStatus
from ai.backend.manager.defs import (
    SESSION_CREATING_EXPIRY_SEC,
    SESSION_PULLING_EXPIRY_SEC,
    UNAVAILABLE_AGENT_RELEASE_SEC,
    LockID,
)
from ai.backend.manager.repositories.scheduler.repository import SchedulerRepository
from ai.backend.manager.sokovan.scheduler.handlers.base import SessionLifecycleHandler
from ai.backend.manager.sokovan.scheduler.results import (
    SessionExecutionResult,
    SessionTransitionInfo,
)
from ai.backend.manager.views.sokovan.lifecycle import SessionWithKernels

if TYPE_CHECKING:
    from ai.backend.manager.sokovan.scheduling_controller import SchedulingController

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

# The status_info of a session (and its kernels) released from an unavailable agent.
RELEASE_REASON: Final[str] = KernelLifecycleEventReason.HANG_TIMEOUT.value


class ExpireStuckStartsLifecycleHandler(SessionLifecycleHandler):
    """Moves sessions stuck while starting to TERMINATING (maintenance operation).

    Agents report a start's progress through events, and some failures report nothing: an
    agent that drops a request, a create that fails before its first event, a session the
    launcher never dispatched. Without an expiry such a session holds its resources for good.

    A session expires when its age in its current status (its ``status_history`` entry for that
    status, on the database clock) is past the bound for the status:

    - SCHEDULED, PREPARED: ``SESSION_CREATING_EXPIRY_SEC``, whatever its kernels are doing.
    - CREATING: ``SESSION_CREATING_EXPIRY_SEC``, while some kernel has not started yet.
    - PREPARING, PULLING: ``SESSION_PULLING_EXPIRY_SEC`` while some kernel is PULLING, or
      waits for a pull another session's kernel started of the same image on the same agent
      (the agent pulls an image once and reports the pull to the first session only);
      otherwise ``SESSION_CREATING_EXPIRY_SEC``, while some kernel has not finished preparing.

    Expired sessions are reported as successes; the coordinator moves them to TERMINATING with
    ``EXPIRY_REASON`` as their status_info, and TERMINATE then destroys their kernels and frees
    their resources. The transition moves only sessions still in the status they were read in,
    so the handler needs no lock and concurrent runs are safe.

    It also releases TERMINATING sessions held by an unavailable agent. TERMINATE cannot
    finish such a session: its destroy never reaches the agent, so the session keeps its
    resources until the agent answers. A TERMINATING session is released when:

    - its open (not terminal) kernels are not empty, all have an agent, and all sit on one
      agent;
    - that agent is LOST or TERMINATED, first seen so ``UNAVAILABLE_AGENT_RELEASE_SEC`` or
      more ago, and some agent of the same resource group is ALIVE (so a fault that marks
      every agent of the group unavailable at once releases nothing).

    The agent is fenced first (made unschedulable, committed): its containers may still run,
    so the resources the release frees must not be handed out on it again until an operator
    has reconciled it and re-enabled it. Only the sessions of agents the fence returned are
    then terminated by force (``RELEASE_REASON``): the session and its kernels become
    TERMINATED, their resources are freed, and their ids are queued for container cleanup,
    which keeps retrying the destroy while the agent stays fenced and away. This branch
    reports nothing in the result, and an error in it never affects the expiry's results.

    ``execute`` dispatches each session to the branch for its status family.
    """

    def __init__(
        self,
        repository: SchedulerRepository,
        scheduling_controller: SchedulingController,
    ) -> None:
        self._repository = repository
        self._scheduling_controller = scheduling_controller

    @classmethod
    @override
    def name(cls) -> str:
        """Get the name of the handler."""
        return "expire-stuck-starts"

    @classmethod
    @override
    def target_statuses(cls) -> list[SessionStatus]:
        """Sessions that are starting, and sessions that are terminating."""
        return [*_STARTING_STATUSES, SessionStatus.TERMINATING]

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
        """Report the sessions whose start has expired as successes, and release the
        terminating sessions held by an unavailable agent."""
        result = SessionExecutionResult()
        starting = [s for s in sessions if s.session_info.lifecycle.status in _STARTING_STATUSES]
        terminating = [
            s for s in sessions if s.session_info.lifecycle.status == SessionStatus.TERMINATING
        ]
        if starting:
            now = await self._repository.get_db_now()
            result.successes.extend(self._expire_stuck_starts(resource_group_id, starting, now))
        if terminating:
            try:
                await self._release_unavailable_agents(resource_group_id, terminating)
            except Exception:
                log.exception(
                    "expire-stuck-starts: failed to release terminating sessions held by"
                    " unavailable agents in resource group {}",
                    resource_group_id,
                )
        return result

    async def _release_unavailable_agents(
        self,
        resource_group_id: ResourceGroupID,
        sessions: Sequence[SessionWithKernels],
    ) -> None:
        candidates: dict[AgentId, list[SessionId]] = {}
        for session in sessions:
            agent_id = sole_agent(session)
            if agent_id is not None:
                candidates.setdefault(agent_id, []).append(session.session_info.identity.id)
        if not candidates:
            return
        # Fence first: the release frees resources on the agent, which must not be handed
        # out again while its containers may still run.
        fenced = await self._repository.fence_unavailable_agents(
            sorted(candidates), resource_group_id, UNAVAILABLE_AGENT_RELEASE_SEC
        )
        to_release: list[SessionId] = []
        for agent in fenced:
            session_ids = candidates.get(agent.agent_id, [])
            log.warning(
                "expire-stuck-starts: fenced agent {} ({} since {}, resource group {}):"
                " schedulable=false; force-releasing {} terminating sessions ({})."
                " Reconcile its containers before re-enabling it.",
                agent.agent_id,
                agent.status.name,
                agent.lost_at.isoformat(),
                resource_group_id,
                len(session_ids),
                RELEASE_REASON,
            )
            to_release.extend(session_ids)
        if not to_release:
            return
        released = await self._scheduling_controller.mark_sessions_for_termination(
            to_release, reason=RELEASE_REASON, forced=True
        )
        log.info(
            "expire-stuck-starts: released {} of {} terminating sessions on fenced agents"
            " (sessions: {})",
            len(released.force_terminated_sessions),
            len(to_release),
            [str(sid) for sid in released.force_terminated_sessions],
        )

    def _expire_stuck_starts(
        self,
        resource_group_id: ResourceGroupID,
        sessions: Sequence[SessionWithKernels],
        now: datetime,
    ) -> list[SessionTransitionInfo]:
        expired: list[SessionTransitionInfo] = []
        pulling = pulling_images(sessions)
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
            bound = expiry_bound(session, pulling)
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


def sole_agent(session: SessionWithKernels) -> AgentId | None:
    """Return the one agent all of the session's open kernels sit on, or None.

    None when the session has no open (not terminal) kernel, when an open kernel has no
    agent (for example a PENDING kernel), or when its open kernels sit on more than one agent.
    """
    open_kernels = [
        k
        for k in session.kernel_infos
        if k.lifecycle.status not in KernelStatus.terminal_statuses()
    ]
    if not open_kernels:
        return None
    agents = {k.resource.agent for k in open_kernels}
    if len(agents) != 1:
        return None
    (agent,) = agents
    if not agent:
        return None
    return AgentId(agent)


def _pull_key(kernel: KernelInfo) -> tuple[str, object] | None:
    """The (agent, image) a kernel's image pull runs as on its agent, or None if unknown."""
    image = kernel.image.image_id or kernel.image.identifier
    if not kernel.resource.agent or image is None:
        return None
    return kernel.resource.agent, image


def pulling_images(sessions: Sequence[SessionWithKernels]) -> frozenset[tuple[str, object]]:
    """The (agent, image) pairs the given sessions' PULLING kernels are pulling."""
    return frozenset(
        key
        for session in sessions
        for kernel in session.kernel_infos
        if kernel.lifecycle.status == KernelStatus.PULLING
        and (key := _pull_key(kernel)) is not None
    )


def expiry_bound(
    session: SessionWithKernels,
    pulling: frozenset[tuple[str, object]] = frozenset(),
) -> float | None:
    """Return the session's start bound in seconds, or None when it cannot be stuck starting.

    ``pulling`` holds the (agent, image) pairs other sessions' kernels are pulling: an agent
    pulls an image once, so a kernel waiting for the same image on the same agent never
    reaches PULLING itself, yet waits as long as the pull does.
    """
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
            waits_for_a_pull = bool(pulling) and any(
                k.lifecycle.status in KernelStatus.pre_prepared_statuses()
                and _pull_key(k) in pulling
                for k in session.kernel_infos
            )
            if KernelStatus.PULLING in kernel_statuses or waits_for_a_pull:
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
