"""Handler for cleaning up containers of force-terminated sessions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, override

from ai.backend.common.clients.valkey_client.valkey_schedule.client import ValkeyScheduleClient
from ai.backend.common.events.event_types.kernel.types import KernelLifecycleEventReason
from ai.backend.common.types import SessionId
from ai.backend.logging import BraceStyleAdapter
from ai.backend.manager.defs import UNAVAILABLE_AGENT_CLEANUP_RETRY_SEC
from ai.backend.manager.repositories.scheduler.repository import SchedulerRepository
from ai.backend.manager.sokovan.scheduler.handlers.cleanup.base import CleanupHandler

if TYPE_CHECKING:
    from ai.backend.manager.sokovan.scheduler.terminator.terminator import SessionTerminator

log = BraceStyleAdapter(logging.getLogger(__name__))


class CleanupForceTerminatedHandler(CleanupHandler):
    """Cleanup containers for force-terminated sessions.

    Force-terminated sessions skip the TERMINATING state and go directly to TERMINATED,
    bypassing the normal terminate handler that sends destroy RPCs to agents.
    This handler reads force-terminated session IDs from Valkey, fetches kernel/agent
    info from DB, and sends destroy RPCs to ensure containers are cleaned up.

    Only successfully cleaned-up session IDs are removed from Valkey; failed ones
    remain for retry on the next cycle.

    The Valkey set expires, so it cannot by itself outlast a long agent outage. Sessions
    released with ``hang-timeout`` from an agent that is still fenced and unavailable are
    therefore queued again on every pass, which refreshes the set's TTL; once the agent is
    back, the remaining TTL leaves time for the destroy retries.
    """

    def __init__(
        self,
        terminator: SessionTerminator,
        repository: SchedulerRepository,
        valkey_schedule: ValkeyScheduleClient,
    ) -> None:
        self._terminator = terminator
        self._repository = repository
        self._valkey_schedule = valkey_schedule

    @classmethod
    @override
    def name(cls) -> str:
        return "cleanup-force-terminated"

    @override
    async def fetch_session_ids(self) -> Sequence[SessionId]:
        requeued = await self._requeue_sessions_on_fenced_agents()
        queued = await self._valkey_schedule.get_force_terminated_sessions()
        # The queued set already holds the requeued ids unless adding them failed.
        return list(dict.fromkeys([*queued, *requeued]))

    async def _requeue_sessions_on_fenced_agents(self) -> list[SessionId]:
        """Queue again the sessions released from agents that are still away.

        Never raises: on any error the handler goes on with the queued set alone.
        """
        try:
            session_ids = await self._repository.get_sessions_terminated_on_fenced_agents(
                KernelLifecycleEventReason.HANG_TIMEOUT.value,
                UNAVAILABLE_AGENT_CLEANUP_RETRY_SEC,
            )
        except Exception:
            log.exception("Failed to read the sessions released from fenced agents")
            return []
        if not session_ids:
            return []
        try:
            await self._valkey_schedule.add_force_terminated_sessions(session_ids)
        except Exception:
            log.exception(
                "Failed to queue {} sessions released from fenced agents for cleanup again",
                len(session_ids),
            )
        else:
            log.debug(
                "Queued {} sessions released from fenced agents for cleanup again",
                len(session_ids),
            )
        return session_ids

    @override
    async def execute(self, session_ids: Sequence[SessionId]) -> None:
        log.info("Processing {} force-terminated sessions for container cleanup", len(session_ids))

        terminating_sessions = await self._repository.get_terminating_sessions_by_ids(
            list(session_ids)
        )
        if not terminating_sessions:
            log.warning(
                "No session data found for force-terminated sessions: {}",
                session_ids,
            )
            # Sessions no longer exist in DB — remove from Valkey to avoid infinite retry
            await self._valkey_schedule.remove_force_terminated_sessions(session_ids)
            return

        # One call for all sessions: the destroy requests run concurrently, so an
        # unresponsive agent does not hold up the cleanup of the others.
        try:
            succeeded_ids = await self._terminator.terminate_sessions_for_handler(
                terminating_sessions
            )
        except Exception:
            log.exception(
                "Failed to send cleanup RPCs for {} force-terminated sessions",
                len(terminating_sessions),
            )
            return

        failed_count = len(terminating_sessions) - len(succeeded_ids)
        if succeeded_ids:
            await self._valkey_schedule.remove_force_terminated_sessions(succeeded_ids)
        log.info(
            "Cleaned up {} force-terminated sessions ({} failed, retried on the next cycle)",
            len(succeeded_ids),
            failed_count,
        )
