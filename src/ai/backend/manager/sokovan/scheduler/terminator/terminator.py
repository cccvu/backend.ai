"""Session termination and sweep operations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from uuid import UUID

from ai.backend.common.clients.valkey_client.valkey_schedule import HealthCheckStatus
from ai.backend.common.clients.valkey_client.valkey_schedule.client import ValkeyScheduleClient
from ai.backend.common.types import AgentId, KernelId, ResourceSlot, SessionId
from ai.backend.logging.utils import BraceStyleAdapter
from ai.backend.manager.clients.agent import AgentClientPool
from ai.backend.manager.data.kernel.types import KernelInfo
from ai.backend.manager.defs import (
    AGENT_CHECK_RUNNING_TIMEOUT_SEC,
    AGENT_DESTROY_KERNEL_TIMEOUT_SEC,
)
from ai.backend.manager.repositories.scheduler import SchedulerRepository
from ai.backend.manager.sokovan.recorder.context import RecorderContext
from ai.backend.manager.views.sokovan.session import (
    KernelTerminationResult,
    TerminatingSessionData,
)

log = BraceStyleAdapter(logging.getLogger(__spec__.name))


@dataclass
class SessionTerminatorArgs:
    """Arguments for SessionTerminator initialization."""

    repository: SchedulerRepository
    agent_client_pool: AgentClientPool
    valkey_schedule: ValkeyScheduleClient


class SessionTerminator:
    """Handles termination and sweep operations for sessions and kernels."""

    _repository: SchedulerRepository
    _agent_client_pool: AgentClientPool
    _valkey_schedule: ValkeyScheduleClient

    def __init__(self, args: SessionTerminatorArgs) -> None:
        self._repository = args.repository
        self._agent_client_pool = args.agent_client_pool
        self._valkey_schedule = args.valkey_schedule

    async def terminate_sessions_for_handler(
        self,
        terminating_sessions: list[TerminatingSessionData],
    ) -> list[SessionId]:
        """
        Send termination requests for the given sessions.

        Handler-specific method that works with pre-fetched data.
        Used by TerminateSessionsLifecycleHandler.

        A session counts as succeeded when every kernel with an assigned agent
        acknowledged its destroy request. Kernels without an agent were never
        placed on an agent, so there is no container to destroy and they do not
        count against the session; a session with no such kernels succeeds trivially.

        :param terminating_sessions: List of sessions to terminate with kernel details
        :return: IDs of the sessions whose destroy requests all succeeded
        """
        return await self._terminate_sessions_internal(terminating_sessions)

    async def _terminate_sessions_internal(
        self,
        terminating_sessions: list[TerminatingSessionData],
    ) -> list[SessionId]:
        """
        Internal implementation for terminating sessions.

        No status updates are performed here; events and the sweep handle them.

        :param terminating_sessions: List of sessions to terminate
        :return: IDs of the sessions whose destroy requests all succeeded
        """
        if not terminating_sessions:
            log.debug("No sessions to terminate")
            return []

        log.info("Processing {} sessions for termination", len(terminating_sessions))

        # Collect all termination tasks from all sessions
        all_tasks: list[Awaitable[KernelTerminationResult]] = []
        task_session_ids: list[SessionId] = []
        skipped_kernels = 0

        for session in terminating_sessions:
            for kernel in session.kernels:
                # Only process kernels with assigned agents
                if kernel.agent_id:
                    task = self._terminate_kernel(
                        kernel.agent_id,
                        kernel.kernel_id,
                        session.session_id,
                        session.status_info,
                        kernel.occupied_slots,
                    )
                    all_tasks.append(task)
                    task_session_ids.append(session.session_id)
                else:
                    # Kernel has no agent assigned - needs sweep
                    skipped_kernels += 1

        # Kernels without agents will be handled by retry/timeout mechanism
        if skipped_kernels > 0:
            log.info(
                "Found {} kernels without agents, will be handled by retry/timeout",
                skipped_kernels,
            )

        # Execute all termination tasks concurrently across all sessions
        if not all_tasks:
            log.debug("No kernels with agents to terminate")
            return [session.session_id for session in terminating_sessions]

        log.info("Terminating {} kernels in parallel", len(all_tasks))

        # Use gather with return_exceptions to ensure partial failures don't block others
        with RecorderContext[SessionId].shared_phase(
            "kernel_destruction",
            success_detail="Kernels terminating",
        ):
            with RecorderContext[SessionId].shared_step(
                "destroy_kernels",
                success_detail="Kernel destruction requested",
            ):
                results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Log results but don't update DB (handled by events and sweep)
        success_count = 0
        failed_count = 0
        failed_session_ids: set[SessionId] = set()
        for session_id, r in zip(task_session_ids, results, strict=True):
            if isinstance(r, BaseException) or not r.success:
                failed_count += 1
                failed_session_ids.add(session_id)
                continue
            success_count += 1

        log.info(
            "Termination RPC calls completed: {} successful, {} failed",
            success_count,
            failed_count,
        )

        return [
            session.session_id
            for session in terminating_sessions
            if session.session_id not in failed_session_ids
        ]

    async def _terminate_kernel(
        self,
        agent_id: AgentId,
        kernel_id: KernelId,
        session_id: SessionId,
        reason: str,
        occupied_slots: ResourceSlot,
    ) -> KernelTerminationResult:
        """
        Terminate a single kernel on an agent.

        The RPC is bounded by AGENT_DESTROY_KERNEL_TIMEOUT_SEC. On timeout the agent
        keeps destroying the kernel; the request is re-sent on a later attempt.

        :param agent_id: The agent ID where the kernel is running
        :param kernel_id: The kernel ID to terminate
        :param session_id: The session ID that owns the kernel
        :param reason: The reason for termination
        :return: KernelTerminationResult with success status
        """
        try:
            # The timeout wraps acquire() so that it reaches the pool as a cancellation,
            # which the pool does not count as a connection failure.
            async with asyncio.timeout(AGENT_DESTROY_KERNEL_TIMEOUT_SEC):
                async with self._agent_client_pool.acquire(agent_id) as client:
                    # Call agent's destroy_kernel RPC method with correct parameters
                    await client.destroy_kernel(
                        kernel_id, session_id, reason, suppress_events=False
                    )
            return KernelTerminationResult(
                kernel_id=kernel_id,
                agent_id=agent_id,
                occupied_slots=occupied_slots,
                success=True,
            )
        except TimeoutError:
            log.warning(
                "Timed out terminating kernel {} on agent {} after {}s",
                kernel_id,
                agent_id,
                AGENT_DESTROY_KERNEL_TIMEOUT_SEC,
            )
            return KernelTerminationResult(
                kernel_id=kernel_id,
                agent_id=agent_id,
                occupied_slots=occupied_slots,
                success=False,
                error=f"destroy_kernel timed out after {AGENT_DESTROY_KERNEL_TIMEOUT_SEC}s",
            )
        except Exception as e:
            log.warning(
                "Failed to terminate kernel {} on agent {}: {}",
                kernel_id,
                agent_id,
                e,
            )

            return KernelTerminationResult(
                kernel_id=kernel_id,
                agent_id=agent_id,
                occupied_slots=occupied_slots,
                success=False,
                error=str(e),
            )

    async def check_stale_kernels(
        self,
        kernels: list[KernelInfo],
    ) -> list[KernelId]:
        """
        Check for stale kernels from given kernel list.

        Kernel handler-specific method that works with KernelInfo directly.
        Used by SweepStaleKernelsKernelHandler.

        This method:
        1. Checks kernel presence status in Valkey
        2. For potentially stale kernels, confirms with agent if truly dead
        3. Returns list of kernel IDs that are confirmed dead

        :param kernels: List of RUNNING kernels to check for staleness
        :return: List of kernel IDs that are dead/stale
        """
        if not kernels:
            return []

        # 1. Extract kernel IDs and the agents hosting them, from the database records
        kernel_ids: list[KernelId] = []
        kernel_agents: dict[KernelId, AgentId] = {}
        for kernel_info in kernels:
            kernel_id = KernelId(kernel_info.id)
            kernel_ids.append(kernel_id)
            if kernel_info.resource.agent:
                kernel_agents[kernel_id] = AgentId(kernel_info.resource.agent)

        if not kernel_ids:
            return []

        # 2. Check presence status in Valkey (kernels without an agent have no presence)
        statuses = await self._valkey_schedule.check_kernel_presence_status_batch(
            kernel_agents,
            agent_ids=set(kernel_agents.values()),
        )

        # 3. Filter STALE kernels (None status or STALE presence)
        stale_kernel_id_set: set[UUID] = {
            kernel_id
            for kernel_id in kernel_ids
            if (status := statuses.get(kernel_id)) is None
            or status.presence == HealthCheckStatus.STALE
        }
        if not stale_kernel_id_set:
            return []

        # 4. Check with agents concurrently - only explicit False terminates
        candidates: list[tuple[KernelId, AgentId]] = []
        for kernel_info in kernels:
            if kernel_info.id not in stale_kernel_id_set:
                continue
            if not kernel_info.resource.agent:
                continue
            candidates.append((KernelId(kernel_info.id), AgentId(kernel_info.resource.agent)))
        if not candidates:
            return []

        # Failures (including timeouts) are returned as results so that one
        # unresponsive agent does not hold up the checks on the others.
        results = await asyncio.gather(
            *(
                self._check_kernel_running(kernel_id, agent_id)
                for kernel_id, agent_id in candidates
            ),
            return_exceptions=True,
        )

        dead_kernel_ids: list[KernelId] = []
        for (kernel_id, agent_id), result in zip(candidates, results, strict=True):
            if isinstance(result, TimeoutError):
                log.warning(
                    "Timed out checking kernel {} status on agent {} after {}s. Skipping.",
                    kernel_id,
                    agent_id,
                    AGENT_CHECK_RUNNING_TIMEOUT_SEC,
                )
                continue
            if isinstance(result, BaseException):
                log.warning(
                    "Failed to check kernel {} status: {!r}. Skipping.",
                    kernel_id,
                    result,
                )
                continue
            if result is False:
                dead_kernel_ids.append(kernel_id)

        if dead_kernel_ids:
            log.info("Found {} stale kernels to be terminated", len(dead_kernel_ids))

        return dead_kernel_ids

    async def _check_kernel_running(self, kernel_id: KernelId, agent_id: AgentId) -> bool:
        """
        Ask the agent whether the kernel is running, bounded by AGENT_CHECK_RUNNING_TIMEOUT_SEC.

        :param kernel_id: The kernel ID to check
        :param agent_id: The agent ID recorded for the kernel
        :return: The agent's answer
        """
        # The timeout wraps acquire() so that it reaches the pool as a cancellation,
        # which the pool does not count as a connection failure.
        async with asyncio.timeout(AGENT_CHECK_RUNNING_TIMEOUT_SEC):
            async with self._agent_client_pool.acquire(agent_id) as client:
                return await client.check_running(kernel_id)
