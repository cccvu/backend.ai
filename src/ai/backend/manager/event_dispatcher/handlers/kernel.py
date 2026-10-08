import logging
from io import BytesIO
from typing import Final

import sqlalchemy as sa

from ai.backend.common.clients.valkey_client.valkey_container_log.client import (
    ValkeyContainerLogClient,
)
from ai.backend.common.clients.valkey_client.valkey_stat.client import ValkeyStatClient
from ai.backend.common.clients.valkey_client.valkey_stream.client import ValkeyStreamClient
from ai.backend.common.events.event_types.kernel.anycast import (
    DoSyncKernelLogsEvent,
    KernelCancelledAnycastEvent,
    KernelCreatingAnycastEvent,
    KernelPreparingAnycastEvent,
    KernelPullingAnycastEvent,
    KernelStartedAnycastEvent,
    KernelTerminatedAnycastEvent,
    KernelTerminatingAnycastEvent,
)
from ai.backend.common.log.types import ContainerLogError
from ai.backend.common.types import (
    AgentId,
)
from ai.backend.logging import BraceStyleAdapter
from ai.backend.manager.models.kernel import kernels
from ai.backend.manager.models.utils import (
    ExtendedAsyncSAEngine,
    execute_with_retry,
)
from ai.backend.manager.registry import AgentRegistry
from ai.backend.manager.sokovan.scheduler.coordinator import ScheduleCoordinator

log = BraceStyleAdapter(logging.getLogger(__spec__.name))

# The maximum size of a container log kept in the database.
MAX_CONTAINER_LOG_SIZE: Final = 10 * 1024 * 1024
# The maximum number of stored chunks read for one container.
MAX_CONTAINER_LOG_CHUNKS: Final = 16 * 1024
# The maximum size of one stored chunk: a base64-encoded chunk of up to
# MAX_CONTAINER_LOG_SIZE bytes plus the serialization overhead.
MAX_CONTAINER_LOG_ELEMENT_SIZE: Final = MAX_CONTAINER_LOG_SIZE * 4 // 3 + 64 * 1024


class KernelEventHandler:
    _valkey_container_log: ValkeyContainerLogClient
    _valkey_stat: ValkeyStatClient
    _valkey_stream: ValkeyStreamClient
    _registry: AgentRegistry
    _db: ExtendedAsyncSAEngine
    _schedule_coordinator: ScheduleCoordinator

    def __init__(
        self,
        valkey_container_log: ValkeyContainerLogClient,
        valkey_stat: ValkeyStatClient,
        valkey_stream: ValkeyStreamClient,
        registry: AgentRegistry,
        db: ExtendedAsyncSAEngine,
        schedule_coordinator: ScheduleCoordinator,
    ) -> None:
        self._valkey_container_log = valkey_container_log
        self._valkey_stat = valkey_stat
        self._valkey_stream = valkey_stream
        self._registry = registry
        self._db = db
        self._schedule_coordinator = schedule_coordinator

    async def handle_kernel_log(
        self,
        _context: None,
        source: AgentId,
        event: DoSyncKernelLogsEvent,
    ) -> None:
        try:
            async with self._db.begin_readonly() as conn:
                query = sa.select(kernels.c.agent, kernels.c.container_id).where(
                    kernels.c.id == event.kernel_id
                )
                row = (await conn.execute(query)).first()
        except Exception:
            log.exception("handle_kernel_log: failed to load kernel {}", event.kernel_id)
            return
        if row is None or row.agent is None:
            log.warning("handle_kernel_log: kernel {} has no agent to sync from", event.kernel_id)
            return
        # A kernel that failed to start has no container recorded, but its agent still sends
        # the container's logs: take the container from the event then. The key stays scoped by
        # the kernel's agent, so only that agent's own logs are ever read.
        recorded = row.container_id is not None
        if (
            row.agent != source
            or not event.container_id
            or (recorded and row.container_id != event.container_id)
        ):
            log.warning(
                "handle_kernel_log: ignoring logs of kernel {} sent by agent {} "
                "(kernel agent: {}, container recorded: {}, container matches: {})",
                event.kernel_id,
                source,
                row.agent,
                recorded,
                row.container_id == event.container_id,
            )
            return
        # The agent part of the key comes from the kernel row, never from the event.
        agent_id = AgentId(row.agent)
        container_id = str(row.container_id if recorded else event.container_id)
        try:
            try:
                log_data = await self._read_container_logs(agent_id, container_id)

                async def _update_log() -> None:
                    async with self._db.begin() as conn:
                        update_query = (
                            sa.update(kernels)
                            .values(container_log=log_data)
                            .where(kernels.c.id == event.kernel_id)
                        )
                        await conn.execute(update_query)

                await execute_with_retry(_update_log)
            finally:
                # Clear the log data from Redis on every path, so a bad list never outlives
                # one attempt.
                await self._valkey_container_log.clear_container_logs(
                    agent_id=agent_id,
                    container_id=container_id,
                )
        except Exception:
            # skip all exception in handle_kernel_log
            log.warning("handle_kernel_log: failed to sync logs of kernel {}", event.kernel_id)

    async def _read_container_logs(self, agent_id: AgentId, container_id: str) -> bytes:
        """
        Pop the stored log chunks of a container.

        At most MAX_CONTAINER_LOG_CHUNKS chunks are read and at most MAX_CONTAINER_LOG_SIZE
        bytes are kept; the rest is dropped with a marker.
        """
        log_buffer = BytesIO()
        try:
            list_size = await self._valkey_container_log.container_log_len(
                agent_id=agent_id,
                container_id=container_id,
            )
            truncated = list_size > MAX_CONTAINER_LOG_CHUNKS
            num_chunks = min(list_size, MAX_CONTAINER_LOG_CHUNKS)
            for index in range(num_chunks):
                remaining = MAX_CONTAINER_LOG_SIZE - log_buffer.tell()
                # Read chunk-by-chunk to allow interleaving with other Redis operations.
                try:
                    chunks = await self._valkey_container_log.pop_container_logs(
                        agent_id=agent_id,
                        container_id=container_id,
                        max_element_size=MAX_CONTAINER_LOG_ELEMENT_SIZE,
                    )
                    if chunks is None:  # maybe missing
                        log_buffer.write(b"(container log unavailable)\n")
                        break
                    for chunk in chunks:
                        content, chunk_truncated = chunk.get_bounded_content(remaining)
                        log_buffer.write(content)
                        remaining -= len(content)
                        truncated = truncated or chunk_truncated
                except ContainerLogError:
                    log.warning("skipping an unreadable log chunk of container {}", container_id)
                    continue
                if remaining <= 0:
                    # The rest is dropped with the key.
                    truncated = truncated or index + 1 < num_chunks
                    break
            if truncated:
                log_buffer.write(b"(container log truncated)\n")
            return log_buffer.getvalue()
        finally:
            log_buffer.close()

    async def handle_kernel_preparing(
        self,
        _context: None,
        _source: AgentId,
        event: KernelPreparingAnycastEvent,
    ) -> None:
        log.info(
            "handle_kernel_preparing: ev:{} k:{}",
            event.event_name(),
            event.kernel_id,
        )

        await self._schedule_coordinator.handle_kernel_preparing(event)

    async def handle_kernel_pulling(
        self,
        _context: None,
        _source: AgentId,
        event: KernelPullingAnycastEvent,
    ) -> None:
        log.info(
            "handle_kernel_pulling: ev:{} k:{}",
            event.event_name(),
            event.kernel_id,
        )

        await self._schedule_coordinator.handle_kernel_pulling(event)

    async def handle_kernel_creating(
        self,
        _context: None,
        _source: AgentId,
        event: KernelCreatingAnycastEvent,
    ) -> None:
        log.info(
            "handle_kernel_creating: ev:{} k:{}",
            event.event_name(),
            event.kernel_id,
        )

        await self._schedule_coordinator.handle_kernel_creating(event)

    async def handle_kernel_started(
        self,
        _context: None,
        _source: AgentId,
        event: KernelStartedAnycastEvent,
    ) -> None:
        log.info(
            "handle_kernel_started: ev:{} k:{}",
            event.event_name(),
            event.kernel_id,
        )

        await self._schedule_coordinator.handle_kernel_running(event)

    async def handle_kernel_cancelled(
        self,
        _context: None,
        _source: AgentId,
        event: KernelCancelledAnycastEvent,
    ) -> None:
        log.info(
            "handle_kernel_cancelled: ev:{} k:{}",
            event.event_name(),
            event.kernel_id,
        )

        await self._schedule_coordinator.handle_kernel_cancelled(event)

    async def handle_kernel_terminating(
        self,
        context: None,
        source: AgentId,
        event: KernelTerminatingAnycastEvent,
    ) -> None:
        # `destroy_kernel()` has already changed the kernel status to "TERMINATING".
        # No additional handling needed for terminating state
        pass

    async def handle_kernel_terminated(
        self,
        _context: None,
        _source: AgentId,
        event: KernelTerminatedAnycastEvent,
    ) -> None:
        await self._schedule_coordinator.handle_kernel_terminated(event)
