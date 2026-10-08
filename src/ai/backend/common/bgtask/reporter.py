import uuid
from typing import Final

from ai.backend.common.events.dispatcher import EventProducer
from ai.backend.common.events.event_types.bgtask.broadcast import BgtaskUpdatedEvent

from .types import bgtask_cache_id


class ProgressReporter:
    total_progress: int | float
    current_progress: int | float

    _event_producer: Final[EventProducer]
    _task_id: Final[uuid.UUID]
    _cache_id: Final[str]

    def __init__(
        self,
        event_producer: EventProducer,
        task_id: uuid.UUID,
        current_progress: int = 0,
        total_progress: int = 0,
        *,
        cache_scope: str | None = None,
    ) -> None:
        self._event_producer = event_producer
        self._task_id = task_id
        self._cache_id = bgtask_cache_id(task_id, cache_scope)
        self.current_progress = current_progress
        self.total_progress = total_progress

    async def update(
        self,
        increment: int | float = 0,
        message: str | None = None,
    ) -> None:
        self.current_progress += increment
        # keep the state as local variables because they might be changed
        # due to interleaving at await statements below.
        current, total = self.current_progress, self.total_progress
        await self._event_producer.broadcast_event_with_cache(
            self._cache_id,
            BgtaskUpdatedEvent(
                self._task_id,
                message=message,
                current_progress=current,
                total_progress=total,
            ),
        )
