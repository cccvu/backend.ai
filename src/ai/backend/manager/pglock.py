from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, override

from ai.backend.common.lock import AbstractDistributedLock

from .defs import LockID
from .errors.resource import DBOperationFailed

if TYPE_CHECKING:
    from .models.utils import ExtendedAsyncSAEngine


class PgAdvisoryLock(AbstractDistributedLock):
    """A PostgreSQL session-level advisory lock.

    With ``blocking=False``, entering raises ``LockNotAcquired`` instead of
    waiting when another holder has the lock.
    """

    _lock_ctx: AbstractAsyncContextManager[None] | None

    def __init__(
        self,
        db: ExtendedAsyncSAEngine,
        lock_id: LockID,
        *,
        blocking: bool = True,
    ) -> None:
        self.db = db
        self.lock_id = lock_id
        self.blocking = blocking
        self._lock_ctx = None

    @override
    async def __aenter__(self) -> Any:
        lock_ctx = self.db.advisory_lock(self.lock_id, blocking=self.blocking)
        await lock_ctx.__aenter__()
        self._lock_ctx = lock_ctx

    @override
    async def __aexit__(self, *exc_info: Any) -> bool | None:
        if self._lock_ctx is None:
            raise DBOperationFailed("Lock context is not initialized")
        try:
            return await self._lock_ctx.__aexit__(*exc_info)
        finally:
            self._lock_ctx = None
