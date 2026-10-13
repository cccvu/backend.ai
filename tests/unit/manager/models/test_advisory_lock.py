import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
import sqlalchemy as sa

from ai.backend.manager.defs import LockID
from ai.backend.manager.errors.resource import LockNotAcquired
from ai.backend.manager.models.base import pgsql_connect_opts
from ai.backend.manager.models.utils import ExtendedAsyncSAEngine
from ai.backend.manager.pglock import PgAdvisoryLock
from ai.backend.manager.repositories.db.engine import create_async_engine


async def test_lock(database_connection: ExtendedAsyncSAEngine) -> None:
    enter_count = 0
    done_count = 0

    async def critical_section(db: ExtendedAsyncSAEngine) -> None:
        nonlocal enter_count
        async with db.advisory_lock(LockID.LOCKID_TEST):
            enter_count += 1
            await asyncio.sleep(1.0)

    tasks = []
    for idx in range(5):
        tasks.append(
            asyncio.create_task(
                critical_section(database_connection),
                name=f"critical-section-{idx}",
            ),
        )
    await asyncio.sleep(0.5)

    async with database_connection.connect() as conn:
        result = await conn.exec_driver_sql(
            "SELECT objid, granted, pid FROM pg_locks WHERE locktype = 'advisory' AND objid = 42;",
        )
        rows = result.fetchall()
        print(rows)
        result = await conn.exec_driver_sql(
            "SELECT objid, granted FROM pg_locks "
            "WHERE locktype = 'advisory' AND objid = 42 AND granted = 't';",
        )
        rows = result.fetchall()
        assert len(rows) == 1

    await asyncio.sleep(2.5)
    for t in tasks:
        if t.done():
            done_count += 1
        else:
            try:
                t.cancel()
                await t
            except asyncio.CancelledError:
                pass
    await asyncio.sleep(0.1)

    assert 2 <= done_count <= 3
    assert enter_count >= done_count

    # Check all tasks have unlocked.
    async with database_connection.connect() as conn:
        result = await conn.exec_driver_sql(
            "SELECT objid, granted, pid FROM pg_locks "
            "WHERE locktype = 'advisory' AND objid = 42 AND granted = 't';",
        )
        rows = result.fetchall()
        print(rows)
        assert len(rows) == 0


async def test_lock_timeout(database_connection: ExtendedAsyncSAEngine) -> None:
    lock_connection_timeout = 0.5
    sleep = 1

    with pytest.raises(asyncio.TimeoutError):
        database_connection.lock_conn_timeout = lock_connection_timeout
        async with database_connection.advisory_lock(LockID.LOCKID_TEST):
            await asyncio.sleep(sleep)


# =============================================================================
# Lock connection mode, non-blocking acquisition and release
# =============================================================================


def _url(engine: ExtendedAsyncSAEngine) -> str:
    return engine.url.render_as_string(hide_password=False)


@asynccontextmanager
async def _engine(url: str, **kwargs: Any) -> AsyncIterator[ExtendedAsyncSAEngine]:
    engine = create_async_engine(url, pool_pre_ping=False, **kwargs)
    try:
        yield engine
    finally:
        await engine.dispose()


class _CheckoutCounter:
    """Counts pool checkouts and the peak number of checked-out connections."""

    def __init__(self, engine: ExtendedAsyncSAEngine) -> None:
        self.checkouts = 0
        self.current = 0
        self.peak = 0
        sa.event.listen(engine.sync_engine, "checkout", self._on_checkout)
        sa.event.listen(engine.sync_engine, "checkin", self._on_checkin)

    def _on_checkout(self, *args: Any) -> None:
        self.checkouts += 1
        self.current += 1
        self.peak = max(self.peak, self.current)

    def _on_checkin(self, *args: Any) -> None:
        self.current -= 1


async def _lock_rows(engine: ExtendedAsyncSAEngine, lock_id: int) -> list[Any]:
    """The advisory lock rows for ``lock_id`` with their backends' state."""
    async with engine.connect() as conn:
        result = await conn.exec_driver_sql(
            "SELECT l.granted, l.pid, a.state FROM pg_locks l "
            "JOIN pg_stat_activity a ON a.pid = l.pid "
            f"WHERE l.locktype = 'advisory' AND l.objid = {lock_id:d};",
        )
        return list(result.fetchall())


async def _is_free(engine: ExtendedAsyncSAEngine, lock_id: int) -> bool:
    """Whether another session can take ``lock_id`` right now (it releases it at once)."""
    async with engine.connect() as conn:
        result = await conn.exec_driver_sql(f"SELECT pg_try_advisory_lock({lock_id:d});")
        acquired = result.scalar()
        if acquired:
            await conn.exec_driver_sql(f"SELECT pg_advisory_unlock({lock_id:d});")
        return bool(acquired)


async def _wait_for(predicate: Any, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not await predicate():
            await asyncio.sleep(0.02)


LOCK = LockID.LOCKID_TEST


class TestLockConnection:
    async def test_lock_connection_is_not_idle_in_a_transaction(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        async with database_connection.advisory_lock(LOCK):
            rows = await _lock_rows(database_connection, LOCK)
            assert [(granted, state) for granted, _, state in rows] == [(True, "idle")]

    async def test_lock_outlives_idle_in_transaction_timeout(
        self, database_connection: ExtendedAsyncSAEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Without AUTOCOMMIT the server ends the lock's session after 500 ms of
        # idling in the implicit transaction, and the lock with it.
        async with _engine(
            _url(database_connection),
            connect_args={"server_settings": {"idle_in_transaction_session_timeout": "500"}},
        ) as engine:
            with caplog.at_level(logging.WARNING):
                async with engine.advisory_lock(LOCK):
                    await asyncio.sleep(1.0)
                    assert not await _is_free(database_connection, LOCK)
            assert "was not held at release" not in caplog.text
            assert await _is_free(database_connection, LOCK)

    @pytest.mark.parametrize("isolation_level", ["SERIALIZABLE", "READ COMMITTED"])
    async def test_autocommit_does_not_leak_into_the_pool(
        self, database_connection: ExtendedAsyncSAEngine, isolation_level: str
    ) -> None:
        async def isolation(engine: ExtendedAsyncSAEngine) -> str:
            async with engine.connect() as conn:
                result = await conn.exec_driver_sql("SHOW transaction_isolation;")
                value = result.scalar()
                assert isinstance(value, str)
                return value

        expected = isolation_level.lower()
        async with _engine(
            _url(database_connection),
            pool_size=1,
            max_overflow=0,
            isolation_level=isolation_level,
        ) as engine:
            async with engine.advisory_lock(LOCK):
                pass
            assert await isolation(engine) == expected

            # The same after a refused non-blocking attempt.
            async with database_connection.advisory_lock(LOCK):
                with pytest.raises(LockNotAcquired):
                    async with engine.advisory_lock(LOCK, blocking=False):
                        pass
            assert await isolation(engine) == expected

    def test_no_idle_session_timeout(self) -> None:
        # It would end an AUTOCOMMIT lock connection, which idles while the body runs.
        assert "idle_session_timeout" not in pgsql_connect_opts["server_settings"]


class TestNonBlocking:
    async def test_refused_when_another_session_holds_it(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        async with _engine(_url(database_connection)) as other:
            async with other.advisory_lock(LOCK):
                started = time.perf_counter()
                with pytest.raises(LockNotAcquired):
                    async with database_connection.advisory_lock(LOCK, blocking=False):
                        pytest.fail("the body must not run")
                assert time.perf_counter() - started < 0.5
            assert LOCK not in database_connection._held_advisory_locks
            # No waiter was left behind.
            assert await _lock_rows(database_connection, LOCK) == []

    async def test_acquires_a_free_lock(self, database_connection: ExtendedAsyncSAEngine) -> None:
        async with database_connection.advisory_lock(LOCK, blocking=False):
            assert not await _is_free(database_connection, LOCK)
        assert await _is_free(database_connection, LOCK)

    async def test_same_process_try_skips_without_a_connection(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        counter = _CheckoutCounter(database_connection)
        async with database_connection.advisory_lock(LOCK):
            checkouts = counter.checkouts
            with pytest.raises(LockNotAcquired):
                async with database_connection.advisory_lock(LOCK, blocking=False):
                    pytest.fail("the body must not run")
            assert counter.checkouts == checkouts
            # The refused attempt did not remove the holder's entry.
            assert LOCK in database_connection._held_advisory_locks
        assert LOCK not in database_connection._held_advisory_locks

    async def test_concurrent_tries_check_and_add_before_awaiting(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        counter = _CheckoutCounter(database_connection)
        entered = 0

        async def attempt() -> None:
            nonlocal entered
            async with database_connection.advisory_lock(LOCK, blocking=False):
                entered += 1
                await asyncio.sleep(0.2)

        results = await asyncio.gather(*[attempt() for _ in range(10)], return_exceptions=True)
        assert entered == 1
        assert sum(isinstance(r, LockNotAcquired) for r in results) == 9
        assert counter.checkouts == 1
        assert LOCK not in database_connection._held_advisory_locks


class TestHeldSet:
    async def test_cleared_after_a_body_exception(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        with pytest.raises(ValueError):
            async with database_connection.advisory_lock(LOCK, blocking=False):
                raise ValueError("body")
        assert LOCK not in database_connection._held_advisory_locks
        assert await _is_free(database_connection, LOCK)

    async def test_cleared_after_cancelling_a_waiter(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        async with _engine(_url(database_connection)) as other:
            async with other.advisory_lock(LOCK):

                async def wait() -> None:
                    async with database_connection.advisory_lock(LOCK):
                        pass

                task = asyncio.create_task(wait())

                async def waiting() -> bool:
                    return any(not granted for granted, _, _ in await _lock_rows(other, LOCK))

                await _wait_for(waiting)
                assert LOCK in database_connection._held_advisory_locks
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert LOCK not in database_connection._held_advisory_locks

    async def test_only_the_adder_removes_the_id(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        holder_entered = asyncio.Event()
        release_holder = asyncio.Event()

        async def hold() -> None:
            async with database_connection.advisory_lock(LOCK):
                holder_entered.set()
                await release_holder.wait()

        holder = asyncio.create_task(hold())
        await holder_entered.wait()

        # A blocking waiter in the same process finds the id present; it is not the
        # adder, so its cancellation leaves the holder's entry in place.
        async def wait() -> None:
            async with database_connection.advisory_lock(LOCK):
                pass

        waiter = asyncio.create_task(wait())

        async def waiting() -> bool:
            return any(not granted for granted, _, _ in await _lock_rows(database_connection, LOCK))

        await _wait_for(waiting)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert LOCK in database_connection._held_advisory_locks
        with pytest.raises(LockNotAcquired):
            async with database_connection.advisory_lock(LOCK, blocking=False):
                pass

        release_holder.set()
        await holder
        assert LOCK not in database_connection._held_advisory_locks


class TestAcquireAndRelease:
    async def test_cancel_after_grant_releases_the_lock(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        async with (
            _engine(_url(database_connection)) as holder_engine,
            _engine(_url(database_connection)) as waiter_engine,
        ):
            holder = holder_engine.advisory_lock(LOCK)
            await holder.__aenter__()

            async def wait() -> None:
                async with waiter_engine.advisory_lock(LOCK):
                    await asyncio.sleep(100)

            waiter = asyncio.create_task(wait())

            async def waiting() -> bool:
                return any(not granted for granted, _, _ in await _lock_rows(holder_engine, LOCK))

            await _wait_for(waiting)
            # The server grants the lock to the waiter as soon as the holder unlocks;
            # cancel the waiter before it can observe the grant.
            await holder.__aexit__(None, None, None)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter

            # Another engine gets the lock: the cancelled waiter does not keep it.
            async with asyncio.timeout(3.0):
                async with database_connection.advisory_lock(LOCK):
                    pass

    async def test_lock_timeout_on_acquire_maps_to_cancelled(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        async with (
            _engine(
                _url(database_connection),
                connect_args={"server_settings": {"lock_timeout": "100"}},
            ) as engine,
            database_connection.advisory_lock(LOCK),
        ):
            with pytest.raises(asyncio.CancelledError):
                async with engine.advisory_lock(LOCK):
                    pytest.fail("the body must not run")
            assert LOCK not in engine._held_advisory_locks

    async def test_lock_not_available_from_the_body_passes_through(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        class _Orig(Exception):
            pgcode = "55P03"

        error = sa.exc.DBAPIError("SELECT 1", None, _Orig())
        with pytest.raises(sa.exc.DBAPIError) as exc_info:
            async with database_connection.advisory_lock(LOCK):
                raise error
        assert exc_info.value is error
        assert await _is_free(database_connection, LOCK)

    async def test_lost_lock_is_logged_and_does_not_mask_the_body(
        self, database_connection: ExtendedAsyncSAEngine, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def terminate_holder() -> None:
            (row,) = await _lock_rows(database_connection, LOCK)
            async with database_connection.connect() as conn:
                await conn.exec_driver_sql(f"SELECT pg_terminate_backend({row[1]:d});")

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match="body"):
                async with database_connection.advisory_lock(LOCK):
                    await terminate_holder()
                    raise ValueError("body")
            assert "was not held at release" in caplog.text

            caplog.clear()
            async with database_connection.advisory_lock(LOCK):
                await terminate_holder()
            assert "was not held at release" in caplog.text
        assert await _is_free(database_connection, LOCK)

    async def test_unlock_returning_false_invalidates_the_connection(
        self,
        database_connection: ExtendedAsyncSAEngine,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Enter without taking the lock, so the unlock reports it was not held.
        async def _not_acquired(*args: Any) -> None:
            return None

        monkeypatch.setattr(
            ExtendedAsyncSAEngine, "_acquire_advisory_lock", staticmethod(_not_acquired)
        )
        invalidations: list[Any] = []
        sa.event.listen(
            database_connection.sync_engine,
            "invalidate",
            lambda *args: invalidations.append(args),
        )
        with caplog.at_level(logging.WARNING):
            async with database_connection.advisory_lock(LOCK):
                pass
        assert "was not held at release" in caplog.text
        assert len(invalidations) == 1


class TestPgAdvisoryLock:
    async def test_non_blocking_refusal_leaves_no_context(
        self, database_connection: ExtendedAsyncSAEngine
    ) -> None:
        async with PgAdvisoryLock(database_connection, LOCK):
            lock = PgAdvisoryLock(database_connection, LOCK, blocking=False)
            with pytest.raises(LockNotAcquired):
                async with lock:
                    pytest.fail("the body must not run")
            assert lock._lock_ctx is None
        async with PgAdvisoryLock(database_connection, LOCK, blocking=False):
            assert not await _is_free(database_connection, LOCK)
        assert await _is_free(database_connection, LOCK)
