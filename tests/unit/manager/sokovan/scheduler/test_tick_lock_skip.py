"""Scheduler ticks never wait for a lock another pass holds.

A real ``ScheduleCoordinator`` with the real handlers and the real
``create_lock_factory`` (``pg_advisory``) on a real PostgreSQL. The repository,
Valkey and the scheduling controller are stand-ins.

Proven here:

- While another pass holds a handler's lock, every tick (short or long cycle)
  returns False at once, without a waiting advisory lock, holding no more than
  one connection, and marks its schedule type again.
- Once the lock is free, a later tick runs the pass; a skipped sweep runs
  within a short-cycle interval of the release.
- Every schedule type whose handler takes a lock has a short-cycle task, so the
  re-mark of a skipped tick is consumed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Coroutine, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa

from ai.backend.common.typed_validators import HostPortPair as HostPortPairModel
from ai.backend.manager.defs import LockID
from ai.backend.manager.dependencies.domain.distributed_lock import create_lock_factory
from ai.backend.manager.models.base import pgsql_connect_opts
from ai.backend.manager.models.utils import ExtendedAsyncSAEngine
from ai.backend.manager.repositories.db.engine import create_async_engine
from ai.backend.manager.sokovan.scheduler.coordinator import ScheduleCoordinator
from ai.backend.manager.sokovan.scheduler.factory import (
    CoordinatorHandlers,
    CoordinatorHandlersArgs,
    create_coordinator_handlers,
)
from ai.backend.manager.sokovan.scheduler.scheduler import SchedulerComponents
from ai.backend.manager.sokovan.scheduler.types import ScheduleType
from ai.backend.manager.types import DistributedLockFactory
from ai.backend.testutils.bootstrap import postgres_container  # noqa: F401

TERMINATING_LOCK = LockID.LOCKID_SOKOVAN_TARGET_TERMINATING
TERMINATE = ScheduleType.TERMINATE
SWEEP_STALE_KERNELS = ScheduleType.SWEEP_STALE_KERNELS
LOCK_LIFETIME = 30.0
# A skipped tick returns well inside this.
SKIP_BOUND_SEC = 0.5
# Stand-in for the short-cycle interval (2 s in production).
SHORT_INTERVAL_SEC = 0.1


# =============================================================================
# Stand-ins
# =============================================================================


class _Marks:
    """The schedule marks Valkey keeps, as ``load_and_delete_schedule_mark`` sees them."""

    def __init__(self) -> None:
        self.marks: set[str] = set()
        self.always = False

    async def load_and_delete(self, schedule_type: str) -> bool:
        if self.always:
            return True
        present = schedule_type in self.marks
        self.marks.discard(schedule_type)
        return present

    async def mark(self, schedule_types: list[ScheduleType]) -> None:
        self.marks.update(st.value for st in schedule_types)


class _CheckoutCounter:
    """Tracks the peak number of connections checked out of an engine's pool."""

    def __init__(self, engine: ExtendedAsyncSAEngine) -> None:
        self.current = 0
        self.peak = 0
        sa.event.listen(engine.sync_engine, "checkout", self._on_checkout)
        sa.event.listen(engine.sync_engine, "checkin", self._on_checkin)

    def _on_checkout(self, *args: Any) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)

    def _on_checkin(self, *args: Any) -> None:
        self.current -= 1


@dataclass
class _Harness:
    coordinator: ScheduleCoordinator
    repository: AsyncMock
    scheduling_controller: AsyncMock
    metrics: MagicMock
    marks: _Marks
    lock_factory: DistributedLockFactory


def _lock_factory(engine: ExtendedAsyncSAEngine) -> DistributedLockFactory:
    config_provider = MagicMock()
    config_provider.config.manager.distributed_lock = "pg_advisory"
    config_provider.config.manager.ipc_base_path = Path("/tmp")
    config_provider.config.manager.id = "test-manager"
    return create_lock_factory(config_provider, engine, MagicMock())


def _handlers() -> CoordinatorHandlers:
    return create_coordinator_handlers(
        CoordinatorHandlersArgs(
            provisioner=MagicMock(),
            launcher=MagicMock(),
            terminator=MagicMock(),
            repository=AsyncMock(),
            valkey_schedule=AsyncMock(),
            scheduling_controller=AsyncMock(),
            fair_share_aggregator=MagicMock(),
            fair_share_calculator=MagicMock(),
            resource_usage_repository=AsyncMock(),
            fair_share_repository=AsyncMock(),
        )
    )


async def _tick(tick: Coroutine[Any, Any, bool]) -> bool:
    """Run one tick; fail, rather than hang, if it waits for the lock."""
    async with asyncio.timeout(5.0):
        return await tick


async def _advisory_waits(engine: ExtendedAsyncSAEngine) -> tuple[int, int]:
    """(advisory lock rows not granted, backends waiting on a lock)."""
    async with engine.connect() as conn:
        not_granted = await conn.exec_driver_sql(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND NOT granted;"
        )
        waiting = await conn.exec_driver_sql(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock';"
        )
        return int(not_granted.scalar_one()), int(waiting.scalar_one())


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def database_url(
    postgres_container: tuple[str, HostPortPairModel],  # noqa: F811
) -> str:
    _, addr = postgres_container
    return f"postgresql+asyncpg://postgres:develove@{addr.host}:{addr.port}/testing"


@pytest.fixture
async def db(database_url: str) -> AsyncIterator[ExtendedAsyncSAEngine]:
    """The manager's engine, configured as in production."""
    engine = create_async_engine(
        database_url,
        connect_args=pgsql_connect_opts,
        pool_size=8,
        max_overflow=64,
        pool_pre_ping=False,
        isolation_level="SERIALIZABLE",
    )
    yield engine
    await engine.dispose()


@pytest.fixture
async def other_db(database_url: str) -> AsyncIterator[ExtendedAsyncSAEngine]:
    """Another manager process's engine; also used to inspect the server."""
    engine = create_async_engine(database_url, connect_args=pgsql_connect_opts, pool_pre_ping=False)
    yield engine
    await engine.dispose()


@pytest.fixture
def harness(db: ExtendedAsyncSAEngine) -> Iterator[_Harness]:
    marks = _Marks()
    valkey_schedule = AsyncMock()
    valkey_schedule.load_and_delete_schedule_mark.side_effect = marks.load_and_delete
    scheduling_controller = AsyncMock()
    scheduling_controller.mark_scheduling_needed.side_effect = marks.mark
    repository = AsyncMock()
    repository.get_all_resource_groups.return_value = []
    config_provider = MagicMock()
    config_provider.config.manager.session_schedule_lock_lifetime = LOCK_LIFETIME
    lock_factory = _lock_factory(db)
    coordinator = ScheduleCoordinator(
        valkey_schedule=valkey_schedule,
        components=SchedulerComponents(
            provisioner=MagicMock(),
            launcher=MagicMock(),
            terminator=MagicMock(),
            repository=repository,
            config_provider=config_provider,
            hook_registry=MagicMock(),
        ),
        handlers=_handlers(),
        scheduling_controller=scheduling_controller,
        event_producer=AsyncMock(),
        lock_factory=lock_factory,
    )
    metrics = MagicMock()
    coordinator._operation_metrics = metrics
    yield _Harness(
        coordinator=coordinator,
        repository=repository,
        scheduling_controller=scheduling_controller,
        metrics=metrics,
        marks=marks,
        lock_factory=lock_factory,
    )


# =============================================================================
# Tests
# =============================================================================


class TestTickLockSkip:
    @pytest.mark.parametrize("holder", ["same-process", "other-process"])
    async def test_concurrent_ticks_skip_at_once_while_the_lock_is_held(
        self,
        db: ExtendedAsyncSAEngine,
        other_db: ExtendedAsyncSAEngine,
        harness: _Harness,
        holder: str,
    ) -> None:
        coordinator = harness.coordinator
        harness.marks.always = True
        counter = _CheckoutCounter(db)
        holder_factory = (
            harness.lock_factory if holder == "same-process" else _lock_factory(other_db)
        )
        tick_engine_baseline = 1 if holder == "same-process" else 0

        async with holder_factory(TERMINATING_LOCK, LOCK_LIFETIME):
            assert counter.current == tick_engine_baseline
            ticks = [
                tick
                for _ in range(5)
                for tick in (
                    coordinator.process_if_needed(TERMINATE),
                    coordinator.process_schedule(TERMINATE),
                    coordinator.process_if_needed(SWEEP_STALE_KERNELS),
                    coordinator.process_schedule(SWEEP_STALE_KERNELS),
                )
            ]
            assert len(ticks) == 20
            started = time.perf_counter()
            async with asyncio.timeout(5.0):  # fail, rather than hang, if a tick waits
                results = await asyncio.gather(*ticks)
            elapsed = time.perf_counter() - started

            assert results == [False] * 20
            assert elapsed < SKIP_BOUND_SEC
            # At most one tick checked out a connection to try the lock.
            assert counter.peak == 1
            assert await _advisory_waits(other_db) == (0, 0)

        # Each skip marked its own type again, once.
        calls = harness.scheduling_controller.mark_scheduling_needed.await_args_list
        assert all(len(call.args[0]) == 1 for call in calls)
        assert Counter(call.args[0][0] for call in calls) == {
            TERMINATE: 10,
            SWEEP_STALE_KERNELS: 10,
        }
        skipped = [
            call
            for call in harness.metrics.observe_operation.call_args_list
            if call.kwargs["status"] == "skipped"
        ]
        assert Counter(call.kwargs["operation"] for call in skipped) == {
            "terminate-sessions": 10,
            "sweep-stale-kernels": 10,
        }
        # The lock is taken before the pass is measured or run.
        harness.metrics.measure_operation.assert_not_called()
        harness.repository.get_all_resource_groups.assert_not_awaited()
        assert db._held_advisory_locks == set()

        # A later tick runs the pass.
        assert await _tick(coordinator.process_if_needed(TERMINATE)) is True
        assert await _tick(coordinator.process_if_needed(SWEEP_STALE_KERNELS)) is True
        assert harness.repository.get_all_resource_groups.await_count == 2
        assert harness.metrics.measure_operation.call_count == 2

    async def test_skipped_sweep_runs_within_a_short_interval_of_the_release(
        self, harness: _Harness
    ) -> None:
        coordinator = harness.coordinator
        runs: list[float] = []
        skips = 0

        async def short_cycle_timer() -> None:
            nonlocal skips
            while True:
                if await _tick(coordinator.process_if_needed(SWEEP_STALE_KERNELS)):
                    runs.append(time.monotonic())
                elif SWEEP_STALE_KERNELS.value in harness.marks.marks:
                    skips += 1
                await asyncio.sleep(SHORT_INTERVAL_SEC)

        await harness.marks.mark([SWEEP_STALE_KERNELS])
        async with harness.lock_factory(TERMINATING_LOCK, LOCK_LIFETIME):
            timer = asyncio.create_task(short_cycle_timer())
            await asyncio.sleep(0.5)
            assert runs == []
            assert skips >= 2
        released_at = time.monotonic()
        try:
            async with asyncio.timeout(3.0):
                while not runs:
                    await asyncio.sleep(0.01)
            # Let a few more ticks pass: the consumed mark runs the pass once.
            await asyncio.sleep(3 * SHORT_INTERVAL_SEC)
        finally:
            timer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await timer
        assert len(runs) == 1
        assert runs[0] - released_at < 2 * SHORT_INTERVAL_SEC + SKIP_BOUND_SEC

    async def test_a_failed_re_mark_is_logged_and_never_raised(
        self, harness: _Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        harness.scheduling_controller.mark_scheduling_needed.side_effect = RuntimeError("valkey")
        async with harness.lock_factory(TERMINATING_LOCK, LOCK_LIFETIME):
            with caplog.at_level(logging.WARNING):
                assert await _tick(harness.coordinator.process_schedule(TERMINATE)) is False
        assert any(
            record.levelno == logging.WARNING
            and "Failed to request terminate again" in record.getMessage()
            for record in caplog.records
        )

    async def test_skip_is_logged_at_debug_for_short_and_info_for_long_ticks(
        self, harness: _Harness, caplog: pytest.LogCaptureFixture
    ) -> None:
        harness.marks.always = True
        async with harness.lock_factory(TERMINATING_LOCK, LOCK_LIFETIME):
            with caplog.at_level(logging.DEBUG):
                assert await _tick(harness.coordinator.process_if_needed(TERMINATE)) is False
                assert await _tick(harness.coordinator.process_schedule(TERMINATE)) is False
        levels = [
            record.levelno
            for record in caplog.records
            if "Skipped terminate" in record.getMessage()
        ]
        assert levels == [logging.DEBUG, logging.INFO]

    async def test_a_short_tick_without_a_mark_does_not_try_the_lock(
        self, db: ExtendedAsyncSAEngine, harness: _Harness
    ) -> None:
        counter = _CheckoutCounter(db)
        assert await _tick(harness.coordinator.process_if_needed(TERMINATE)) is False
        assert counter.peak == 0
        harness.scheduling_controller.mark_scheduling_needed.assert_not_awaited()


class TestTaskSpecs:
    def test_every_locked_schedule_type_has_a_short_cycle_task(self) -> None:
        handlers = _handlers()
        locked = {
            schedule_type
            for mapping in (handlers.lifecycle_handlers, handlers.kernel_handlers)
            for schedule_type, handler in mapping.items()
            if handler.lock_id is not None
        }
        assert {TERMINATE, SWEEP_STALE_KERNELS} <= locked
        specs = {spec.schedule_type: spec for spec in ScheduleCoordinator._create_task_specs()}
        missing = sorted(st.value for st in locked if specs[st].short_interval is None)
        assert missing == []

    def test_sweep_stale_kernels_has_a_short_cycle_task(self) -> None:
        specs = {spec.schedule_type: spec for spec in ScheduleCoordinator._create_task_specs()}
        assert specs[SWEEP_STALE_KERNELS].short_interval == 2.0
