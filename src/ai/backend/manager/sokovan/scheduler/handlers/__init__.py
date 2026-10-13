"""
Scheduler operation handlers.
"""

from .base import SessionLifecycleHandler
from .lifecycle.check_precondition import CheckPreconditionLifecycleHandler
from .lifecycle.deprioritize_sessions import DeprioritizeSessionsLifecycleHandler
from .lifecycle.preempt_sessions import PreemptSessionsLifecycleHandler
from .lifecycle.release_reserved_sessions import ReleaseReservedSessionsLifecycleHandler
from .lifecycle.reschedule_sessions import RescheduleSessionsLifecycleHandler
from .lifecycle.schedule_sessions import ScheduleSessionsLifecycleHandler
from .lifecycle.start_sessions import StartSessionsLifecycleHandler
from .lifecycle.terminate_sessions import TerminateSessionsLifecycleHandler
from .maintenance.expire_stuck_starts import ExpireStuckStartsLifecycleHandler
from .maintenance.sweep_sessions import SweepSessionsLifecycleHandler

__all__ = [
    # Session lifecycle handlers (SessionLifecycleHandler interface)
    "SessionLifecycleHandler",
    "CheckPreconditionLifecycleHandler",
    "DeprioritizeSessionsLifecycleHandler",
    "ExpireStuckStartsLifecycleHandler",
    "PreemptSessionsLifecycleHandler",
    "ReleaseReservedSessionsLifecycleHandler",
    "RescheduleSessionsLifecycleHandler",
    "ScheduleSessionsLifecycleHandler",
    "StartSessionsLifecycleHandler",
    "SweepSessionsLifecycleHandler",
    "TerminateSessionsLifecycleHandler",
]
