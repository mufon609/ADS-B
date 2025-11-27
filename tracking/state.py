# tracking/state.py
"""
Defines the core, thread-safe data structures for managing tracking state.
- `TrackingState`: A centralized class holding all shared state.
- `TrackingSnapshot`: An immutable view of the state for safe consumption.
- `TrackingFSMState`, `TrackingIntent`: Enums for the state machine.
"""
import copy
import threading
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingSnapshot:
    """Immutable view of tracking state for safe, lock-free consumption."""
    mode: str
    fsm_state: "TrackingFSMState"
    current_track_icao: Optional[str]
    current_track_ev: float
    manual_target_icao: Optional[str]
    manual_viability: Optional[Dict[str, Any]]
    manual_override_active: Optional[str]
    preempt_requested: bool
    captures_taken: int
    monitor_mode: bool
    scheduler_waiting: bool
    force_stop_active: bool
    shutdown: bool
    stop_requested: bool
    active_timer_id: Optional[int]
    active_timer_alive: bool


class TrackingFSMState(Enum):
    IDLE = "idle"
    MONITOR = "monitor"
    ACQUIRING = "acquiring"
    TRACKING = "tracking"
    FORCE_STOPPED = "force_stopped"
    SHUTTING_DOWN = "shutting_down"


class TrackingIntent(Enum):
    BEGIN_TRACK = "begin_track"
    FINISH_TRACK = "finish_track"
    PREEMPT = "preempt"
    ENTER_MONITOR = "enter_monitor"
    EXIT_MONITOR = "exit_monitor"
    FORCE_STOP = "force_stop"
    CLEAR_FORCE_STOP = "clear_force_stop"
    SHUTDOWN = "shutdown"
    SCHEDULER_WAIT = "scheduler_wait"
    SCHEDULER_READY = "scheduler_ready"
    REQUEST_MANUAL_TARGET = "request_manual_target"
    ABORT = "abort"
    PARK = "park"


class TrackingState:
    """
    Holds shared tracking state and provides thread-safe accessors.
    This centralizes flags and counters that are shared across threads.
    """

    class StopHandle:
        """Lightweight wrapper exposing stop/shutdown signals without leaking raw events."""

        def __init__(self, owner: "TrackingState"):
            self._owner = owner

        def signal_stop(self):
            self._owner.signal_track_stop()

        def clear_stop_if_running(self):
            self._owner.clear_track_stop_if_running()

        def should_stop(self) -> bool:
            return self._owner.should_stop_tracking()

        def is_shutdown(self) -> bool:
            return self._owner.is_shutdown()

    def __init__(self):
        """Initializes all the tracking state variables."""
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        self.track_stop_event = threading.Event()

        # Core tracking state
        self.current_track_icao: Optional[str] = None
        self.current_track_ev: float = 0.0
        self.preempt_requested: bool = False
        self.manual_override_active: Optional[str] = None
        self.captures_taken: int = 0

        # Manual target selection
        self.manual_target_icao: Optional[str] = None
        self.manual_viability_info: Optional[Dict[str, Any]] = None

        # Force-stop interlock (distinct from shutdown)
        self.force_stop_active: bool = False

        # Scheduler state
        self.monitor_mode: bool = False
        self.is_scheduler_waiting: bool = False
        self._scheduler_timer: Optional[threading.Timer] = None
        self.fsm_state: TrackingFSMState = TrackingFSMState.IDLE

    # --- Stop / Shutdown Helpers ---

    def is_shutdown(self) -> bool:
        """Return True if a shutdown has been signaled."""
        return self.shutdown_event.is_set()

    def signal_shutdown(self):
        """Signal a global shutdown."""
        with self.lock:
            self.shutdown_event.set()
            self.fsm_state = TrackingFSMState.SHUTTING_DOWN

    def signal_track_stop(self):
        """Signal that the current track should stop."""
        with self.lock:
            self.track_stop_event.set()

    def clear_track_stop_if_running(self):
        """
        Clear the stop signal if we are not in shutdown.
        Used when resuming tracking or entering monitor mode.
        """
        with self.lock:
            if not self.shutdown_event.is_set() and not self.force_stop_active:
                self.track_stop_event.clear()

    def should_stop_tracking(self) -> bool:
        """Return True if a track stop has been signaled."""
        return self.track_stop_event.is_set()

    def get_stop_handle(self) -> "TrackingState.StopHandle":
        """Return a handle exposing stop/shutdown controls without raw event access."""
        return TrackingState.StopHandle(self)

    # --- Snapshot ---

    def get_snapshot(self) -> TrackingSnapshot:
        """
        Return an immutable copy of the current state.
        This avoids leaking internal objects (e.g., timers, dicts) to callers.
        """
        with self.lock:
            manual_viability_copy = copy.deepcopy(self.manual_viability_info)
            timer = self._scheduler_timer
            timer_id = id(timer) if timer else None
            timer_alive = bool(timer.is_alive()) if timer else False
            mode = "tracking" if self.current_track_icao else ("monitor" if self.monitor_mode else "idle")
            fsm_state = self.fsm_state
            return TrackingSnapshot(
                mode=mode,
                fsm_state=fsm_state,
                current_track_icao=self.current_track_icao,
                current_track_ev=self.current_track_ev,
                manual_target_icao=self.manual_target_icao,
                manual_viability=manual_viability_copy,
                manual_override_active=self.manual_override_active,
                preempt_requested=self.preempt_requested,
                captures_taken=self.captures_taken,
                monitor_mode=self.monitor_mode,
                scheduler_waiting=self.is_scheduler_waiting,
                force_stop_active=self.force_stop_active,
                shutdown=self.shutdown_event.is_set(),
                stop_requested=self.track_stop_event.is_set(),
                active_timer_id=timer_id,
                active_timer_alive=timer_alive,
            )

    # --- Track Lifecycle Helpers ---

    def begin_track(self, icao: str, ev: float, clear_manual_override: bool = True):
        """
        Atomically start tracking a target:
        - set current track/ev
        - clear preempt flag
        - optionally clear manual override (when not a manual request)
        - clear stop flag if safe
        - reset captures
        """
        with self.lock:
            self.current_track_icao = icao
            self.current_track_ev = ev
            self.preempt_requested = False
            if clear_manual_override:
                self.manual_override_active = None
            self.clear_track_stop_if_running()
            self.reset_captures()
            self.fsm_state = TrackingFSMState.ACQUIRING

    def finish_track(self, finished_icao: Optional[str] = None):
        """
        Atomically clear current track state and related flags.
        Clears manual override if it matches the finished track.
        """
        with self.lock:
            self.current_track_icao = None
            self.current_track_ev = 0.0
            self.preempt_requested = False
            if finished_icao and self.manual_override_active == finished_icao:
                self.manual_override_active = None
            # If we were tracking/acquiring, fall back to monitor/idle based on flag
            self.fsm_state = TrackingFSMState.MONITOR if self.monitor_mode else TrackingFSMState.IDLE
    # --- Capture Accessors ---

    def reset_captures(self) -> int:
        """Resets the capture counter to zero."""
        with self.lock:
            self.captures_taken = 0
            return self.captures_taken

    def increment_captures(self, n: int) -> int:
        """Increments the capture counter by n."""
        with self.lock:
            self.captures_taken += n
            return self.captures_taken

    def set_captures_if_current(self, icao: str, captures: int) -> int:
        """Sets the capture count only if the provided ICAO matches the current track."""
        with self.lock:
            if self.current_track_icao == icao:
                self.captures_taken = captures
            return self.captures_taken

    def get_captures(self) -> int:
        """Returns the current capture count."""
        with self.lock:
            return self.captures_taken

    def clear_current_track(self):
        """Clears the current tracking target and resets preemption."""
        with self.lock:
            self.current_track_icao = None
            self.current_track_ev = 0.0
            self.preempt_requested = False

    def get_current_track(self) -> Tuple[Optional[str], float]:
        """Returns the current tracking target and exposure value."""
        with self.lock:
            return self.current_track_icao, self.current_track_ev

    # --- Force Stop Interlock ---

    def force_stop_tracking(self, reason: Optional[str] = None):
        """
        Assert a force-stop interlock: block scheduling, keep the stop flag set,
        and clear per-track/manual state without marking global shutdown.
        """
        with self.lock:
            self.force_stop_active = True
            self.fsm_state = TrackingFSMState.FORCE_STOPPED
            logger.warning(
                "Force-stop interlock asserted%s.",
                f" (reason: {reason})" if reason else ""
            )
            self.cancel_scheduler_timer()
            self.clear_scheduler_waiting()
            self.clear_current_track()
            self.clear_manual_target()
            self.set_manual_override(None)
            self.set_preempt_requested(False)
            self.track_stop_event.set()
            self.set_scheduler_busy(False)
            self.reset_captures()

    def clear_force_stop(self):
        """Clear the force-stop interlock and the stop flag."""
        with self.lock:
            self.force_stop_active = False
            self.track_stop_event.clear()
            self.fsm_state = TrackingFSMState.IDLE if not self.monitor_mode else TrackingFSMState.MONITOR
            logger.warning("Force-stop interlock cleared; stop flag released.")

    def is_force_stop_active(self) -> bool:
        """Return True if the force-stop interlock is asserted."""
        with self.lock:
            return self.force_stop_active

    # --- Manual Override and Preemption ---

    def set_manual_override(self, icao: Optional[str]):
        """Sets or clears the manual override target."""
        with self.lock:
            self.manual_override_active = icao

    def set_preempt_requested(self, flag: bool):
        """Sets the preemption flag."""
        with self.lock:
            self.preempt_requested = flag

    def mark_preempt(self, flag: bool = True):
        """Intent helper to mark/unmark preemption."""
        self.set_preempt_requested(flag)

    # --- FSM Transition Wrapper ---

    def apply_intent(self, intent: TrackingIntent, **kwargs) -> TrackingFSMState:
        """
        Validate and apply an intent against the current FSM state.
        Delegates to existing intent methods; returns the new FSM state.
        """
        with self.lock:
            current = self.fsm_state

            # Guard global shutdown
            if current == TrackingFSMState.SHUTTING_DOWN and intent not in {
                TrackingIntent.SHUTDOWN,
                TrackingIntent.FINISH_TRACK,
            }:
                return current

            # Guard force-stop
            if current == TrackingFSMState.FORCE_STOPPED and intent not in {
                TrackingIntent.CLEAR_FORCE_STOP,
                TrackingIntent.SHUTDOWN,
            }:
                return current

            # Transition handling
            if intent == TrackingIntent.BEGIN_TRACK:
                icao = kwargs.get("icao")
                ev = kwargs.get("ev", 0.0)
                clear_manual_override = kwargs.get("clear_manual_override", True)
                if not icao:
                    return current
                if current in {
                    TrackingFSMState.IDLE,
                    TrackingFSMState.MONITOR,
                    TrackingFSMState.ACQUIRING,
                    TrackingFSMState.TRACKING,
                }:
                    self.begin_track(icao, ev, clear_manual_override=clear_manual_override)
                    # Transition to acquiring, actual tracking thread will set state to TRACKING
                    self.fsm_state = TrackingFSMState.ACQUIRING
                return self.fsm_state

            if intent == TrackingIntent.FINISH_TRACK:
                finished_icao = kwargs.get("icao")
                self.finish_track(finished_icao)
                return self.fsm_state

            if intent == TrackingIntent.PREEMPT:
                self.mark_preempt(True)
                return self.fsm_state

            if intent == TrackingIntent.ENTER_MONITOR:
                self.enter_monitor_mode()
                return self.fsm_state

            if intent == TrackingIntent.EXIT_MONITOR:
                self.exit_monitor_mode()
                return self.fsm_state

            if intent == TrackingIntent.FORCE_STOP:
                reason = kwargs.get("reason")
                self.force_stop_tracking(reason=reason)
                return self.fsm_state

            if intent == TrackingIntent.CLEAR_FORCE_STOP:
                self.clear_force_stop()
                return self.fsm_state

            if intent == TrackingIntent.SHUTDOWN:
                self.signal_shutdown()
                return self.fsm_state

            if intent == TrackingIntent.REQUEST_MANUAL_TARGET:
                icao = kwargs.get("icao")
                viability = kwargs.get("viability")
                if icao:
                    self.set_manual_target(icao)
                    self.set_manual_viability(viability)
                return self.fsm_state

            if intent in {TrackingIntent.ABORT, TrackingIntent.PARK}:
                # Both map to monitor/idle behavior; PARK adds hardware action elsewhere.
                self.enter_monitor_mode()
                # Track stop is signaled by callers as needed.
                return self.fsm_state

            # Scheduler intents are no-ops here; timers handled externally.
            return self.fsm_state

    def is_preempt_requested(self) -> bool:
        """Returns True if preemption has been requested."""
        with self.lock:
            return self.preempt_requested

    # --- Manual Target Helpers ---

    def set_manual_target(self, icao: Optional[str]):
        """Sets or clears the manual tracking target."""
        with self.lock:
            self.manual_target_icao = icao
            if icao is None:
                self.manual_viability_info = None

    def set_manual_viability(self, info: Optional[Dict[str, Any]]):
        """Sets the viability information for the manual target."""
        with self.lock:
            self.manual_viability_info = info

    def clear_manual_state(self):
        """Clear manual target/viability and manual override flags."""
        with self.lock:
            self.manual_target_icao = None
            self.manual_viability_info = None
            self.manual_override_active = None

    def clear_manual_target(self):
        """Clears the manual tracking target."""
        with self.lock:
            self.manual_target_icao = None
            self.manual_viability_info = None

    # --- Monitor Mode ---

    def enter_monitor_mode(self):
        """Enter monitor mode and clear manual/preempt flags."""
        with self.lock:
            self.monitor_mode = True
            self.preempt_requested = False
            self.manual_override_active = None
            self.manual_target_icao = None
            self.manual_viability_info = None
            self.fsm_state = TrackingFSMState.MONITOR

    def exit_monitor_mode(self):
        """Exit monitor mode."""
        with self.lock:
            self.monitor_mode = False
            if not self.current_track_icao:
                self.fsm_state = TrackingFSMState.IDLE

    # --- Scheduler Flags and Timer Helpers ---

    def mark_scheduler_waiting(self, timer: Optional[threading.Timer]):
        """
        Marks the scheduler as waiting and stores the timer object.
        Cancels any previously stored timer.
        """
        with self.lock:
            # Cancel previous timer if different
            if self._scheduler_timer and self._scheduler_timer is not timer:
                try:
                    self._scheduler_timer.cancel()
                except Exception:
                    pass
            self._scheduler_timer = timer
            self.is_scheduler_waiting = bool(timer)

    def clear_scheduler_waiting(self):
        """Clears the scheduler waiting flag and timer."""
        with self.lock:
            self.is_scheduler_waiting = False
            self._scheduler_timer = None

    def cancel_scheduler_timer(self):
        """Cancels the scheduler timer, if any."""
        with self.lock:
            if self._scheduler_timer:
                try:
                    self._scheduler_timer.cancel()
                except Exception:
                    pass
            self._scheduler_timer = None
            self.is_scheduler_waiting = False

    def has_live_scheduler_timer(self) -> bool:
        """Return True if a scheduler timer exists and is alive."""
        with self.lock:
            return bool(self._scheduler_timer and self._scheduler_timer.is_alive())
