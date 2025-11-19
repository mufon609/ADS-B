# commands/track.py
import logging
import threading
import time
from typing import Optional
from .base import Command

logger = logging.getLogger(__name__)

class TrackCommand(Command):
    """Force track a specific ICAO with bulletproof preemption."""

    def execute(self, params: dict) -> bool:
        icao = params.get('track_icao', '').lower().strip()
        if not icao:
            logger.warning("Track command missing valid ICAO")
            return False

        logger.info(f"=== MANUAL TRACK REQUEST: {icao.upper()} ===")
        
        old_thread: Optional[threading.Thread] = None
        was_tracking = False

        with self.context.track_lock:
            # Cancel scheduler timer
            if self.context._scheduler_timer and self.context._scheduler_timer.is_alive():
                self.context._scheduler_timer.cancel()
                self.context._scheduler_timer = None
                self.context.is_scheduler_waiting = False

            # Mark preemption and signal the track thread to stop
            if self.context.current_track_icao:
                was_tracking = True
                logger.info(f"  Preempting current track: {self.context.current_track_icao}")
                self.context.preempt_requested = True
                self.context.track_stop_event.set()
                old_thread = self.context.active_thread

            # Set new target (under lock to prevent race with process_file)
            self.context.manual_target_icao = icao
            self.context.manual_viability_info = None
            self.context.monitor_mode = False

        # === CRITICAL SECTION: Wait for old thread with fallback nuclear reset ===
        if old_thread and old_thread.is_alive():
            logger.info("  Waiting for old tracking thread to terminate...")
            old_thread.join(timeout=8.0)  # Longer than park, this is the important one

            if old_thread.is_alive():
                logger.error("  Old tracking thread DID NOT DIE within timeout!")
                # NUCLEAR OPTION: Force state reset (safe because shutdown_event is set)
                self.context._force_reset_tracking_state()
            else:
                logger.info("  Previous thread terminated cleanly.")

        # Final safety: ensure clean state before scheduling
        with self.context.track_lock:
            if self.context.current_track_icao:
                logger.warning("  State inconsistency detected — forcing reset")
                self.context._force_reset_tracking_state()

        # Clear the stop signal now that the previous thread is gone (unless global shutdown)
        if not self.context.shutdown_event.is_set():
            self.context.track_stop_event.clear()

        # Trigger scheduler (will pick up manual_target_icao immediately)
        logger.info("  Launching new track via scheduler...")
        self.context._run_scheduler()

        # Extra safety net — if somehow still not tracking in 2s, force another run
        threading.Timer(2.0, self.context._run_scheduler).start()

        return True
