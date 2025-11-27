# commands/track.py
import logging
import threading
import time
from typing import Optional
from tracking.state import TrackingIntent
from .base import Command

logger = logging.getLogger(__name__)

class TrackCommand(Command):
    """Force track a specific ICAO with bulletproof preemption."""

    def execute(self, params: dict) -> bool:
        """
        Preempt any current tracking session and start tracking a specific target.

        Args:
            params: Must include ``track_icao`` (str). Additional keys are ignored.

        Returns:
            True when the scheduler is triggered to track the requested ICAO; False on
            invalid input or if scheduler setup fails.
        """
        icao = params.get('track_icao', '').lower().strip()
        if not icao:
            logger.warning("Track command missing valid ICAO")
            return False

        logger.info(f"=== MANUAL TRACK REQUEST: {icao.upper()} ===")
        
        old_thread: Optional[threading.Thread] = None
        was_tracking = False

        with self.context.state.lock:
            # Cancel scheduler timer
            self.context.state.cancel_scheduler_timer()

            # Mark preemption and signal the track thread to stop
            current_track = self.context.state.current_track_icao
            if current_track:
                was_tracking = True
                logger.info(f"  Preempting current track: {current_track}")
                self.context.state.mark_preempt(True)
                self.context.state.signal_track_stop()
                old_thread = self.context.active_thread

            # Set new target (under lock to prevent race with process_file)
            self.context.state.set_manual_target(icao)
            self.context.state.set_manual_viability(None)
            self.context.state.exit_monitor_mode()

        # === CRITICAL SECTION: Wait for old thread with fallback nuclear reset ===
        if old_thread and old_thread.is_alive():
            logger.info("  Waiting for old tracking thread to terminate...")
            old_thread.join(timeout=8.0)  # Longer than park, this is the important one

            if old_thread.is_alive():
                logger.error("  Old tracking thread DID NOT DIE within timeout!")
                # Assert force-stop interlock; do not clear stop flag automatically
                self.context._force_reset_tracking_state()
            else:
                logger.info("  Previous thread terminated cleanly.")

        # Final safety: ensure clean state before scheduling
        with self.context.state.lock:
            current_track = self.context.state.current_track_icao
            if current_track:
                logger.warning("  State inconsistency detected — forcing reset")
                self.context._force_reset_tracking_state()

        if self.context.state.is_force_stop_active():
            logger.error("  Force-stop asserted; refusing to clear stop or schedule new track.")
            return False

        # Clear the stop signal now that the previous thread is gone (unless global shutdown)
        self.context.state.apply_intent(intent=TrackingIntent.EXIT_MONITOR)
        self.context.state.clear_track_stop_if_running()

        # Trigger scheduler (will pick up manual_target_icao immediately)
        logger.info("  Launching new track via scheduler...")
        self.context._run_scheduler()

        return True
