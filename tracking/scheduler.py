import logging
import threading
import time

from adsb.data_reader import read_aircraft_data
from adsb.selector import calculate_expected_value, evaluate_manual_target_viability, select_aircraft
from config.loader import CONFIG
from tracking.session import track_aircraft
from tracking.state import TrackingIntent, TrackingFSMState
from utils.status_writer import write_status

logger = logging.getLogger(__name__)


def run_scheduler(handler, candidates=None):
    """
    Decides whether to start a new track or wait.
    Skips scheduling when monitor mode is active (unless manual override), during shutdown,
    or when a track is already active; respects manual targets and preemption.
    """
    with handler.state.lock:
        if handler.state.is_force_stop_active():
            logger.warning("  Scheduler: Force-stop interlock active; skipping scheduling.")
            return
        snapshot = handler.state.get_snapshot()
        try:
            # Do not schedule new tracks during shutdown
            if snapshot.shutdown or snapshot.scheduler_waiting:
                return
            if snapshot.current_track_icao:
                return
            manual_icao = snapshot.manual_target_icao
            is_manual_override = bool(manual_icao)
            if snapshot.monitor_mode and not is_manual_override:
                logger.debug("  Scheduler: Monitor mode active; skipping auto-tracking.")
                return

            if not handler.indi_controller:
                logger.warning(
                    "  Scheduler: Cannot run, hardware controller not ready.")
                return
            hs = handler.indi_controller.get_hardware_status() or {}

            if candidates is None:
                aircraft_dict = read_aircraft_data()
                current_az_el = hs.get("mount_az_el", (0.0, 0.0))
                try:
                    candidates = select_aircraft(
                        aircraft_dict, current_az_el) or []
                except Exception as e:
                    logger.error(f"  Scheduler: Error selecting aircraft: {e}")
                    candidates = []
            else:
                aircraft_dict = read_aircraft_data()

            target_to_consider = None
            is_manual_override = False

            if manual_icao:
                icao = manual_icao
                logger.info(f"  Scheduler: Evaluating manual target {icao}...")
                manual_target_in_candidates = next(
                    (c for c in candidates if c['icao'] == icao), None)

                if manual_target_in_candidates:
                    logger.info(
                        "  Manual target %s is now viable (EV: %.1f). "
                        "Selecting.",
                        icao,
                        manual_target_in_candidates['ev']
                    )
                    target_to_consider = manual_target_in_candidates
                    is_manual_override = True
                    handler.state.set_manual_viability(None)
                    handler.state.set_manual_override(icao)
                    handler.state.clear_manual_state()
                else:
                    logger.info(
                        f"  Manual target {icao} not in top candidates. Checking viability...")
                    ok, reasons, details = evaluate_manual_target_viability(
                        icao, aircraft_dict, observer_loc=handler.observer_loc
                    )
                    ev_reason_text = None
                    ev_val = None
                    if ok and icao in aircraft_dict:
                        try:
                            current_az_el_ev = hs.get(
                                "mount_az_el", (0.0, 0.0))  # Use hs here
                            ev_res = calculate_expected_value(
                                current_az_el_ev, icao, aircraft_dict[icao])
                            ev_val = float(ev_res.get('ev', 0.0))
                            if ev_val <= 0.0:
                                ev_reason = ev_res.get(
                                    'reason', 'unknown_ev_reason')
                                reason_map = {
                                    'no_intercept': "cannot slew to intercept within limits",
                                    'late_intercept': "intercept occurs too late (outside horizon)",
                                    'low_quality': "predicted quality below minimum during track",
                                    'no_prediction': "cannot predict flight path",
                                    'prediction_failed': "prediction calculation error"
                                }
                                ev_reason_text = reason_map.get(
                                    ev_reason, f"EV too low ({ev_val:.2f})")
                                if ev_reason_text not in reasons:
                                    reasons.append(ev_reason_text)
                        except Exception as e:
                            logger.warning(
                                f"  Warning: EV calculation failed for manual target {icao}: {e}")
                            reasons.append("EV calculation failed")

                    viability_info = {
                        "viable": bool(ok and ev_val is not None and ev_val > 0.0),
                        "reasons": reasons, "details": details, "ev": ev_val,
                    }
                    handler.state.set_manual_viability(viability_info)
                    reason_str = "; ".join(
                        reasons) if reasons else "basic checks passed, but EV <= 0 or not calculated"
                    retry_s = float(CONFIG.get('selection', {}).get(
                        'manual_retry_interval_s', 5.0))
                    logger.info(
                        f"  Manual target {icao} rejected: {reason_str}. Retrying in {retry_s}s.")
                    write_status(
                        {"manual_target": {"icao": icao, "viability": viability_info}})

                    if not handler.state.has_live_scheduler_timer():
                        logger.info(
                            f"  Scheduling retry for manual target in {retry_s}s.")
                        timer = threading.Timer(
                            retry_s, handler._run_scheduler_callback)  # Use callback
                        timer.daemon = True
                        handler.state.mark_scheduler_waiting(timer)
                        timer.start()
                    return

            if not target_to_consider:
                if candidates:
                    target_to_consider = candidates[0]
                else:
                    return

            now = time.time()
            has_intercept_time = 'intercept_time' in target_to_consider
            has_start_state = 'start_state' in target_to_consider
            has_start_state_time = (has_start_state
                                    and 'time' in target_to_consider['start_state'])

            if not (has_intercept_time and has_start_state and has_start_state_time):
                logger.warning(
                    f"  Scheduler: Target {target_to_consider['icao']} missing required timing data. Skipping.")
                return

            intercept_duration = target_to_consider['intercept_time']
            track_start_time_abs = target_to_consider['start_state']['time']
            slew_start_time_abs = track_start_time_abs - intercept_duration - 5.0
            delay_needed = slew_start_time_abs - now

            if delay_needed > 1.0:
                wait_duration = min(delay_needed, 30.0)
                handler.state.cancel_scheduler_timer()
                timer = threading.Timer(
                    wait_duration, handler._run_scheduler_callback)  # Use callback
                timer.daemon = True
                handler.state.mark_scheduler_waiting(timer)
                timer.start()
                logger.info(
                    "  Scheduler: Waiting %.1fs (of %.1fs total) to start slew for %s.",
                    wait_duration,
                    delay_needed,
                    target_to_consider['icao'],
                )
                return

            # --- Start Tracking ---
            handler.state.cancel_scheduler_timer()
            handler.state.clear_scheduler_waiting()

            icao_to_track = target_to_consider['icao']
            latest_data = read_aircraft_data().get(icao_to_track)
            if not latest_data:
                logger.info(
                    f"  Scheduler: No current data for {icao_to_track} just before starting track; deferring.")
                if not handler.state.has_live_scheduler_timer():
                    handler.state.clear_scheduler_waiting()
                    timer = threading.Timer(
                        2.0, handler._run_scheduler_callback)  # Retry in 2s, use callback
                    timer.daemon = True
                    handler.state.mark_scheduler_waiting(timer)
                    timer.start()
                return

            handler.state.apply_intent(
                intent=TrackingIntent.BEGIN_TRACK,
                icao=icao_to_track,
                ev=target_to_consider['ev'],
                clear_manual_override=not is_manual_override,
            )

            write_status({"mode": "acquiring", "icao": icao_to_track})
            
            # --- ENHANCED LOGGING ---
            ev_val = target_to_consider.get('ev', 0.0)
            details = target_to_consider.get('score_details', {})
            d_contrib = details.get('contrib_dist', 0.0)
            c_contrib = details.get('contrib_close', 0.0)
            i_contrib = details.get('contrib_int', 0.0)
            raw_dist = details.get('raw_range_km', 0.0)
            raw_rate = details.get('raw_closure_ms', 0.0)
            raw_slew = details.get('raw_int_s', 0.0)
            
            logger.info(
                f"  Scheduler: Starting track for {icao_to_track} (Total EV: {ev_val:.3f})\n"
                f"    > Score Breakdown: Dist: {d_contrib:.2f} + Close: "
                f"{c_contrib:.2f} + Slew: {i_contrib:.2f}\n"
                f"    > Flight Stats:    {raw_dist:.1f}km Range | "
                f"{raw_rate:+.0f}m/s Rate | {raw_slew:.1f}s Slew"
            )

            start_state_info = target_to_consider['start_state'].copy()
            start_state_info['slew_duration_s'] = intercept_duration

            handler.active_thread = threading.Thread(
                target=track_aircraft,
                args=(handler, icao_to_track, latest_data, start_state_info, hs),
                daemon=True
            )
            handler.active_thread.start()
        finally:
            pass


def scheduler_callback(handler):
    """Callback function used by the scheduler's timer."""
    with handler.state.lock:
        # If shutdown has been signaled, do not invoke the scheduler again
        handler.state.clear_scheduler_waiting()
        snapshot = handler.state.get_snapshot()
        if snapshot.force_stop_active:
            logger.warning("  Scheduler callback: Force-stop interlock active; skipping.")
            return
        if snapshot.shutdown:
            return
        # Now call the actual scheduler
        handler._run_scheduler()
