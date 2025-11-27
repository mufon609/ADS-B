import logging
import math
import os
import time
import numpy as np

from adsb.data_reader import read_aircraft_data
from adsb.dead_reckoning import estimate_positions_at_times
from astro.coords import (
    angular_sep_deg,
    angular_speed_deg_s,
    calculate_plate_scale,
    distance_km,
    get_altaz_frame,
    get_sun_azel,
    latlonalt_to_azel,
)
from config.loader import CONFIG
from imaging.analysis import (
    _detect_aircraft_from_data,
    _estimate_exposure_adjustment_from_data,
    _load_fits_data,
    _save_png_preview_from_data,
)
from utils.status_writer import write_status
from utils.tracking_utils import is_target_viable
from utils.capture_types import CaptureJob
from utils.image_utils import to_img_url
from tracking.utils import predict_target_az_el
from tracking.state import TrackingFSMState

logger = logging.getLogger(__name__)


def track_aircraft(handler, icao: str, aircraft_data: dict, start_state: dict, hardware_status: dict):
    """
    Main tracking and guiding loop for a single aircraft.

    This method is executed in a dedicated thread for each tracking session.
    It performs the initial slew, autofocuses, and then enters a continuous
    loop of capturing guide frames, detecting the aircraft's position,
    sending guide pulses to the mount, and capturing science frames when
    the guiding is stable.

    Args:
        handler: The FileHandler instance managing tracking state.
        icao: The ICAO identifier of the target aircraft.
        aircraft_data: The most recent ADS-B data for the target.
        start_state: Information about the calculated intercept point,
                     including target Az/El and slew time.
        hardware_status: The hardware status at the start of the track.
    """
    try:
        if handler.state.should_stop_tracking():
            logger.info(
                f"  Track [{icao}]: Aborted immediately before slew.")
            return

        target_az = start_state.get('az')
        target_el = start_state.get('el')
        if target_az is None or target_el is None:
            logger.error(
                f"  Track [{icao}]: Invalid start state coordinates. Aborting.")
            return

        target_az_el = (target_az, target_el)
        write_status(
            {"mode": "slewing", "target_az_el": list(target_az_el), "icao": icao})
        logger.info(
            f"  Track [{icao}]: Slewing to intercept point ({target_az:.2f}, {target_el:.2f})...")

        def _slew_progress(cur_az: float, cur_el: float, state: str):
            live_target_az_el = predict_target_az_el(
                aircraft_data, handler.observer_loc, when=time.time()) or target_az_el
            payload = {
                "mode": "slewing", "icao": icao,
                "mount_az_el": [float(cur_az), float(cur_el)],
                "target_az_el": list(live_target_az_el),
            }
            write_status(payload)

        snapshot = handler.state.get_snapshot()
        manual_icao = snapshot.manual_target_icao
        manual_viability = snapshot.manual_viability

        if not handler.indi_controller.slew_to_az_el(target_az, target_el, progress_cb=_slew_progress):
            logger.error(
                f"  Track [{icao}]: Initial slew failed or aborted. Ending track.")
            current_status = {"mode": "idle", "icao": None}
            if manual_icao == icao:
                current_status["manual_target"] = {
                    "icao": icao,
                    "viability": manual_viability or {"viable": False, "reasons": ["slew failed"]},
                }
            else:
                current_status["manual_target"] = None
            write_status(current_status)
            return

        if handler.state.should_stop_tracking():
            logger.info(
                f"  Track [{icao}]: Aborted after slew completion.")
            return

        current_alt_ft = aircraft_data.get('alt')
        write_status(
            {"mode": "focusing", "autofocus_alt_ft": current_alt_ft, "icao": icao})
        logger.info(f"  Track [{icao}]: Performing autofocus...")
        if not handler.indi_controller.autofocus(current_alt_ft):
            logger.error(
                f"  Track [{icao}]: Autofocus failed. Ending track.")
            write_status({"mode": "idle", "icao": None,
                         "error_message": "Autofocus failed"})
            return

        # Mark FSM as actively tracking after successful slew+focus.
        handler.state.fsm_state = TrackingFSMState.TRACKING

        if handler.state.should_stop_tracking():
            logger.info(
                f"  Track [{icao}]: Aborted after focus completion.")
            return

        logger.info(f"  Track [{icao}]: Starting optical guiding loop.")
        guide_cfg = CONFIG['capture'].get('guiding', {})
        calib_cfg = CONFIG.get('pointing_calibration', {})
        cam_specs = CONFIG.get('camera_specs', {})
        frame_w = cam_specs.get('resolution_width_px', 640)
        frame_h = cam_specs.get('resolution_height_px', 480)
        frame_center = (frame_w / 2.0, frame_h / 2.0)
        frame = get_altaz_frame(handler.observer_loc)
        plate_scale = calculate_plate_scale()

        simulated_detection_result = None
        if CONFIG['development']['dry_run']:
            slew_duration_s = start_state.get('slew_duration_s', 30.0)
            timing_error_s = abs(start_state.get(
                'time', time.time()) - (time.time() + slew_duration_s)) * 0.5
            if timing_error_s > 0:
                # This logic is moved from analysis._detect_aircraft_from_data
                # to avoid calling it with a null path. We can create a synthetic
                # detection result without needing to load a fake image first.
                error_px = float(timing_error_s) * 20.0
                max_offset = 150.0
                dx = min(
                    error_px * np.random.choice([-1, 1]), max_offset * np.sign(error_px))
                dy = 0
                cx = frame_center[0] + dx
                cy = frame_center[1] + dy
                cx = max(0, min(frame_w - 1, cx))
                cy = max(0, min(frame_h - 1, cy))
                simulated_detection_result = {
                    'detected': True,
                    'center_px': (cx, cy),
                    'confidence': 0.95,
                    'sharpness': 100.0  # Use a plausible dummy sharpness
                }

        # Initialize the local capture counter from the shared state.
        captures_taken = handler.get_captures()
        consecutive_losses = 0
        iteration = 0
        last_seq_ts = 0.0
        max_losses = int(guide_cfg.get('max_consecutive_losses', 5))
        deadzone_px = float(guide_cfg.get('deadzone_px', 5.0))
        settle_s = float(guide_cfg.get('settle_time_s', 0.5))
        save_guide_png = guide_cfg.get('save_guide_png', True)
        min_sequence_interval_s = float(
            CONFIG['capture'].get('min_sequence_interval_s', 1.0))

        # --- Main Guide Loop ---
        while True:
            try:
                latest_full_dict = read_aircraft_data()
                if icao in latest_full_dict:
                    latest_data = latest_full_dict[icao]
                    update_keys = ['lat', 'lon', 'gs', 'track',
                        'alt', 'vert_rate', 'timestamp', 'age_s']
                    for k in update_keys:
                        if k in latest_data:
                            aircraft_data[k] = latest_data[k]
                else:
                    logger.info(
                        f"  Track [{icao}]: Target lost from ADS-B feed. Ending track.")
                    break
                current_age = aircraft_data.get('age_s', float('inf'))
                if current_age > 60.0:
                    logger.warning(
                        f"  Track [{icao}]: ADS-B data is too old ({current_age:.1f}s > 60s). Aborting track.")
                    break
            except Exception as e:
                logger.warning(
                    f"  Track [{icao}]: Error refreshing ADS-B data: {e}. Continuing with previous data.")

            iteration += 1
            loop_start_time = time.time()

            if handler.state.should_stop_tracking():
                logger.info(
                    f"  Track [{icao}]: Shutdown signaled, exiting guide loop.")
                break

            guide_path = None
            guide_data = None
            # In dry run, we still need to "create" the guide image file path and data
            # so that it can be loaded for analysis (sharpness, etc.)
            if CONFIG['development']['dry_run']:
                try:
                    # Call snap_image, which creates the flat+text FITS
                    guide_path = handler.indi_controller.snap_image(
                        f"{icao}_g{iteration}")
                    if guide_path and os.path.exists(guide_path):
                        # Load the FITS file we just created
                        guide_data = _load_fits_data(guide_path)
                    else:
                        # Fallback if snap_image fails or file not found
                        logger.warning(
                            f"  Track [{icao}]: [DRY RUN] snap_image failed or file not found.")
                        sim_h = CONFIG.get('camera_specs', {}).get(
                            'resolution_height_px', 480)
                        sim_w = CONFIG.get('camera_specs', {}).get(
                            'resolution_width_px', 640)
                        # Simple flat fallback
                        guide_data = np.full(
                            (sim_h, sim_w), 1000, np.float32)
                    if guide_data is None:  # Final fallback
                        guide_data = np.full((480, 640), 1000, np.float32)
                except Exception as e:
                    logger.error(
                        f"  Track [{icao}]: [DRY RUN] Error simulating guide snap: {e}")
                    sim_h = CONFIG.get('camera_specs', {}).get(
                        'resolution_height_px', 480)
                    sim_w = CONFIG.get('camera_specs', {}).get(
                        'resolution_width_px', 640)
                    guide_data = np.full((sim_h, sim_w), 1000, np.float32)
            else:
                # --- Real Hardware Capture ---
                try:
                    guide_path = handler.indi_controller.snap_image(
                        f"{icao}_g{iteration}")
                    if not guide_path:
                        raise RuntimeError(
                            "snap_image returned None or empty path")
                except Exception as e:
                    logger.error(
                        f"  Track [{icao}]: Guide frame {iteration} capture failed: {e}")
                    consecutive_losses += 1
                    if consecutive_losses >= max_losses:
                        break
                    time.sleep(1.0)
                    continue
                guide_data = _load_fits_data(guide_path)
                if guide_data is None:
                    logger.warning(
                        f"  Track [{icao}]: Failed to load guide image data for {guide_path}. Skipping frame.")
                    consecutive_losses += 1
                    if consecutive_losses >= max_losses:
                        break
                    time.sleep(0.5)
                    continue

            if guide_data is None:  # Handle case where guide_data failed to load/simulate
                logger.warning(
                    f"  Track [{icao}]: guide_data is None. Skipping frame.")
                consecutive_losses += 1
                if consecutive_losses >= max_losses:
                    break
                time.sleep(0.5)
                continue

            guide_shape = guide_data.shape

            guide_png_url = ""
            guide_png_path_generated = ""
            if save_guide_png and guide_path:
                try:
                    guide_png_path_expected = os.path.splitext(guide_path)[
                                                       0] + ".png"
                    saved_png_path = _save_png_preview_from_data(
                        guide_data, guide_png_path_expected)
                    if saved_png_path and os.path.exists(saved_png_path):
                        guide_png_path_generated = saved_png_path
                        guide_png_url = to_img_url(
                            guide_png_path_generated)
                    elif not CONFIG['development']['dry_run']:
                        logger.warning(
                            f"  Track [{icao}]: Warning - _save_png_preview_from_data did not return valid path.")
                except Exception as e:
                    logger.warning(
                        f"  Track [{icao}]: Warning - Error occurred during PNG creation/check: {e}")

            # Use the simulated result on the first iteration, then do real detection
            if iteration == 1 and simulated_detection_result:
                detection = simulated_detection_result
            else:
                detection = _detect_aircraft_from_data(
                    guide_data, original_shape=guide_shape)

            if not detection or not detection.get('detected') or not detection.get('center_px'):
                consecutive_losses += 1
                reason = (detection or {}).get('reason', 'unknown')
                sharp = detection.get('sharpness', -1)
                conf = detection.get('confidence', -1)
                logger.info(
                    "  Track [%s]: Guide frame %d: Target lost (%d/%d). "
                    "Reason: %s (Sharp: %.1f, Conf: %.2f)",
                    icao,
                    iteration,
                    consecutive_losses,
                    max_losses,
                    reason,
                    sharp,
                    conf,
                )
                write_status({"mode": "tracking", "icao": icao, "iteration": iteration,
                             "stable": False, "last_guide_png": guide_png_url, "guide_offset_px": None})
                if consecutive_losses >= max_losses:
                    break
                time.sleep(1.0);
                continue

            consecutive_losses = 0
            center_px = detection['center_px']
            if not (len(center_px) == 2 and math.isfinite(center_px[0]) and math.isfinite(center_px[1])):
                logger.warning(
                    f"  Track [{icao}]: Invalid centroid coordinates received: {center_px}. Treating as lost.")
                consecutive_losses += 1
                write_status({"mode": "tracking", "icao": icao, "iteration": iteration,
                             "stable": False, "last_guide_png": guide_png_url})
                time.sleep(0.5)
                continue

            offset_px_x = center_px[0] - frame_center[0]
            offset_px_y = center_px[1] - frame_center[1]
            rotation_rad = math.radians(
                calib_cfg.get('rotation_angle_deg', 0.0))
            if abs(rotation_rad) > 1e-3:
                cos_rot, sin_rot = math.cos(
                    rotation_rad), math.sin(rotation_rad)
                corrected_dx = offset_px_x * cos_rot - offset_px_y * sin_rot
                corrected_dy = offset_px_x * sin_rot + offset_px_y * cos_rot
            else:
                corrected_dx = offset_px_x
                corrected_dy = offset_px_y

            guide_error_mag = max(abs(corrected_dx), abs(corrected_dy))
            is_stable_np = guide_error_mag <= deadzone_px
            is_stable = bool(is_stable_np)

            # In dry run, we can now simulate the mount correcting the error
            if CONFIG['development']['dry_run']:
                # Simulate the mount correcting the error over time
                new_offset_x = offset_px_x * 0.3 + \
                    (np.random.rand() - 0.5) * 2.0
                new_offset_y = offset_px_y * 0.3 + \
                    (np.random.rand() - 0.5) * 2.0
                # Update the 'simulated_detection_result' for the *next* iteration
                simulated_detection_result = detection.copy()
                simulated_detection_result['center_px'] = (
                    frame_center[0] + new_offset_x, frame_center[1] + new_offset_y)
                # The first frame might not be stable, but subsequent ones should be
                if iteration > 1:
                    is_stable = True

            now = time.time()
            try:
                sun_az, sun_el = get_sun_azel(now, handler.observer_loc)
                est_list = estimate_positions_at_times(
                    aircraft_data, [now, now + 1.0])
                if len(est_list) < 2:
                    logger.error(
                        f"  Track [{icao}]: Could not predict motion for quality check. Ending track.")
                    break
                current_pos_est, next_sec_pos_est = est_list[0], est_list[1]
                current_az, current_el = latlonalt_to_azel(
                    current_pos_est['est_lat'],
                    current_pos_est['est_lon'],
                    current_pos_est['est_alt'],
                    now,
                    handler.observer_loc,
                )
                next_sec_az, next_sec_el = latlonalt_to_azel(
                    next_sec_pos_est['est_lat'],
                    next_sec_pos_est['est_lon'],
                    next_sec_pos_est['est_alt'],
                    now + 1.0,
                    handler.observer_loc,
                )
                ang_speed = angular_speed_deg_s(
                    (current_az, current_el), (next_sec_az, next_sec_el), 1.0, frame)
                # === NEW HARD SAFETY FILTERS (matches the hybrid selector exactly) ===
                current_range_km = distance_km(
                    handler.observer_loc.lat.deg, handler.observer_loc.lon.deg,
                    current_pos_est['est_lat'], current_pos_est['est_lon'])
                current_sun_sep = angular_sep_deg(
                    (current_az, current_el), (sun_az, sun_el), frame)

                viable, reasons = is_target_viable(
                    az=current_az,
                    el=current_el,
                    range_km=current_range_km,
                    sun_az=sun_az,
                    sun_el=sun_el,
                    frame=frame,
                )

                if not viable:
                    logger.info(
                        " Track [%s]: Safety filter violated during track (%s). Ending track.",
                        icao,
                        reasons[0] if reasons else 'unknown_reason',
                    )
                    break
            except Exception as e:
                logger.error(
                    " Track [%s]: Error during live safety checks / prediction: %s. "
                    "Ending track.",
                    icao,
                    e,
                    exc_info=True,
                )
                break

            status_payload = {
                "mode": "tracking",
                "icao": icao,
                "iteration": iteration,
                "guide_offset_px": {"dx": float(corrected_dx), "dy": float(corrected_dy)},
                "stable": is_stable,
                "last_guide_png": guide_png_url,
                "last_guide_file": os.path.basename(guide_path) if guide_path else "N/A",
                "target_details": {
                    "lat": current_pos_est.get('est_lat'),
                    "lon": current_pos_est.get('est_lon'),
                    "alt": current_pos_est.get('est_alt'),
                    "flight": aircraft_data.get('flight', '').strip() or 'N/A',
                    "gs": aircraft_data.get('gs'),
                    "track": aircraft_data.get('track'),
                },
                "sun_pos": {"az": sun_az, "el": sun_el},
                "sun_sep_target_deg": current_sun_sep,
                "sun_sep_mount_deg": angular_sep_deg(
                    hardware_status.get(
                        "mount_az_el", (0, 0)), (sun_az, sun_el), frame
                ),
                "image_vitals": {
                    "sharpness": detection.get('sharpness'),
                    "confidence": detection.get('confidence'),
                },
                "current_range_km": round(current_range_km, 1) if np.isfinite(current_range_km) else None,
                "captures_taken": captures_taken,
            }
            status_payload.update(hardware_status)
            write_status(status_payload)

            pulse_sent = False
            az_sign = calib_cfg.get('az_offset_sign', 1)
            el_sign = calib_cfg.get('el_offset_sign', 1)

            guide_params = CONFIG['capture'].get('guiding', {})
            gain = float(guide_params.get('proportional_gain', 20.0))
            min_p = int(guide_params.get('min_pulse_ms', 10))
            max_p = int(guide_params.get('max_pulse_ms', 500))

            def calculate_pulse(error_px):
                duration = int(abs(error_px) * gain)
                return max(min_p, min(max_p, duration))

            pulse_ms_x, pulse_ms_y = 0, 0  # Initialize pulse durations
            if corrected_dy > deadzone_px:
                pulse_ms_y = calculate_pulse(corrected_dy)
                handler.indi_controller.guide_pulse(
                    'N' if el_sign > 0 else 'S', pulse_ms_y)
                pulse_sent = True
            elif corrected_dy < -deadzone_px:
                pulse_ms_y = calculate_pulse(corrected_dy)
                handler.indi_controller.guide_pulse(
                    'S' if el_sign > 0 else 'N', pulse_ms_y)
                pulse_sent = True
            if corrected_dx > deadzone_px:
                pulse_ms_x = calculate_pulse(corrected_dx)
                handler.indi_controller.guide_pulse(
                    'E' if az_sign > 0 else 'W', pulse_ms_x)
                pulse_sent = True
            elif corrected_dx < -deadzone_px:
                pulse_ms_x = calculate_pulse(corrected_dx)
                handler.indi_controller.guide_pulse(
                    'W' if az_sign > 0 else 'E', pulse_ms_x)
                pulse_sent = True

            if pulse_sent:
                # if pulse_ms_x > 0 or pulse_ms_y > 0: logger.info(f"    - Guide Pulses: X={pulse_ms_x}ms, Y={pulse_ms_y}ms")
                time.sleep(settle_s)

            now = time.time()
            if is_stable and (now - last_seq_ts) > min_sequence_interval_s:
                # Before scheduling a new sequence, ensure we are not shutting down
                if handler.state.should_stop_tracking():
                    break
                logger.info(
                    f"  Track [{icao}]: Guiding stable. Capturing sequence...")
                capture_cfg = CONFIG['capture']
                base_exposure = float(
                    capture_cfg.get('sequence_exposure_s', 0.5))

                # Estimate exposure adjustment using the guide frame data
                # This now uses the "flat+text" image in dry run
                exposure_factor = _estimate_exposure_adjustment_from_data(
                    guide_data)
                logger.info(
                    f"    - Auto-exposure factor: {exposure_factor:.2f}")

                adj_min = float(capture_cfg.get(
                    'exposure_adjust_factor_min', 0.2))
                adj_max = float(capture_cfg.get(
                    'exposure_adjust_factor_max', 5.0))
                exposure_factor = max(
                    adj_min, min(adj_max, exposure_factor))
                recommended_exp = base_exposure * exposure_factor

                blur_px_limit = float(
                    capture_cfg.get('target_blur_px', 1.0))
                t_max_blur = float('inf')
                if ang_speed > 1e-3 and plate_scale > 1e-3:
                    t_max_blur = (blur_px_limit * plate_scale) / \
                                  (ang_speed * 3600.0)

                min_exp_limit = float(
                    capture_cfg.get('exposure_min_s', 0.001))
                max_exp_limit = float(
                    capture_cfg.get('exposure_max_s', 5.0))

                final_exp = recommended_exp
                limit_reason = "recommended"
                if final_exp > t_max_blur:
                    final_exp = t_max_blur
                    limit_reason = f"blur_limit ({t_max_blur:.3f}s)"
                if final_exp > max_exp_limit:
                    if limit_reason == "recommended" or t_max_blur > max_exp_limit:
                        limit_reason = f"config_max ({max_exp_limit:.3f}s)"
                    final_exp = max_exp_limit
                if final_exp < min_exp_limit:
                    if limit_reason == "recommended" or t_max_blur < min_exp_limit:
                        limit_reason = f"config_min ({min_exp_limit:.3f}s)"
                    final_exp = min_exp_limit

                logger.info(
                    f"    - Final sequence exposure: {final_exp:.4f}s (Reason: {limit_reason})")

                last_seq_ts = time.time()  # Set timestamp immediately

                # Create the log and status payloads here, in the guide thread
                seq_log = {
                    'type': 'sequence',
                    'sequence_id': int(last_seq_ts * 1000),
                    'icao': icao,
                    'timestamp': last_seq_ts,
                    'per_frame_exposure_s': float(final_exp),
                    'exposure_limit_reason': limit_reason,
                    'guide_offset_px': (float(corrected_dx), float(corrected_dy)),
                    'plate_scale_arcsec_px': float(plate_scale),
                    'predicted_ang_speed_deg_s': float(ang_speed),
                    'target_blur_px_limit': float(blur_px_limit),
                    'last_guide_sharpness': detection.get('sharpness'),
                    'last_guide_confidence': detection.get('confidence'),
                    # n_frames and image_paths will be added by worker
                }

                status_payload_base = {
                    "sequence_exposure_s": float(final_exp),
                    "sequence_meta": {"ang_speed_deg_s": float(ang_speed),
                                      "target_blur_px_limit": float(blur_px_limit),
                                       "exposure_limit_reason": limit_reason}
                }

                # Do not queue new sequences if shutdown has been signaled
                if handler.state.should_stop_tracking():
                    logger.info(
                        f"  Track [{icao}]: Shutdown signaled, skipping sequence scheduling.")
                    break
                # Put the job on the queue for the worker
                job = CaptureJob(
                    icao=icao,
                    exposure=final_exp,
                    seq_log=seq_log,
                    captures_taken=captures_taken,
                    status_payload_base=status_payload_base
                )
                handler.capture_queue.put(job)

                # Update captures_taken *immediately* in this thread
                # The worker will add the *real* count later
                captures_taken = handler.increment_captures(
                    int(CONFIG['capture'].get('num_sequence_images', 5))
                )

                # Write status immediately, worker will update it again
                status_update = status_payload_base.copy()
                status_update['captures_taken'] = captures_taken
                status_update['sequence_count'] = 0  # Worker will set this
                write_status(status_update)

            loop_duration = time.time() - loop_start_time
            sleep_time = max(0.01, 0.1 - loop_duration)
            time.sleep(sleep_time)

    finally:
        logger.info(f"  Track [{icao}]: Exiting tracking thread.")
        with handler.state.lock:
            was_preempted = handler.state.is_preempt_requested()
            finished_icao = handler.state.current_track_icao
            from tracking.state import TrackingIntent  # Local import to avoid circularity
            handler.state.apply_intent(
                intent=TrackingIntent.FINISH_TRACK,
                icao=finished_icao,
            )
            handler.active_thread = None

        was_aborted_by_signal = handler.state.should_stop_tracking()
        end_snapshot = handler.state.get_snapshot()
        manual_icao = end_snapshot.manual_target_icao
        manual_viability = end_snapshot.manual_viability

        final_mode = "monitor" if end_snapshot.monitor_mode else "idle"
        final_status = {"mode": final_mode, "icao": None}
        if was_aborted_by_signal and not was_preempted:
            logger.info(f"  Tracking for {finished_icao} was interrupted.")
            if manual_icao == finished_icao:
                final_status["manual_target"] = {"icao": finished_icao, "viability": {
                    "viable": False, "reasons": ["track aborted"]}}
            else:
                final_status["manual_target"] = None
        elif was_preempted:
            logger.info(
                f"  Tracking for {finished_icao} preempted. Re-evaluating targets...")
            final_status["manual_target"] = None
        else:
            logger.info(
                f"  Tracking for {finished_icao} finished normally. System idle.")
            final_status["manual_target"] = None

        write_status(final_status)

        if was_preempted or not was_aborted_by_signal:
            logger.info("  Triggering scheduler to find next target...")
            latest_snapshot = handler.state.get_snapshot()
            if not latest_snapshot.scheduler_waiting:
                handler._run_scheduler()
