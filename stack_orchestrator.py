"""
Asynchronous orchestrator for image stacking.

This module runs the stacking pipeline in background threads. It reads
``stacking.num_workers`` from the configuration to determine how many
concurrent sequences can be processed at once. Each stacking job calls
``stack_images`` from ``stacker.py``, which itself computes mean,
robust and anomaly stacks concurrently.
"""

from __future__ import annotations

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

from config_loader import CONFIG, LOG_DIR
from stacker import stack_images

# Global executor shared across all stacking requests
_executor: Optional[ThreadPoolExecutor] = None

# Indicates that shutdown has been requested. When true, no new stacking jobs
# will be accepted. This flag is set by ``shutdown()`` or explicitly via
# ``request_shutdown()``.  Existing queued jobs will still run to
# completion.
_shutdown_requested: bool = False

def request_shutdown() -> None:
    """Mark that the orchestrator should not accept any new stacking jobs."""
    global _shutdown_requested
    _shutdown_requested = True


def _ensure_executor() -> None:
    """Initializes the background thread pool based on the config."""
    global _executor
    if _executor is None:
        stacking_cfg = CONFIG.get('stacking', {})
        max_workers = int(stacking_cfg.get('num_workers', 1))
        # Ensure at least one worker
        _executor = ThreadPoolExecutor(max_workers=max(1, max_workers))


def _rel_to_logs_url(path: str) -> Optional[str]:
    """Map an absolute path under LOG_DIR to a /logs/... URL for the manifest."""
    if not path:
        return None
    try:
        abs_path = os.path.abspath(path)
        abs_root = os.path.abspath(LOG_DIR)
        if os.path.commonpath([abs_path, abs_root]) != abs_root:
            return None
        rel = os.path.relpath(abs_path, abs_root).replace(os.sep, "/")
        return f"/logs/{rel}"
    except Exception:
        return None


def schedule_stack_and_publish(sequence_id: str, image_paths: List[str], capture_meta: Dict) -> None:
    """
    Submit a stacking job to the background worker.

    If stacking is disabled or no images are provided, the function
    returns immediately.
    """
    # Do not schedule new jobs if stacking is disabled or shutdown has been requested
    if not CONFIG.get('stacking', {}).get('enabled', False):
        return
    if not image_paths:
        return
    # Respect global shutdown flag: skip submitting new tasks once shutdown is requested
    if _shutdown_requested:
        return
    _ensure_executor()
    # It's possible that the executor has been shut down; guard against None or
    # rejected submissions.  Note: we do not wrap submit in try/except here; the
    # caller should ensure shutdown is coordinated.
    if _executor:
        try:
            _executor.submit(_run_stacking_pipeline, sequence_id, image_paths, capture_meta)
        except RuntimeError:
            # If the executor is shut down, ignore new tasks
            pass


def _derive_icao(sequence_id: str) -> str:
    """Extract the ICAO code from the sequence ID (prefix before the first underscore)."""
    if not sequence_id:
        return "unknown"
    if "_" in sequence_id:
        return sequence_id.split("_", 1)[0]
    return sequence_id


def _run_stacking_pipeline(sequence_id: str, image_paths: List[str], capture_meta: Dict) -> None:
    """
    Worker function that stacks a sequence in a background thread.

    It builds a nested directory under LOG_DIR/stack/ICAO/sequence_id, calls
    ``stack_images``, then writes a manifest JSON summarizing the results.
    """
    try:
        icao = _derive_icao(sequence_id)
        print(f"  Orchestrator: Starting background stacking for sequence {sequence_id} (ICAO: {icao})")
        # Create output directories
        aircraft_dir = os.path.join(LOG_DIR, "stack", icao)
        out_dir = os.path.join(aircraft_dir, sequence_id)
        os.makedirs(out_dir, exist_ok=True)
        # Copy stacking parameters from config; these include sigma_clip_z, anomaly_mask_radius_px, internal_threads, etc.
        params = CONFIG.get('stacking', {}).copy()
        master_fits, qc = stack_images(image_paths, out_dir, params)
        # Determine status
        if not master_fits or qc.get("error"):
            print(f"  Orchestrator: Stacking failed for sequence {sequence_id}. Reason: {qc.get('error', 'Unknown')}")
            qc["status"] = "failed"
        else:
            print(f"  Orchestrator: Stacking complete for sequence {sequence_id}.")
            qc["status"] = qc.get("status", "success")
        # Build manifest
        manifest = {
            "sequence_id": sequence_id,
            "icao": icao,
            "paths": image_paths,
            "capture_meta": capture_meta,
            "stack_params_used": params,
            "qc": qc,
            "outputs": {
                "master_fits": _rel_to_logs_url(master_fits) if master_fits else None,
                "sequence_mean_png": _rel_to_logs_url(os.path.join(out_dir, "stack_mean.png")),
                "sequence_robust_png": _rel_to_logs_url(os.path.join(out_dir, "stack_robust.png")),
                "sequence_anomaly_png": _rel_to_logs_url(os.path.join(out_dir, "stack_anomaly.png")),
                "sequence_mean_fits": _rel_to_logs_url(os.path.join(out_dir, "stack_mean.fits")),
                "sequence_robust_fits": _rel_to_logs_url(os.path.join(out_dir, "stack_robust.fits")),
                "sequence_anomaly_fits": _rel_to_logs_url(os.path.join(out_dir, "stack_anomaly.fits")),
            },
            "timestamp": int(time.time())
        }
        manifest_path = os.path.join(out_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"  Orchestrator: Unhandled exception in stacking worker: {e}")
        import traceback
        traceback.print_exc()


def shutdown() -> None:
    """Shuts down the background executor, waiting for all queued tasks to finish."""
    global _executor, _shutdown_requested
    # Mark that no new jobs should be accepted
    _shutdown_requested = True
    if _executor:
        print("  Orchestrator: Shutting down stacking thread pool...")
        try:
            _executor.shutdown(wait=True)
        finally:
            _executor = None
            print("  Orchestrator: Stacking thread pool shut down.")