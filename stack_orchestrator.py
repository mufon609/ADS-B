"""
Asynchronous orchestrator for the image stacking pipeline.
Runs stacking in a background thread.
The dashboard now reads products directly from the sequence directory.

Enhancements:
- Per-aircraft nesting: logs/stack/<ICAO>/<sequence_id>/
- Each sequence directory contains stacked FITS and PNG previews.
- Removed logic for publishing to a single global preview path.
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

import numpy as np
import cv2
from astropy.io import fits

from config_loader import CONFIG, LOG_DIR
from stacker import stack_images

try:
    from image_analyzer import save_png_preview as _save_png_preview_helper
except Exception:
    _save_png_preview_helper = None

_executor: Optional[ThreadPoolExecutor] = None


def _ensure_executor():
    """Initializes the background thread pool on first use."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1)


def _resolve_preview_dst(preview_dst_cfg: str) -> str:
    """
    Resolve a configured path to an absolute filesystem path.
    (Kept in case manifest needs to resolve paths, but no longer used for publishing)
    """
    if os.path.isabs(preview_dst_cfg):
        return preview_dst_cfg
    if preview_dst_cfg.startswith("logs/") or preview_dst_cfg.startswith("logs" + os.sep):
        parts = preview_dst_cfg.replace("\\", "/").split("/", 1)
        rest = parts[1] if len(parts) > 1 else ""
        return os.path.join(LOG_DIR, rest)
    return os.path.abspath(preview_dst_cfg)


def _rel_to_logs_url(path: str) -> Optional[str]:
    """Map an absolute path under LOG_DIR to a /logs/... URL for the manifest."""
    if not path: return None # Handle None or empty paths
    try:
        abs_path = os.path.abspath(path)
        abs_root = os.path.abspath(LOG_DIR)
        if os.path.commonpath([abs_path, abs_root]) != abs_root:
            return None # Path is not inside LOG_DIR
        rel = os.path.relpath(abs_path, abs_root).replace(os.sep, "/")
        return f"/logs/{rel}"
    except Exception:
        return None

def schedule_stack_and_publish(sequence_id: str, image_paths: List[str], capture_meta: Dict):
    """
    Submits a stacking job to the background worker thread.
    """
    if not CONFIG.get('stacking', {}).get('enabled', False):
        return
    if not image_paths:
        return
    _ensure_executor()
    if _executor:
        _executor.submit(_run_stacking_pipeline, sequence_id, image_paths, capture_meta)


def _derive_icao(sequence_id: str) -> str:
    """Get ICAO from the sequence_id (prefix before first underscore)."""
    if not sequence_id:
        return "unknown"
    if "_" in sequence_id:
        return sequence_id.split("_", 1)[0]
    return sequence_id


def _run_stacking_pipeline(sequence_id: str, image_paths: List[str], capture_meta: Dict):
    """The main worker function that runs in a separate thread."""
    try:
        icao = _derive_icao(sequence_id)
        print(f"  Orchestrator: Starting background stacking for sequence {sequence_id} (ICAO: {icao})")

        # Nested structure: logs/stack/<ICAO>/<sequence_id>/
        aircraft_dir = os.path.join(LOG_DIR, "stack", icao)
        out_dir = os.path.join(aircraft_dir, sequence_id)
        os.makedirs(aircraft_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        params = CONFIG.get('stacking', {}).copy()
        # stack_images now calls stack_images_multi, which creates all 3 FITS + 3 PNGs
        # inside the 'out_dir'
        master_fits, qc = stack_images(image_paths, out_dir, params) # master_fits is robust stack path

        
        if not master_fits or qc.get("error"):
            print(f"  Orchestrator: Stacking failed for sequence {sequence_id}. Reason: {qc.get('error', 'Unknown')}")
            qc["status"] = "failed"
            # Try to write a failure manifest
        else:
            print(f"  Orchestrator: Stacking complete for sequence {sequence_id}.")
            qc["status"] = qc.get("status", "success") # Use status from stacker if present

        # Log a manifest for this sequence inside its directory
        manifest = {
            "sequence_id": sequence_id,
            "icao": icao,
            "paths": image_paths,
            "capture_meta": capture_meta,
            "stack_params_used": params,
            "qc": qc,
            "outputs": {
                "master_fits": _rel_to_logs_url(master_fits) if master_fits else None, # This is stack_robust.fits
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

def shutdown():
    """Shuts down the background thread pool executor."""
    global _executor
    if _executor:
        print("  Orchestrator: Shutting down stacking thread pool...")
        _executor.shutdown(wait=True) # Wait for pending tasks to complete
        _executor = None
        print("  Orchestrator: Stacking thread pool shut down.")

