"""
Asynchronous orchestrator for the image stacking pipeline.
Runs stacking in a background thread and publishes a PNG preview to the dashboard.

Enhancements:
- Per-aircraft nesting: logs/stack/<ICAO>/<sequence_id>/
- Each sequence directory contains BOTH the stacked FITS (stack.fits)
  and a finished PNG preview (stack.png).
- Global preview is still atomically published per config.stacking.preview_path.
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

# Try to use the project's PNG preview helper if available
try:
    from image_analyzer import save_png_preview as _save_png_preview_helper  # type: ignore
except Exception:
    _save_png_preview_helper = None

_executor: Optional[ThreadPoolExecutor] = None


def _ensure_executor():
    """Initializes the background thread pool on first use."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1)


def _fits_to_png(src_fits: str, dst_png: str) -> str:
    """
    Minimal FITS->PNG writer that doesn't rely on image_analyzer.
    Produces an 8-bit PNG suitable for the dashboard preview.
    """
    with fits.open(src_fits) as hdul:
        data = hdul[0].data

    if data is None:
        raise ValueError("FITS has no primary data")

    arr = np.asarray(data, dtype=np.float32)
    if not np.isfinite(arr).any():
        raise ValueError("FITS data contains no finite pixels")

    # Robust normalize: clip at 1st-99th percentiles to avoid hot/cold outliers
    finite = arr[np.isfinite(arr)]
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)

    arr8 = (arr * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(dst_png), exist_ok=True)
    if not cv2.imwrite(dst_png, arr8):  # Must end with .png
        raise IOError(f"cv2.imwrite failed for '{dst_png}'")
    return dst_png


def _resolve_preview_dst(preview_dst_cfg: str) -> str:
    """
    Resolve the configured preview destination to an absolute filesystem path.
    - If absolute, keep it.
    - If starts with 'logs/', map to LOG_DIR/<rest>.
    - Otherwise resolve relative to current working directory.
    """
    if os.path.isabs(preview_dst_cfg):
        return preview_dst_cfg

    if preview_dst_cfg.startswith("logs/") or preview_dst_cfg.startswith("logs" + os.sep):
        parts = preview_dst_cfg.replace("\\", "/").split("/", 1)
        rest = parts[1] if len(parts) > 1 else ""
        return os.path.join(LOG_DIR, rest)

    return os.path.abspath(preview_dst_cfg)


def _make_sequence_png_from_fits(fits_path: str, seq_png_path: str) -> str:
    """
    Create a PNG preview next to the stacked FITS in the sequence directory.
    Uses project helper if available (supports both 2-arg and 1-arg signatures),
    otherwise falls back to local FITS->PNG conversion.

    Returns the path to the written PNG.
    """
    os.makedirs(os.path.dirname(seq_png_path), exist_ok=True)
    tmp_png = seq_png_path + ".tmp.png"

    # Attempt helper if available (support both signatures)
    if _save_png_preview_helper is not None:
        try:
            # Preferred signature: (fits_in, png_out)
            _save_png_preview_helper(fits_path, tmp_png)
            os.replace(tmp_png, seq_png_path)
            return seq_png_path
        except TypeError:
            # Older helper signature: (fits_in) -> writes next to FITS
            try:
                _save_png_preview_helper(fits_path)
                guess = os.path.splitext(fits_path)[0] + ".png"
                if os.path.exists(guess):
                    os.replace(guess, tmp_png)
                    os.replace(tmp_png, seq_png_path)
                    return seq_png_path
            except Exception as e:
                print(f"  Orchestrator: save_png_preview (1-arg) failed: {e}")
        except Exception as e:
            print(f"  Orchestrator: save_png_preview failed: {e}")

    # Fallback: local conversion
    try:
        _fits_to_png(fits_path, tmp_png)
        os.replace(tmp_png, seq_png_path)
        return seq_png_path
    finally:
        # Clean up any stray temp file on failure
        if os.path.exists(tmp_png):
            try:
                os.remove(tmp_png)
            except OSError:
                pass


def _publish_preview_from_fits(fits_path: str, preview_dst_cfg: str):
    """
    Create and publish a PNG preview from a stacked FITS.
    - Uses image_analyzer.save_png_preview if available (both 2-arg and 1-arg forms).
    - Otherwise falls back to a local FITS->PNG conversion.
    - Publishes atomically via os.replace.
    """
    preview_dst = _resolve_preview_dst(preview_dst_cfg)
    os.makedirs(os.path.dirname(preview_dst), exist_ok=True)

    # Write to a temp file that STILL ends with .png so OpenCV selects the PNG encoder.
    tmp_png = preview_dst + ".tmp.png"

    # Attempt helper if available (support both signatures)
    if _save_png_preview_helper is not None:
        try:
            # Preferred signature: (fits_in, png_out)
            _save_png_preview_helper(fits_path, tmp_png)
            os.replace(tmp_png, preview_dst)
            return
        except TypeError:
            # Older helper signature: (fits_in) -> writes next to FITS
            try:
                _save_png_preview_helper(fits_path)
                guess = os.path.splitext(fits_path)[0] + ".png"
                if os.path.exists(guess):
                    os.replace(guess, tmp_png)
                    os.replace(tmp_png, preview_dst)
                    return
            except Exception as e:
                print(f"  Orchestrator: save_png_preview (1-arg) failed: {e}")
        except Exception as e:
            print(f"  Orchestrator: save_png_preview failed: {e}")

    # Fallback: local conversion
    try:
        _fits_to_png(fits_path, tmp_png)
        os.replace(tmp_png, preview_dst)
    finally:
        # Clean up any stray temp file on failure
        if os.path.exists(tmp_png):
            try:
                os.remove(tmp_png)
            except OSError:
                pass


def _publish_preview_from_png(src_png: str, preview_dst_cfg: str):
    """
    Atomically publish an already-generated PNG (sequence PNG) to the global path.
    """
    preview_dst = _resolve_preview_dst(preview_dst_cfg)
    os.makedirs(os.path.dirname(preview_dst), exist_ok=True)
    tmp_png = preview_dst + ".tmp.png"
    # Copy to temp then atomic replace
    with open(src_png, "rb") as fin, open(tmp_png, "wb") as fout:
        fout.write(fin.read())
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(tmp_png, preview_dst)


def schedule_stack_and_publish(sequence_id: str, image_paths: List[str], capture_meta: Dict):
    """
    Submits a stacking job to the background worker thread.
    NOTE: We derive the ICAO from sequence_id (prefix before first '_') so callers
    don't need to change their API usage.
    """
    if not CONFIG.get('stacking', {}).get('enabled', False):
        return
    if not image_paths:
        return
    _ensure_executor()
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
        master_fits, qc = stack_images(image_paths, out_dir, params)

        if not master_fits:
            print(f"  Orchestrator: Stacking failed for sequence {sequence_id}. Reason: {qc.get('error')}")
            return

        # --- NEW: write a PNG next to the stacked FITS inside the sequence directory ---
        seq_png_path = os.path.join(out_dir, "stack.png")
        try:
            _make_sequence_png_from_fits(master_fits, seq_png_path)
        except Exception as e:
            print(f"  Orchestrator: Failed to create per-sequence PNG: {e}")
            seq_png_path = None

        # Publish the PNG preview for the dashboard (atomic replace)
        preview_cfg_path = CONFIG.get('stacking', {}).get('preview_path', "logs/latest_stack_preview.png")
        try:
            if seq_png_path and os.path.exists(seq_png_path):
                _publish_preview_from_png(seq_png_path, preview_cfg_path)
            else:
                # Fallback to generating directly to the global path if sequence PNG failed
                _publish_preview_from_fits(master_fits, preview_cfg_path)
            print("  Orchestrator: Published new stack preview.")
        except Exception as e:
            print(f"  Orchestrator: Preview publish from master failed: {e}")

        # Log a manifest for this sequence inside its directory
        manifest = {
            "sequence_id": sequence_id,
            "icao": icao,
            "paths": image_paths,
            "capture_meta": capture_meta,
            "stack_params_used": params,
            "qc": qc,
            "outputs": {
                "master_fits": master_fits,
                # Keep legacy field name for compatibility, plus explicit global path:
                "preview_png": preview_cfg_path,
                "sequence_png": seq_png_path,
                "global_preview_png": _resolve_preview_dst(preview_cfg_path)
            },
            "timestamp": int(time.time())
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    except Exception as e:
        print(f"  Orchestrator: Unhandled exception in stacking worker: {e}")
