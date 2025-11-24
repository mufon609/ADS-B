# stacker.py
"""
Image stacking utilities for the ADS‑B tracker.

This module is adapted from the upstream project but adds concurrency to
speed up stacking operations. Once images are aligned into a 3‑D volume,
the mean, robust sigma‑clipped mean and anomaly map are computed in
parallel threads.  The number of internal threads is determined by the
``internal_threads`` parameter passed via ``params`` (default 3).

The module exposes two public entry points:

* ``stack_images_multi`` – Given a list of FITS paths, produces mean,
  robust and anomaly stacks and writes FITS/PNG outputs into an output
  directory. Returns a dictionary of saved products and a QC report.

* ``stack_images`` – Legacy wrapper that returns only the robust FITS
  path and QC report for backwards compatibility.
"""

import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import traceback
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from imaging.image_analyzer import _detect_aircraft_from_data, _load_fits_data

logger = logging.getLogger(__name__)


def _normalize_to_png(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Robustly stretch a float image to 0..255 uint8 using percentiles."""
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    arr = np.nan_to_num(
        arr, nan=lo, posinf=hi, neginf=lo)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _write_fits(path: str, data: np.ndarray) -> str:
    """Writes a NumPy array to a FITS file, creating directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if data.dtype not in [
        np.float32, np.int16, np.int32, np.uint16
    ]:
        data = data.astype(np.float32)
    fits.HDUList([fits.PrimaryHDU(data)]).writeto(
        path, overwrite=True, checksum=True)
    return path


def _write_png(path: str, data_float: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> str:
    """Normalizes float data and saves it as PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img8 = _normalize_to_png(data_float, lo_pct, hi_pct)
    success = cv2.imwrite(path, img8)
    if not success:
        raise IOError(f"cv2.imwrite failed for '{path}'")
    return path


def _load_and_center(paths: List[str]) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Loads FITS images and finds aircraft centers. Skips unusable frames."""
    images: List[np.ndarray] = []
    centers: List[Tuple[float, float]] = []
    skipped = 0
    for i, p in enumerate(paths):
        img_data = _load_fits_data(p)
        if img_data is None:
            skipped += 1
            continue
        original_shape = img_data.shape
        ctr = None
        det = None
        try:
            det = _detect_aircraft_from_data(
                img_data, original_shape=original_shape)
        except Exception as e:
            logger.error(
                f"  - Stacker: Detection function failed on frame {i+1} ({os.path.basename(p)}): {e}")
            det = None
        if (
            det
            and det.get("detected")
            and det.get("center_px") is not None
        ):
            ctr_candidate = det.get("center_px")
            if (
                isinstance(ctr_candidate, (tuple, list))
                and len(ctr_candidate) == 2
                and isinstance(ctr_candidate[0], (int, float))
                and isinstance(ctr_candidate[1], (int, float))
                and np.isfinite(ctr_candidate[0])
                and np.isfinite(ctr_candidate[1])
            ):
                ctr = ctr_candidate
            else:
                logger.warning(
                    f"  - Stacker: Invalid center_px format in frame {i+1}: {ctr_candidate}. Skipping frame.")
                ctr = None
                skipped += 1
        else:
            reason = det.get('reason', 'detection_failed') if det else \
                'load_failed_or_detect_exception'
            logger.warning(
                "  - Stacker: No valid detection in frame %d (%s). "
                "Reason: %s. Skipping frame.",
                i + 1,
                os.path.basename(p),
                reason,
            )
            ctr = None
            skipped += 1
        if ctr is not None:
            images.append(img_data)
            centers.append(ctr)
    if skipped > 0:
        logger.info(
            f"  - Stacker: Skipped {skipped} / {len(paths)} frames due to load/detection issues.")
    return images, centers


def _align_images(images: List[np.ndarray], centers: List[Tuple[float, float]]) -> List[np.ndarray]:
    """
    Translates images so the aircraft remains at a fixed location, modifying
    the input `images` list in-place.
    """
    if len(images) < 2:
        return images
    ref_center = centers[0]
    H, W = images[0].shape
    for i, (img, ctr) in enumerate(zip(images, centers)):
        if img.shape != (H, W):
            logger.warning(
                f"  - Stacker: Shape mismatch (got {img.shape}, "
                f"expected {(H, W)}), skipping frame.")
            # Replace with a zero array to maintain list length, or handle removal
            images[i] = np.zeros((H, W), dtype=img.dtype)
            continue
        dx = ref_center[0] - ctr[0]
        dy = ref_center[1] - ctr[1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(
            img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        images[i] = shifted # Overwrite original image with aligned version
    return images


def _sigma_clipped_mean(volume: np.ndarray, clip_z: float = 3.0, max_iters: int = 5) -> Tuple[np.ndarray, float]:
    """Computes a robust mean via iterative sigma-clipping along the frame axis."""
    if (volume.ndim != 3) or (volume.shape[0] < 2):
        return np.nanmean(volume, axis=0), 0.0
    try:
        mean_stack, _, _ = sigma_clipped_stats(
            volume,
            sigma=clip_z,
            maxiters=max_iters,
            axis=0,
            cenfunc='median',
            stdfunc='mad_std'
        )
        masked_pixels = np.count_nonzero(
            mean_stack.mask) if hasattr(mean_stack, 'mask') else 0
        total_pixels = mean_stack.size if hasattr(mean_stack, 'size') else 0
        masked_fraction = (
            (masked_pixels / total_pixels) if total_pixels > 0
            else 0.0
        )

        if hasattr(mean_stack, 'filled'):
            fill_val = np.nanmedian(volume, axis=0)
            stacked_filled = mean_stack.filled(fill_val)
        else:
            stacked_filled = mean_stack

        return stacked_filled.astype(np.float32), masked_fraction
    except Exception:
        return np.nanmean(volume, axis=0), 0.0


def _anomaly_map(volume: np.ndarray, mask_radius_px: int = 20) -> np.ndarray:
    """Generates an anomaly map by finding max absolute Z-score per pixel."""
    if (volume.ndim != 3) or (volume.shape[0] < 2):
        return (
            np.zeros_like(volume[0])
            if volume.ndim == 3 and volume.size > 0
            else np.zeros((100, 100), dtype=np.float32)
        )
    N, H, W = volume.shape
    med = np.nanmedian(volume, axis=0)
    abs_dev = np.abs(volume - med)
    mad = np.nanmedian(abs_dev, axis=0)
    sigma_est = 1.4826 * mad + 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        z_scores = (volume - med) / sigma_est
    z_abs_max = np.nanmax(np.abs(z_scores), axis=0)
    z_abs_max = np.nan_to_num(z_abs_max, nan=0.0)
    if mask_radius_px > 0:
        cy, cx = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        dist_sq = (x - cx)**2 + (y - cy)**2
        mask = dist_sq <= mask_radius_px**2
        z_abs_max = np.where(
            mask, z_abs_max * 0.25, z_abs_max)
    return z_abs_max.astype(np.float32)


def stack_images_multi(image_paths: List[str], output_dir: str, params: Dict) -> Tuple[Dict, Dict]:
    """
    Main stacking pipeline with concurrent computations for speed.
    Loads/detects/aligns frames, then uses an internal ThreadPoolExecutor for mean/robust/anomaly stacks.
    Requires at least 2 usable frames; returns product paths and a QC dict (with errors when applicable).
    """
    qc: Dict = {}
    products: Dict = {}
    if not image_paths:
        qc["error"] = "No image paths provided."
        return products, qc
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Load images and find detection centers
        images, centers = _load_and_center(image_paths)
        qc["n_frames_input"] = len(image_paths)
        qc["n_frames_loaded_detected"] = len(images)
        if len(images) < 2:
            qc["error"] = (
                f"Stacking failed: Only {len(images)} usable frames "
                "loaded/detected for alignment."
            )
            return products, qc
        # Align images (in-place)
        images = _align_images(images, centers)
        qc["n_frames_aligned"] = len(images)
        if len(images) < 2:
            qc["error"] = "Stacking failed: Fewer than 2 images were successfully aligned."
            return products, qc
        # Build 3D volume
        vol = np.stack(images, axis=0).astype(np.float32)
        # Parameters for robust and anomaly stacking
        clip_z = float(params.get("sigma_clip_z", 3.0))
        mask_r = int(params.get("anomaly_mask_radius_px", 20))
        internal_threads = int(params.get("internal_threads", 3))
        # Compute mean, robust and anomaly stacks concurrently
        try:
            with ThreadPoolExecutor(max_workers=max(1, internal_threads)) as executor:
                f_mean = executor.submit(np.nanmean, vol, 0)
                f_robust = executor.submit(_sigma_clipped_mean, vol, clip_z)
                f_anom = executor.submit(_anomaly_map, vol, mask_r)
                mean_stack = f_mean.result().astype(np.float32)
                robust_res = f_robust.result()
                anomaly = f_anom.result()
            robust_stack, clipped_fraction = robust_res
        except RuntimeError as e:
            # When the interpreter is shutting down, ThreadPoolExecutor may reject
            # new tasks. Fall back to sequential computation.
            if "cannot schedule new futures" in str(e):
                mean_stack = np.nanmean(vol, axis=0).astype(np.float32)
                robust_stack, clipped_fraction = _sigma_clipped_mean(
                    vol, clip_z)
                anomaly = _anomaly_map(vol, mask_r)
            else:
                raise
        qc["sigma_clip_z"] = clip_z
        qc["clipped_fraction"] = round(clipped_fraction, 6)
        qc["anomaly_mask_radius_px"] = mask_r
        # Define output file paths
        mean_fits_path = os.path.join(output_dir, "stack_mean.fits")
        robust_fits_path = os.path.join(output_dir, "stack_robust.fits")
        anomaly_fits_path = os.path.join(output_dir, "stack_anomaly.fits")
        mean_png_path = os.path.join(output_dir, "stack_mean.png")
        robust_png_path = os.path.join(output_dir, "stack_robust.png")
        anomaly_png_path = os.path.join(output_dir, "stack_anomaly.png")
        # Clip to uint16 range
        mean_stack_u16 = np.clip(np.nan_to_num(
            mean_stack, nan=0), 0, 65535).astype(np.uint16)
        robust_stack_u16 = np.clip(np.nan_to_num(
            robust_stack, nan=0), 0, 65535).astype(np.uint16)
        anomaly_finite = anomaly[np.isfinite(anomaly)]
        if anomaly_finite.size > 0:
            vmin, vmax = np.percentile(anomaly_finite, [1, 99.5])
            if vmax <= vmin:
                vmax = vmin + 1e-6
            anomaly_scaled_u16 = (
                np.clip((anomaly - vmin) / (vmax - vmin), 0, 1) * 65535
            )
            anomaly_scaled_u16 = np.nan_to_num(
                anomaly_scaled_u16, nan=0).astype(np.uint16)
        else:
            anomaly_scaled_u16 = np.zeros_like(anomaly, dtype=np.uint16)
        # Save FITS
        saved_products: Dict[str, Dict[str, str]] = {
            "mean": {}, "robust": {}, "anomaly": {}}
        try:
            saved_products["mean"]["fits"] = _write_fits(
                mean_fits_path, mean_stack_u16)
        except Exception as e:
            qc.setdefault("save_errors", []).append(f"Mean FITS: {e}")
        try:
            saved_products["robust"]["fits"] = _write_fits(
                robust_fits_path, robust_stack_u16)
        except Exception as e:
            qc.setdefault("save_errors", []).append(f"Robust FITS: {e}")
        try:
            saved_products["anomaly"]["fits"] = _write_fits(
                anomaly_fits_path, anomaly_scaled_u16)
        except Exception as e:
            qc.setdefault("save_errors", []).append(f"Anomaly FITS: {e}")
        # Save PNGs using unscaled floats for proper normalization
        try:
            saved_products["mean"]["png"] = _write_png(
                mean_png_path, mean_stack, 1.0, 99.0)
        except Exception as e:
            qc.setdefault("save_errors", []).append(f"Mean PNG: {e}")
        try:
            saved_products["robust"]["png"] = _write_png(
                robust_png_path, robust_stack, 1.0, 99.0)
        except Exception as e:
            qc.setdefault("save_errors", []).append(f"Robust PNG: {e}")
        try:
            saved_products["anomaly"]["png"] = _write_png(
                anomaly_png_path, anomaly, 5.0, 99.5)
        except Exception as e:
            qc.setdefault("save_errors", []).append(f"Anomaly PNG: {e}")
        if not saved_products.get("robust", {}).get("fits"):
            qc["error"] = "Stacking failed: Could not save the primary robust FITS file."
            return saved_products, qc
        qc["status"] = (
            "success" if "error" not in qc and not qc.get("save_errors")
            else "partial_success"
        )
        return saved_products, qc
    except Exception as e:
        logger.error(
            f"  - Stacker: Unhandled error in stack_images_multi: {e}")
        traceback.print_exc()
        qc["error"] = f"Unhandled exception: {e}"
        return products, qc


def stack_images(image_paths: List[str], output_dir: str, params: Dict) -> Tuple[Optional[str], Dict]:
    """Legacy API wrapper returning only the robust FITS path and QC."""
    products, qc = stack_images_multi(image_paths, output_dir, params)
    if qc.get("error") or (
        not products.get("robust", {}).get("fits")
    ):
        if not qc.get("error"):
            qc["error"] = "Robust FITS output missing after stacking."
        logger.error(
            f"  - Stacker (Legacy): Stacking failed. Reason: {qc.get('error', 'Unknown error')}")
        return None, qc
    return products["robust"]["fits"], qc
