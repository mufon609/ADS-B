#stacker.py
"""
A post-processing script to align and stack a sequence of FITS images
captured by the aircraft tracker. This reduces noise and motion blur.

Now supports multiple outputs:
 - Mean stack
 - Robust (sigma-clipped) mean stack
 - Anomaly map (highlights objects not moving with the aircraft)
"""

import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from astropy.io import fits

# --- FIX: Import necessary internal functions from image_analyzer ---
from image_analyzer import _load_fits_data, _detect_aircraft_from_data
# --- End FIX ---


def _fallback_center(img: np.ndarray) -> Tuple[float, float]:
    """Returns the geometric center of the image."""
    h, w = img.shape[:2]
    return (w / 2.0, h / 2.0)


def _normalize_to_png(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Robustly stretch a float image to 0..255 uint8 using percentiles."""
    # Ensure input is float32 for calculations
    arr = np.asarray(arr, dtype=np.float32)
    # Filter out non-finite values for percentile calculation
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        # Return black image if no valid data
        return np.zeros_like(arr, dtype=np.uint8)
    # Calculate percentile limits
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    # Handle flat images where lo == hi
    if hi <= lo:
        hi = lo + 1e-6 # Add tiny epsilon to prevent division by zero
    # Clip and scale data to 0-1 range
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    # Convert to 8-bit unsigned integer
    return (arr * 255.0).astype(np.uint8)


def _write_fits(path: str, data: np.ndarray) -> str:
    """Saves NumPy array to a FITS file, creating directory if needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Ensure data is in a FITS-compatible format (e.g., float32 or int16/32)
        # Convert uint16 explicitly if needed, handle other types as float32
        if data.dtype == np.uint16:
            # Astropy might handle uint16, but converting can be safer
            # Or pass as is if astropy version supports it well
            pass # Keep uint16
        elif data.dtype != np.float32:
             data = data.astype(np.float32) # Default to float32

        fits.HDUList([fits.PrimaryHDU(data)]).writeto(path, overwrite=True)
        return path
    except Exception as e:
         # Raise exception to be caught by orchestrator
         raise IOError(f"Failed to write FITS file '{path}': {e}") from e


def _write_png(path: str, data_float: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> str:
    """Normalizes float data and saves as PNG, creating directory if needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img8 = _normalize_to_png(data_float, lo_pct, hi_pct)
        # Check if imwrite was successful
        success = cv2.imwrite(path, img8)
        if not success:
            raise IOError(f"cv2.imwrite failed for '{path}' (check path/permissions/data)")
        return path
    except Exception as e:
         # Raise exception to be caught by orchestrator
        raise IOError(f"Failed to write PNG file '{path}': {e}") from e


def _load_and_center(paths: List[str]) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Load FITS images (float32) and get aircraft centers using in-memory data."""
    images, centers = [], []
    for p in paths:
        # --- FIX: Load image data ONCE ---
        img_data = _load_fits_data(p)
        # --- End FIX ---

        if img_data is None:
            # _load_fits_data already prints warnings/errors
            continue # Skip this file if loading failed

        # Store original shape before potential modifications
        original_shape = img_data.shape

        ctr = None
        det = None
        try:
            # --- FIX: Call detection function with NumPy array ---
            det = _detect_aircraft_from_data(img_data, original_shape=original_shape)
            # --- End FIX ---
        except Exception as e:
            # Catch errors specifically from the detection function call
            print(f"  - Stacker: _detect_aircraft_from_data failed on {os.path.basename(p)}: {e}")
            det = None # Ensure det is None on error

        # Check detection results
        if det and det.get("detected") and det.get("center_px") is not None:
            ctr = det.get("center_px")
            # Basic validation of center coordinates
            if not (isinstance(ctr, (tuple, list)) and len(ctr) == 2 and
                    isinstance(ctr[0], (int, float)) and isinstance(ctr[1], (int, float))):
                 print(f"  - Stacker: Invalid center_px format from detection: {ctr}. Using fallback.")
                 ctr = None # Invalidate bad center

        # Fallback to geometric center if detection failed or center invalid
        if ctr is None:
            ctr = _fallback_center(img_data)
            reason = det.get('reason', 'detection_failed') if det else 'detection_failed'
            print(f"  - Stacker: No valid detection in {os.path.basename(p)} (Reason: {reason}). Using fallback center {ctr}.")

        # --- FIX: Append the loaded image DATA (NumPy array) ---
        images.append(img_data)
        # --- End FIX ---
        centers.append(ctr)

    return images, centers


def _align_images(images: List[np.ndarray], centers: List[Tuple[float, float]]) -> List[np.ndarray]:
    """Translate-align images so the aircraft stays fixed."""
    if len(images) < 2: # Need at least two images to align
        print("  - Stacker: Not enough images (<2) to perform alignment.")
        return images # Return original list (might be empty or single image)

    # Use the center from the first image as the reference point
    ref_center = centers[0]
    # Get dimensions from the first image
    H, W = images[0].shape
    aligned = []

    for i, (img, ctr) in enumerate(zip(images, centers)):
        # Check if image dimensions match the first image
        if img.shape != (H, W):
            print(f"  - Stacker: Shape mismatch in frame {i} ({img.shape} vs {(H,W)}), skipping alignment for this frame.")
            # Decide: skip frame entirely or append unaligned? Skipping for safety.
            continue

        # Calculate shift needed to move current center (ctr) to reference center (ref_center)
        dx = ref_center[0] - ctr[0]
        dy = ref_center[1] - ctr[1]

        # Create affine transformation matrix for pure translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply the warp (shift)
        # Use INTER_LINEAR for smoother results, BORDER_REFLECT to handle edges reasonably
        shifted = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        aligned.append(shifted)

    return aligned


def _sigma_clipped_mean(volume: np.ndarray, clip_z: float = 3.0) -> Tuple[np.ndarray, float]:
    """
    Robust mean via sigma-clipping using Median Absolute Deviation (MAD).
    Returns (stacked_image, fraction_of_pixels_clipped).
    """
    if volume.ndim != 3 or volume.shape[0] < 2: # Need N>1 frames along axis 0
        print("  - Stacker Warning: Sigma clipping requires at least 2 frames.")
        # Fallback to simple mean if insufficient data
        return np.mean(volume, axis=0), 0.0

    # Calculate median across frames (axis 0)
    med = np.median(volume, axis=0)
    # Calculate MAD across frames
    # Keepdims=True might simplify broadcasting but uses more memory temporarily
    mad = np.median(np.abs(volume - med), axis=0)
    # Scale MAD to estimate standard deviation (for Gaussian data)
    # Add small epsilon to prevent division by zero in flat areas
    sigma_est = 1.4826 * mad + 1e-6

    # Calculate Z-score relative to median and estimated sigma
    z = (volume - med) / sigma_est
    # Create mask where absolute Z-score is within threshold
    mask = np.abs(z) <= clip_z

    # Calculate clipped fraction
    kept_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    clipped_fraction = 1.0 - (kept_pixels / total_pixels) if total_pixels > 0 else 0.0

    # Apply mask: Replace clipped values with NaN for nanmean calculation
    vol_masked = np.where(mask, volume, np.nan)

    # Calculate mean ignoring NaNs
    stacked = np.nanmean(vol_masked, axis=0)

    # Fill any remaining NaNs (pixels clipped in *all* frames) with the median value
    stacked = np.nan_to_num(stacked, nan=med)

    return stacked, clipped_fraction


def _anomaly_map(volume: np.ndarray, mask_radius_px: int = 20) -> np.ndarray:
    """
    Highlights transient objects or background streaks by finding max deviation.
    Assumes 'volume' is already aircraft-aligned: [N, H, W].
    """
    if volume.ndim != 3 or volume.shape[0] < 2:
        print("  - Stacker Warning: Anomaly map requires at least 2 frames.")
        # Return zeros or median if insufficient data? Zeros for now.
        return np.zeros_like(volume[0]) if volume.ndim==3 else np.zeros((100,100)) # Placeholder shape

    N, H, W = volume.shape
    # Calculate robust statistics
    med = np.median(volume, axis=0)
    mad = np.median(np.abs(volume - med), axis=0)
    sigma_est = 1.4826 * mad + 1e-6 # Add epsilon

    # Calculate max absolute Z-score per pixel across all frames
    z_abs_max = np.max(np.abs((volume - med) / sigma_est), axis=0)

    # Optional: Mask (reduce intensity) near the expected aircraft center
    if mask_radius_px > 0:
        # Assume aircraft is near center after alignment
        cy, cx = H // 2, W // 2
        # Create grid of coordinates
        y, x = np.ogrid[:H, :W]
        # Create circular mask
        dist_sq = (x - cx)**2 + (y - cy)**2
        mask = dist_sq <= mask_radius_px**2
        # Apply mask: reduce Z-score within radius (e.g., by 75%)
        z_abs_max = np.where(mask, z_abs_max * 0.25, z_abs_max)

    return z_abs_max


def stack_images_multi(image_paths: List[str], output_dir: str, params: Dict) -> Tuple[Dict, Dict]:
    """
    Main stacking pipeline: Loads, aligns, stacks (multiple methods), saves outputs.
    """
    qc = {} # Initialize QC dictionary
    products = {} # Initialize products dictionary

    if not image_paths:
        qc["error"] = "No image paths provided."
        return products, qc

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Load images and find centers (using refactored function)
        images, centers = _load_and_center(image_paths)
        qc["n_frames_input"] = len(image_paths)
        qc["n_frames_loaded"] = len(images)

        if len(images) < 2:
            qc["error"] = f"Stacking failed: Only {len(images)} usable frames loaded."
            return products, qc # Return empty products and QC with error

        # Align images based on detected/fallback centers
        aligned = _align_images(images, centers)
        qc["n_frames_aligned"] = len(aligned)

        if len(aligned) < 2:
            qc["error"] = "Stacking failed: Fewer than 2 images were successfully aligned."
            return products, qc

        # Stack aligned images into a 3D volume [N, H, W]
        vol = np.stack(aligned, axis=0).astype(np.float32) # Ensure float32 for calculations

        # --- Generate Stacked Products ---
        # 1) Simple Mean stack
        mean_stack = np.mean(vol, axis=0)

        # 2) Robust sigma-clipped mean
        clip_z = float(params.get("sigma_clip_z", 3.0)) # Get clipping threshold from params
        robust_stack, clipped_fraction = _sigma_clipped_mean(vol, clip_z=clip_z)
        qc["sigma_clip_z"] = clip_z
        qc["clipped_fraction"] = round(clipped_fraction, 6) # Store clipped fraction

        # 3) Anomaly map
        mask_r = int(params.get("anomaly_mask_radius_px", 20)) # Get mask radius from params
        anomaly = _anomaly_map(vol, mask_radius_px=mask_r)
        qc["anomaly_mask_radius_px"] = mask_r

        # --- Save Outputs (FITS + PNG) ---
        # Define output paths
        mean_fits_path = os.path.join(output_dir, "stack_mean.fits")
        robust_fits_path = os.path.join(output_dir, "stack_robust.fits")
        anomaly_fits_path = os.path.join(output_dir, "stack_anomaly.fits")
        mean_png_path = os.path.join(output_dir, "stack_mean.png")
        robust_png_path = os.path.join(output_dir, "stack_robust.png")
        anomaly_png_path = os.path.join(output_dir, "stack_anomaly.png")

        # Save FITS (convert to uint16 for reasonable size, check data range first?)
        # For anomaly, scale before saving to uint16
        anomaly_scaled_u16 = (np.clip(anomaly / max(1.0, np.percentile(anomaly[np.isfinite(anomaly)], 99.5)), 0, 1) * 65535).astype(np.uint16)

        # Wrap writes in try/except to report errors but attempt all saves
        saved_products = {"mean": {}, "robust": {}, "anomaly": {}}
        try: saved_products["mean"]["fits"] = _write_fits(mean_fits_path, mean_stack.astype(np.uint16))
        except Exception as e: qc.setdefault("save_errors", []).append(f"Mean FITS: {e}")
        try: saved_products["robust"]["fits"] = _write_fits(robust_fits_path, robust_stack.astype(np.uint16))
        except Exception as e: qc.setdefault("save_errors", []).append(f"Robust FITS: {e}")
        try: saved_products["anomaly"]["fits"] = _write_fits(anomaly_fits_path, anomaly_scaled_u16)
        except Exception as e: qc.setdefault("save_errors", []).append(f"Anomaly FITS: {e}")

        # Save PNGs (use appropriate normalization percentiles)
        try: saved_products["mean"]["png"] = _write_png(mean_png_path, mean_stack, 1.0, 99.0)
        except Exception as e: qc.setdefault("save_errors", []).append(f"Mean PNG: {e}")
        try: saved_products["robust"]["png"] = _write_png(robust_png_path, robust_stack, 1.0, 99.0)
        except Exception as e: qc.setdefault("save_errors", []).append(f"Robust PNG: {e}")
        try: saved_products["anomaly"]["png"] = _write_png(anomaly_png_path, anomaly, 5.0, 99.5) # Tighter stretch for anomaly map
        except Exception as e: qc.setdefault("save_errors", []).append(f"Anomaly PNG: {e}")

        # Final check if essential files were saved
        if not saved_products.get("robust", {}).get("fits"):
             qc["error"] = "Stacking failed: Could not save the primary robust FITS file."
             # Return potentially partial products dict along with error
             return saved_products, qc

        # If robust FITS saved, consider it mostly successful
        qc["status"] = "success" if "error" not in qc else "partial_success"
        return saved_products, qc

    except Exception as e:
         # Catch unexpected errors during the main process
        print(f"  - Stacker: Unhandled error in stack_images_multi: {e}")
        qc["error"] = f"Unhandled exception: {e}"
        # Return empty products dict and QC with error
        return products, qc


# Backward-compatibility adapter used by older callers (e.g., stack_orchestrator)
def stack_images(image_paths: List[str], output_dir: str, params: Dict) -> Tuple[Optional[str], Dict]:
    """
    Legacy API wrapper for stack_images_multi.
    Returns the path to the robust FITS stack (master) and QC dictionary.
    Returns (None, qc) on failure, where qc contains an 'error' key.
    """
    products, qc = stack_images_multi(image_paths, output_dir, params)

    # Check if stacking failed (error in QC) or if essential product missing
    if qc.get("error") or not products.get("robust", {}).get("fits"):
        # Log the error if not already logged by stack_images_multi
        if not qc.get("error"): qc["error"] = "Robust FITS output missing after stacking."
        print(f"  - Stacker (Legacy): Stacking failed. Reason: {qc['error']}")
        return None, qc # Return None for path, plus QC dict with error

    # Return the path to the robust FITS file and the QC dict
    return products["robust"]["fits"], qc