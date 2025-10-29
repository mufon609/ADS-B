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
from astropy.stats import sigma_clipped_stats # More robust sigma clipping available

# --- Import necessary internal functions from image_analyzer ---
from image_analyzer import _load_fits_data, _detect_aircraft_from_data


def _normalize_to_png(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Robustly stretch a float image to 0..255 uint8 using percentiles."""
    # Ensure input is float32 for calculations
    arr = np.asarray(arr, dtype=np.float32)
    # Filter out non-finite values for percentile calculation
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        # Return black image if no valid data
        return np.zeros(arr.shape, dtype=np.uint8) # Use arr.shape
    # Calculate percentile limits
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    # Handle flat images where lo == hi
    if hi <= lo:
        hi = lo + 1e-6 # Add tiny epsilon to prevent division by zero
    # Clip and scale data to 0-1 range (handle NaNs before scaling)
    arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo) # Replace non-finite with bounds
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    # Convert to 8-bit unsigned integer
    return (arr * 255.0).astype(np.uint8)


def _write_fits(path: str, data: np.ndarray) -> str:
    """Saves NumPy array to a FITS file, creating directory if needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Ensure data is in a FITS-compatible format (float32, int16/32, uint16)
        if data.dtype not in [np.float32, np.int16, np.int32, np.uint16]:
             # Default to float32 if not already compatible
             data = data.astype(np.float32)

        # Create HDU and write, enable checksum for integrity
        fits.HDUList([fits.PrimaryHDU(data)]).writeto(path, overwrite=True, checksum=True)
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
    """
    Load FITS images (float32) and get aircraft centers using in-memory data.
    Skips frames where detection fails.
    """
    images, centers = [], []
    skipped_count = 0
    for i, p in enumerate(paths):
        img_data = _load_fits_data(p)
        if img_data is None:
            # _load_fits_data already prints warnings/errors
            skipped_count += 1
            continue # Skip this file if loading failed

        original_shape = img_data.shape
        ctr = None
        det = None
        try:
            det = _detect_aircraft_from_data(img_data, original_shape=original_shape)
        except Exception as e:
            print(f"  - Stacker: Detection function failed on frame {i+1} ({os.path.basename(p)}): {e}")
            det = None # Ensure det is None on error

        # Check detection results
        if det and det.get("detected") and det.get("center_px") is not None:
            ctr = det.get("center_px")
            # Basic validation of center coordinates
            if not (isinstance(ctr, (tuple, list)) and len(ctr) == 2 and
                    isinstance(ctr[0], (int, float)) and isinstance(ctr[1], (int, float)) and
                    np.isfinite(ctr[0]) and np.isfinite(ctr[1])): # Check for finite numbers
                print(f"  - Stacker: Invalid center_px format from detection in frame {i+1}: {ctr}. Skipping frame.")
                ctr = None # Invalidate bad center
                skipped_count += 1
        else:
            # Detection failed or didn't return a valid center
            reason = det.get('reason', 'detection_failed') if det else 'load_failed_or_detect_exception'
            print(f"  - Stacker: No valid detection in frame {i+1} ({os.path.basename(p)}). Reason: {reason}. Skipping frame.")
            skipped_count += 1
            ctr = None # Ensure ctr is None if detection failed

        # Only append if detection was successful and center is valid
        if ctr is not None:
            images.append(img_data)
            centers.append(ctr)
        # Removed the fallback to _fallback_center

    if skipped_count > 0:
         print(f"  - Stacker: Skipped {skipped_count} / {len(paths)} frames due to load/detection issues.")

    return images, centers


def _align_images(images: List[np.ndarray], centers: List[Tuple[float, float]]) -> List[np.ndarray]:
    """Translate-align images so the aircraft stays fixed."""
    if len(images) < 2: # Need at least two valid images to align
        return images # Return original list (might be empty or single image)

    # Use the center from the *first valid* image as the reference point
    ref_center = centers[0]
    # Get dimensions from the first valid image
    H, W = images[0].shape
    aligned = []

    for i, (img, ctr) in enumerate(zip(images, centers)):
        # Check if image dimensions match the first image (robustness check)
        if img.shape != (H, W):
            print(f"  - Stacker: Shape mismatch in frame {i} ({img.shape} vs {(H,W)}), skipping alignment for this frame.")
            continue # Skip this frame

        # Calculate shift needed
        dx = ref_center[0] - ctr[0]
        dy = ref_center[1] - ctr[1]

        # Create affine transformation matrix for pure translation
        M = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply the warp (shift)
        shifted = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) # Use reflect_101
        aligned.append(shifted)

    return aligned


def _sigma_clipped_mean(volume: np.ndarray, clip_z: float = 3.0, max_iters: int = 5) -> Tuple[np.ndarray, float]:
    """
    Robust mean via iterative sigma-clipping.
    Uses astropy's sigma_clipped_stats for robustness.
    Returns (stacked_image, fraction_of_pixels_masked).
    """
    if volume.ndim != 3 or volume.shape[0] < 2: # Need N>1 frames along axis 0
        print("  - Stacker Warning: Sigma clipping requires at least 2 frames.")
        # Fallback to simple mean if insufficient data, handle potential NaNs
        return np.nanmean(volume, axis=0) if volume.ndim == 3 else volume, 0.0

    # Use astropy's sigma_clipped_stats along the frame axis (axis=0)
    try:
        # --- FIX: Use 'mask=None' as hinted by your log ---
        mean_stack, median_unused, std_unused = sigma_clipped_stats(
            volume, sigma=clip_z, maxiters=max_iters, axis=0,
            cenfunc='median', stdfunc='mad_std', mask=None # <-- FIX HERE
        )
        # --- END FIX ---
    except TypeError as e:
         # Fallback to 'return_masked_array' if 'mask' also fails
         if "got an unexpected keyword argument 'mask'" in str(e):
             try:
                 print("  - Stacker Warning: 'mask=None' failed, falling back to 'return_masked_array=True'.")
                 mean_stack, median_unused, std_unused = sigma_clipped_stats(
                     volume, sigma=clip_z, maxiters=max_iters, axis=0,
                     cenfunc='median', stdfunc='mad_std', return_masked_array=True
                 )
             except Exception as e2:
                 print(f"  - Stacker Error: Both 'mask' and 'return_masked_array' failed: {e2}. Falling back to simple mean.")
                 return np.nanmean(volume, axis=0), 0.0
         else:
             print(f"  - Stacker Warning: astropy.sigma_clipped_stats failed: {e}. Falling back to simple mean.")
             return np.nanmean(volume, axis=0), 0.0
    except Exception as e:
         print(f"  - Stacker Warning: astropy.sigma_clipped_stats failed: {e}. Falling back to simple mean.")
         return np.nanmean(volume, axis=0), 0.0


    # Calculate the fraction of masked pixels (optional, for QC)
    masked_pixels = np.count_nonzero(mean_stack.mask) if hasattr(mean_stack, 'mask') else 0
    total_pixels = mean_stack.size if hasattr(mean_stack, 'size') else 0
    masked_fraction = (masked_pixels / total_pixels) if total_pixels > 0 else 0.0

    # Fill masked values (pixels clipped in all frames) with the median if needed
    if hasattr(mean_stack, 'filled'):
         # Calculate median separately if needed for filling
         fill_value = np.nanmedian(volume, axis=0) # Calculate median ignoring NaNs
         stacked_filled = mean_stack.filled(fill_value)
    else:
         # If input wasn't masked or stats didn't return mask, result is likely ndarray
         stacked_filled = mean_stack # Assume it's already filled or wasn't masked

    # Ensure output is float32
    return stacked_filled.astype(np.float32), masked_fraction


def _anomaly_map(volume: np.ndarray, mask_radius_px: int = 20) -> np.ndarray:
    """
    Highlights transient objects or background streaks by finding max deviation from median.
    Assumes 'volume' is already aircraft-aligned: [N, H, W].
    """
    if volume.ndim != 3 or volume.shape[0] < 2:
        print("  - Stacker Warning: Anomaly map requires at least 2 frames.")
        return np.zeros_like(volume[0]) if volume.ndim==3 and volume.shape[0]>0 else np.zeros((100,100), dtype=np.float32)

    N, H, W = volume.shape
    # Calculate robust statistics (median and MAD) ignoring NaNs
    med = np.nanmedian(volume, axis=0)
    # Calculate MAD manually ignoring NaNs
    abs_dev = np.abs(volume - med) # Broadcasting med
    mad = np.nanmedian(abs_dev, axis=0)
    # Scale MAD to estimate standard deviation, add epsilon
    sigma_est = 1.4826 * mad + 1e-6

    # Calculate max absolute Z-score per pixel across all frames, ignore NaNs in max
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore potential division by zero or NaN comparison
        z_scores = (volume - med) / sigma_est
    z_abs_max = np.nanmax(np.abs(z_scores), axis=0)
    # Fill any remaining NaNs (pixels NaN in all frames) with 0
    z_abs_max = np.nan_to_num(z_abs_max, nan=0.0)

    # Optional: Mask (reduce intensity) near the expected aircraft center
    if mask_radius_px > 0:
        # Assume aircraft is near center after alignment
        cy, cx = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        dist_sq = (x - cx)**2 + (y - cy)**2
        mask = dist_sq <= mask_radius_px**2
        # Apply mask: reduce Z-score within radius
        z_abs_max = np.where(mask, z_abs_max * 0.25, z_abs_max)

    return z_abs_max.astype(np.float32) # Ensure float32


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

        # Load images and find centers (using refactored function that skips bad frames)
        images, centers = _load_and_center(image_paths)
        qc["n_frames_input"] = len(image_paths)
        qc["n_frames_loaded_detected"] = len(images) # Renamed for clarity

        if len(images) < 2:
            qc["error"] = f"Stacking failed: Only {len(images)} usable frames loaded/detected for alignment."
            return products, qc # Return empty products and QC with error

        # Align images based on detected centers
        aligned = _align_images(images, centers)
        qc["n_frames_aligned"] = len(aligned)

        if len(aligned) < 2:
            qc["error"] = "Stacking failed: Fewer than 2 images were successfully aligned."
            return products, qc

        # Stack aligned images into a 3D volume [N, H, W]
        vol = np.stack(aligned, axis=0).astype(np.float32) # Ensure float32

        # --- Generate Stacked Products ---
        # 1) Simple Mean stack (handle potential NaNs from alignment/input)
        mean_stack = np.nanmean(vol, axis=0).astype(np.float32)

        # 2) Robust sigma-clipped mean
        clip_z = float(params.get("sigma_clip_z", 3.0))
        robust_stack, clipped_fraction = _sigma_clipped_mean(vol, clip_z=clip_z)
        qc["sigma_clip_z"] = clip_z
        qc["clipped_fraction"] = round(clipped_fraction, 6)

        # 3) Anomaly map
        mask_r = int(params.get("anomaly_mask_radius_px", 20))
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

        # --- FIX (#13): Clip data before casting to uint16 ---
        # Clip mean and robust stacks to valid uint16 range (0-65535)
        mean_stack_u16 = np.clip(np.nan_to_num(mean_stack, nan=0), 0, 65535).astype(np.uint16)
        robust_stack_u16 = np.clip(np.nan_to_num(robust_stack, nan=0), 0, 65535).astype(np.uint16)
        # --- END FIX (#13) ---

        # Scale anomaly map for saving (e.g., to uint16) - Use robust scaling
        anomaly_finite = anomaly[np.isfinite(anomaly)]
        if anomaly_finite.size > 0:
             vmin, vmax = np.percentile(anomaly_finite, [1, 99.5])
             if vmax <= vmin: vmax = vmin + 1e-6
             anomaly_scaled_u16 = np.clip((anomaly - vmin) / (vmax - vmin), 0, 1) * 65535
             anomaly_scaled_u16 = np.nan_to_num(anomaly_scaled_u16, nan=0).astype(np.uint16)
        else:
             anomaly_scaled_u16 = np.zeros_like(anomaly, dtype=np.uint16)

        # Wrap writes in try/except to report errors but attempt all saves
        saved_products = {"mean": {}, "robust": {}, "anomaly": {}}
        try: saved_products["mean"]["fits"] = _write_fits(mean_fits_path, mean_stack_u16)
        except Exception as e: qc.setdefault("save_errors", []).append(f"Mean FITS: {e}")
        try: saved_products["robust"]["fits"] = _write_fits(robust_fits_path, robust_stack_u16)
        except Exception as e: qc.setdefault("save_errors", []).append(f"Robust FITS: {e}")
        try: saved_products["anomaly"]["fits"] = _write_fits(anomaly_fits_path, anomaly_scaled_u16)
        except Exception as e: qc.setdefault("save_errors", []).append(f"Anomaly FITS: {e}")

        # Save PNGs (use appropriate normalization percentiles on float data)
        try: saved_products["mean"]["png"] = _write_png(mean_png_path, mean_stack, 1.0, 99.0)
        except Exception as e: qc.setdefault("save_errors", []).append(f"Mean PNG: {e}")
        try: saved_products["robust"]["png"] = _write_png(robust_png_path, robust_stack, 1.0, 99.0)
        except Exception as e: qc.setdefault("save_errors", []).append(f"Robust PNG: {e}")
        try: saved_products["anomaly"]["png"] = _write_png(anomaly_png_path, anomaly, 5.0, 99.5) # Tighter stretch for anomaly
        except Exception as e: qc.setdefault("save_errors", []).append(f"Anomaly PNG: {e}")

        # Final check if essential files were saved (robust FITS is primary)
        if not saved_products.get("robust", {}).get("fits"):
            qc["error"] = "Stacking failed: Could not save the primary robust FITS file."
            return saved_products, qc

        qc["status"] = "success" if "error" not in qc and not qc.get("save_errors") else "partial_success"
        return saved_products, qc

    except Exception as e:
        print(f"  - Stacker: Unhandled error in stack_images_multi: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        qc["error"] = f"Unhandled exception: {e}"
        return products, qc


# Backward-compatibility adapter used by older callers (e.g., stack_orchestrator)
def stack_images(image_paths: List[str], output_dir: str, params: Dict) -> Tuple[Optional[str], Dict]:
    """
    Legacy API wrapper for stack_images_multi.
    Returns the path to the robust FITS stack (master) and QC dictionary.
    Returns (None, qc) on failure, where qc contains an 'error' key.
    """
    products, qc = stack_images_multi(image_paths, output_dir, params)

    if qc.get("error") or not products.get("robust", {}).get("fits"):
        if not qc.get("error"): qc["error"] = "Robust FITS output missing after stacking."
        print(f"  - Stacker (Legacy): Stacking failed. Reason: {qc.get('error', 'Unknown error')}")
        return None, qc

    return products["robust"]["fits"], qc