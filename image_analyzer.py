#image_analyzer.py
"""
Module for analyzing images to detect aircraft, measure sharpness,
estimate exposure, and create previews. Functions operate on file paths
by loading data, but internal helpers operating on NumPy arrays are available.
"""

import cv2
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import os
from typing import Optional, Dict, Tuple
from config_loader import CONFIG

# === Internal Helper Functions ===

def _load_fits_data(image_path: str) -> Optional[np.ndarray]:
    """Loads FITS data, validates, handles NaNs, converts to float32."""
    try:
        # Using memmap=False might be safer for frequent access/modification?
        with fits.open(image_path, memmap=False) as hdul:
            # Check if HDUList is valid and if the primary HDU exists
            if not hdul or len(hdul) == 0:
                 print(f"Warning: Invalid or empty FITS file: {image_path}.")
                 return None
            # Check if the data attribute itself exists and is not None
            img_data = hdul[0].data
            if img_data is None:
                 print(f"Warning: No data found in primary HDU of {image_path}.")
                 return None

        # Check dimensions AFTER confirming data is not None
        if img_data.ndim != 2:
            print(f"Warning: Invalid data dimensions in {image_path}. Expected 2D, got {img_data.ndim}.")
            return None

        # Ensure data is float32 and handle non-finite values
        img_data = np.nan_to_num(img_data, copy=False).astype(np.float32)
        return img_data
    except FileNotFoundError:
        print(f"Warning: FITS file not found: {image_path}")
        return None
    except Exception as e:
        # Catch other errors during FITS loading
        print(f"Error loading FITS file {image_path}: {e}")
        return None

def _normalize_zscale(image_data: np.ndarray) -> Optional[np.ndarray]:
    """Applies ZScale normalization to float32 data."""
    try:
        # Check for empty or non-numeric data before normalization
        if image_data.size == 0 or not np.issubdtype(image_data.dtype, np.number):
             print("Warning: Cannot normalize empty or non-numeric data.")
             return None
        # ZScale can fail on flat images, handle this
        min_val, max_val = np.nanmin(image_data), np.nanmax(image_data)
        if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val == max_val:
            # Return a constant array if image is flat or all non-finite
             return np.zeros_like(image_data, dtype=np.float32)

        norm = ImageNormalize(interval=ZScaleInterval())
        normed_data = norm(image_data)
        # Handle potential masked arrays from astropy
        if hasattr(normed_data, 'mask'):
            # Fill masked values with median of non-masked, or 0 if all masked
            fill_val = np.ma.median(normed_data) if not np.all(normed_data.mask) else 0
            normed_data = np.ma.filled(normed_data, fill_value=fill_val)

        # Ensure output is float32
        return normed_data.astype(np.float32)
    except Exception as e:
        print(f"Error during ZScale normalization: {e}")
        # Return zeros on error
        return np.zeros_like(image_data, dtype=np.float32)

def _normalize_to_8bit(normed_float_data: np.ndarray) -> Optional[np.ndarray]:
    """Normalizes float data (expected range 0-1 or ZScale output) to uint8 (0-255)."""
    try:
        # Check input data validity
        if normed_float_data is None or normed_float_data.size == 0:
            return None
        # Use cv2.normalize for robust min-max scaling to 0-255
        img_8bit = cv2.normalize(normed_float_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return img_8bit
    except Exception as e:
        print(f"Error converting to 8-bit: {e}")
        return None

# === Internal Processing Functions (operate on NumPy arrays) ===

def _measure_sharpness_from_data(image_data: np.ndarray) -> float:
    """
    Measures sharpness using Laplacian variance.
    """
    if image_data is None or not np.any(np.isfinite(image_data)):
        return 0.0
    img_8bit = cv2.normalize(np.nan_to_num(image_data), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img_8bit is None:
        return 0.0
    try:
        # Use CV_64F for higher precision before taking variance
        laplacian = cv2.Laplacian(img_8bit, cv2.CV_64F)
        if not np.all(np.isfinite(laplacian)):
             laplacian = np.nan_to_num(laplacian)
        return laplacian.var()
    except Exception as e:
        print(f"Error calculating Laplacian variance: {e}")
        return 0.0

def _estimate_exposure_adjustment_from_data(image_data: np.ndarray) -> float:
    """
    Estimates exposure adjustment factor from the RAW float32 image data,
    comparing its median to a scaled 16-bit target.
    """
    capture_cfg = CONFIG['capture']
    try:
        if image_data is None or image_data.size == 0:
             return 1.0 # Neutral factor

        finite_data = image_data[np.isfinite(image_data)]
        if finite_data.size == 0:
             print("Warning: Exposure adjustment based on image with no finite pixels. Returning neutral.")
             return 1.0

        current_median = float(np.median(finite_data))

        target_8bit = int(np.clip(capture_cfg.get('target_brightness', 128), 1, 254))
        # Scale target to 16-bit range
        target_16bit = (target_8bit / 255.0) * 65535.0

        if current_median < 1.0: # Image is black
             return capture_cfg.get('exposure_adjust_factor_max', 10.0)

        adjustment = target_16bit / current_median

        saturation_level = 65000.0
        saturated_pixels_fraction = np.mean(finite_data >= saturation_level)

        if saturated_pixels_fraction > 0.10:
            adjustment = min(adjustment, 0.5)
            # print("  - Exposure Adjust: High saturation detected (>10%), reducing factor.") # Optional: less verbose
        elif saturated_pixels_fraction > 0.01:
            adjustment = min(adjustment, 0.8)
            # print("  - Exposure Adjust: Saturation detected (>1%), reducing factor slightly.")

        adj_min = float(capture_cfg.get('exposure_adjust_factor_min', 0.1))
        adj_max = float(capture_cfg.get('exposure_adjust_factor_max', 10.0))
        final_adjustment = float(np.clip(adjustment, adj_min, adj_max))

        return final_adjustment
    except Exception as e:
        print(f"Error estimating exposure adjustment: {e}")
        return 1.0 # Return neutral factor on error

def _detect_aircraft_from_data(image_data: np.ndarray, original_shape: Tuple[int, int]) -> Dict:
    """Internal aircraft detection logic operating on float32 NumPy array."""
    
    if CONFIG['development']['dry_run']:
        # This function is called by the stacker. In dry run, the image
        # is just text. Bypass detection and return a high-confidence
        # detection at the center so stacking can proceed.
        h, w = original_shape
        # Calculate real sharpness of the flat+text image
        sharpness = _measure_sharpness_from_data(image_data)
        return {'detected': True, 'center_px': (w / 2.0, h / 2.0), 'confidence': 0.95, 'sharpness': sharpness}
    
    try:
        if image_data is None or image_data.size == 0:
            return {'detected': False, 'reason': 'invalid_input_data'}
        
        orig_h, orig_w = original_shape

        sharpness = _measure_sharpness_from_data(image_data) # Uses patched function
        det_cfg = CONFIG['capture']['detection']
        sharpness_min_cfg = float(det_cfg.get('sharpness_min', 10.0))
        if sharpness < sharpness_min_cfg:
            return {'detected': False, 'reason': f'blurry (sharpness {sharpness:.1f} < {sharpness_min_cfg:.1f})', 'sharpness': sharpness}

        if not np.any(np.isfinite(image_data)):
             return {'detected': False, 'reason': 'all_non_finite', 'sharpness': sharpness}
        img_8bit_detect = cv2.normalize(np.nan_to_num(image_data), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        scale = 1.0
        img_8bit_scaled = img_8bit_detect
        max_dim_detect = 2048.0
        current_max_dim = max(orig_h, orig_w)
        if current_max_dim > max_dim_detect:
            scale = max_dim_detect / current_max_dim
            target_w = int(orig_w * scale)
            target_h = int(orig_h * scale)
            if target_w > 0 and target_h > 0:
                img_8bit_scaled = cv2.resize(img_8bit_detect, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                 print(f"Warning: Image scaling resulted in zero dimension (scale={scale}). Using original.")
                 scale = 1.0

        std_dev = img_8bit_scaled.std()
        if std_dev < 2.0:
            return {'detected': False, 'reason': f'low_contrast_or_blank (std_dev={std_dev:.1f})', 'sharpness': sharpness}

        otsu_thresh_val, thresh = cv2.threshold(img_8bit_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k_size = max(1, int(round(3 * scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        if not np.any(thresh):
            return {'detected': False, 'reason': 'threshold_empty', 'sharpness': sharpness}
        if np.all(thresh == 255):
             return {'detected': False, 'reason': 'threshold_all_white', 'sharpness': sharpness}

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]

        if not contours: return {'detected': False, 'reason': 'no_contours_found', 'sharpness': sharpness}

        h_s, w_s = img_8bit_scaled.shape[:2]
        cx0, cy0 = w_s * 0.5, h_s * 0.5

        def score_contour(cnt):
            area = cv2.contourArea(cnt)
            if area < 1: return -np.inf
            M = cv2.moments(cnt)
            if M.get('m00', 0) == 0: return -np.inf
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            dist_sq = (cx - cx0)**2 + (cy - cy0)**2
            max_dist_sq = (w_s*0.5)**2 + (h_s*0.5)**2 + 1e-6
            dist_penalty = dist_sq / max_dist_sq
            return area * (1.0 - 0.5 * dist_penalty)

        best_contour = max(contours, key=score_contour)
        best_score = score_contour(best_contour)
        if not np.isfinite(best_score) or best_score <= 0:
             return {'detected': False, 'reason': 'no_valid_contours', 'sharpness': sharpness}

        area = cv2.contourArea(best_contour)
        threshold_area_px_cfg = float(det_cfg.get('threshold_area_px', 50.0))
        scaled_threshold = threshold_area_px_cfg * (scale * scale)
        if area < scaled_threshold:
            return {'detected': False, 'reason': f'best_contour_too_small ({area:.1f} < {scaled_threshold:.1f} px^2)', 'area': area, 'threshold': scaled_threshold, 'sharpness': sharpness}

        M = cv2.moments(best_contour)
        if M.get('m00', 0) == 0: return {'detected': False, 'reason': 'invalid_moments_on_best', 'sharpness': sharpness}

        cx_s = M['m10'] / M['m00']
        cy_s = M['m01'] / M['m00']
        if scale == 0: return {'detected': False, 'reason': 'invalid_scale_factor', 'sharpness': sharpness}
        cx_full = int(round(cx_s / scale))
        cy_full = int(round(cy_s / scale))
        cx_full = max(0, min(orig_w - 1, cx_full))
        cy_full = max(0, min(orig_h - 1, cy_full))

        area_factor = np.log1p(area / scaled_threshold) / np.log1p(100)
        sharp_factor = np.log1p(sharpness / sharpness_min_cfg) / np.log1p(10)
        confidence = 0.6 * np.clip(area_factor, 0, 1) + 0.4 * np.clip(sharp_factor, 0, 1)
        confidence = np.clip(confidence, 0.0, 1.0)

        confidence_min_cfg = float(det_cfg.get('confidence_min', 0.5))
        if confidence < confidence_min_cfg:
            return {'detected': False, 'reason': f'low_confidence ({confidence:.2f} < {confidence_min_cfg:.2f})', 'sharpness': sharpness, 'confidence': confidence}

        return {'detected': True, 'center_px': (cx_full, cy_full), 'confidence': confidence, 'sharpness': sharpness}

    except Exception as e:
        print(f"Error during aircraft detection: {e}")
        import traceback
        traceback.print_exc()
        return {'detected': False, 'reason': 'processing_error', 'error': str(e)}

def _save_png_preview_from_data(image_data: np.ndarray, png_path: str) -> str:
    """Creates and saves a PNG preview from float32 NumPy data."""
    if image_data is None: return ""

    # ZScale fails on high-contrast synthetic images (flat + text)
    if not np.any(np.isfinite(image_data)):
        img_8bit = np.zeros(image_data.shape[:2], dtype=np.uint8)
    else:
        img_8bit = cv2.normalize(np.nan_to_num(image_data), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img_8bit is None: return ""

    try:
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        success = cv2.imwrite(png_path, img_8bit)
        if not success:
            raise IOError(f"Failed to write PNG file to {png_path} (cv2.imwrite returned False)")
        return png_path
    except Exception as e:
        print(f"Error saving PNG preview to {png_path}: {e}")
        return ""

# === Public Functions (operate on file paths) ===

def measure_sharpness(image_path: str) -> float:
    """Measures image sharpness, using robust min-max normalization."""
    if CONFIG['development']['dry_run']:
        if "autofocus" in os.path.basename(image_path):
             # Keep autofocus simulation logic
            return CONFIG.get('capture', {}).get('autofocus', {}).get('sharpness_threshold', 20.0) + 100
        # For other calls (like stacker), fall through to real calculation
        pass

    img_data = _load_fits_data(image_path)
    if img_data is None:
        return 0.0
    return _measure_sharpness_from_data(img_data) # This now uses the patched (min-max) function

def estimate_exposure_adjustment(image_path: str, current_exposure_s: float) -> float:
    """
    DEPRECATED? Main logic now calls _estimate_exposure_adjustment_from_data directly.
    Keeping this public function for potential external use or testing.
    """
    if CONFIG['development']['dry_run']:
        return 1.0 # Return neutral for dry run

    if not (isinstance(current_exposure_s, (int, float)) and current_exposure_s > 0):
        print(f"Warning: Invalid current_exposure_s ({current_exposure_s}) passed to estimate_exposure_adjustment.")

    img_data = _load_fits_data(image_path)
    if img_data is None:
        return 1.0
    return _estimate_exposure_adjustment_from_data(img_data) # Calls patched 16-bit function


def detect_aircraft(image_path: str, sim_initial_error_s: float = 0.0) -> dict:
    """Loads FITS image, detects aircraft-like blobs, and returns results including sharpness."""
    
    # This public function is called by main.py's guide loop for its *initial* simulation
    if CONFIG['development']['dry_run'] and sim_initial_error_s > 0:
        specs = CONFIG['camera_specs']
        sharpness = CONFIG.get('capture', {}).get('detection', {}).get('sharpness_min', 10.0) + 50
        simulated_confidence = 0.95
        error_px = float(sim_initial_error_s or 0.0) * 20.0
        max_offset = 150.0
        dx = min(error_px * np.random.choice([-1, 1]), max_offset * np.sign(error_px))
        dy = 0
        width = specs.get('resolution_width_px', 640)
        height = specs.get('resolution_height_px', 480)
        cx = width  / 2.0 + dx; cy = height / 2.0 + dy
        cx = max(0, min(width - 1, cx)); cy = max(0, min(height - 1, cy))
        return {'detected': True, 'center_px': (cx, cy), 'confidence': simulated_confidence, 'sharpness': sharpness}

    # --- Real Hardware Logic ---
    # This path is used when not in dry run OR when called by stacker in dry run
    img_data = _load_fits_data(image_path)
    if img_data is None:
        return {'detected': False, 'reason': 'load_error'}
    
    # This call now correctly handles dry run logic internally (returns fake center)
    return _detect_aircraft_from_data(img_data, original_shape=img_data.shape)

def save_png_preview(fits_path: str, png_path_out: Optional[str] = None) -> str:
    """
    Creates a PNG preview from a FITS file.
    Uses ZScale for real images, min-max for dry run.
    """
    if not os.path.exists(fits_path):
        print(f"Warning: FITS file not found for PNG preview: {fits_path}")
        return ""

    img_data = _load_fits_data(fits_path)
    if img_data is None: return ""

    if png_path_out is None:
        png_path = os.path.splitext(fits_path)[0] + ".png"
    else:
        png_path = png_path_out

    # The _save_png_preview_from_data function now uses min-max
    # We will call that directly.
    return _save_png_preview_from_data(img_data, png_path)

