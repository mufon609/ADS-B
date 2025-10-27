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
        with fits.open(image_path) as hdul:
            img_data = hdul[0].data

        if img_data is None or img_data.ndim != 2:
            print(f"Warning: Invalid data dimensions in {image_path}. Expected 2D, got {img_data.ndim if img_data is not None else 'None'}.")
            return None

        # Ensure data is float32 and handle non-finite values
        img_data = np.nan_to_num(img_data, copy=True).astype(np.float32) # Use copy=True to avoid modifying cache
        return img_data
    except FileNotFoundError:
        print(f"Warning: FITS file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error loading FITS file {image_path}: {e}")
        return None

def _normalize_zscale(image_data: np.ndarray) -> Optional[np.ndarray]:
    """Applies ZScale normalization to float32 data."""
    try:
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
        return None

def _normalize_to_8bit(normed_float_data: np.ndarray) -> Optional[np.ndarray]:
    """Normalizes float data (expected range 0-1 or ZScale output) to uint8 (0-255)."""
    try:
        # Use cv2.normalize for robust min-max scaling to 0-255
        img_8bit = cv2.normalize(normed_float_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return img_8bit
    except Exception as e:
        print(f"Error converting to 8-bit: {e}")
        return None

# === Internal Processing Functions (operate on NumPy arrays) ===

def _measure_sharpness_from_data(image_data: np.ndarray) -> float:
    """Measures sharpness using Laplacian variance on ZScale normalized data."""
    normed_data = _normalize_zscale(image_data)
    if normed_data is None:
        return 0.0
    img_8bit = _normalize_to_8bit(normed_data)
    if img_8bit is None:
        return 0.0
    try:
        # Use CV_64F for higher precision before taking variance
        return cv2.Laplacian(img_8bit, cv2.CV_64F).var()
    except Exception as e:
        print(f"Error calculating Laplacian variance: {e}")
        return 0.0

def _estimate_exposure_adjustment_from_data(image_data: np.ndarray) -> float:
    """Estimates exposure adjustment factor from ZScale normalized data."""
    capture_cfg = CONFIG['capture']

    # --- FIX: Use ZScale normalization BEFORE converting to 8-bit ---
    normed_data = _normalize_zscale(image_data)
    if normed_data is None: return 1.0 # Return neutral on error
    img_8bit = _normalize_to_8bit(normed_data)
    if img_8bit is None: return 1.0
    # --- End of FIX ---

    try:
        # Calculate histogram on the robust 8-bit image
        hist = cv2.calcHist([img_8bit], [0], None, [256], [0, 256])
        # Calculate weighted mean brightness from histogram, avoid division by zero
        total_pixels = img_8bit.size
        if total_pixels == 0: return 1.0
        # Ensure weights sum is not zero before averaging
        hist_sum = hist.sum()
        if hist_sum <= 0: return 1.0 # Avoid division by zero if histogram is empty

        current_mean = max(1.0, float(np.average(np.arange(256), weights=hist.flatten())))

        # Get target brightness from config
        target = int(np.clip(capture_cfg['target_brightness'], 1, 250)) # Ensure target is valid

        # Calculate initial adjustment factor
        adjustment = target / current_mean

        # Reduce exposure aggressively if image is saturated
        saturated_pixels = (img_8bit >= 250).mean() # Check saturation on the robust 8-bit image
        if saturated_pixels > 0.10: # More than 10% saturated
              adjustment = min(adjustment, 0.5) # Reduce exposure by at least half
        elif saturated_pixels > 0.01: # More than 1% saturated
            adjustment = min(adjustment, 0.8) # Reduce exposure slightly

        # Clip adjustment factor to configured min/max limits and return
        adj_min = capture_cfg.get('exposure_adjust_factor_min', 0.1) # Added default
        adj_max = capture_cfg.get('exposure_adjust_factor_max', 10.0) # Added default
        return float(np.clip(adjustment, adj_min, adj_max))

    except Exception as e:
        print(f"Error estimating exposure adjustment: {e}")
        return 1.0 # Return neutral factor on error

def _detect_aircraft_from_data(image_data: np.ndarray, original_shape: Tuple[int, int]) -> Dict:
    """Internal aircraft detection logic operating on float32 NumPy array."""
    try:
        orig_h, orig_w = original_shape

        # --- Calculate Sharpness Consistently ---
        # Calculate sharpness on the full-resolution data using the robust method
        sharpness = _measure_sharpness_from_data(image_data)
        det_cfg = CONFIG['capture']['detection']
        if sharpness < det_cfg['sharpness_min']:
            return {'detected': False, 'reason': 'blurry', 'sharpness': sharpness}

        # --- Normalization and Resizing for Detection ---
        # Use simple min-max for speed in detection, applied AFTER sharpness calc
        img_8bit_detect = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize large images for performance
        scale = 1.0
        img_8bit_scaled = img_8bit_detect
        if max(orig_h, orig_w) > 3000:
            scale = 2048.0 / max(orig_h, orig_w)
            img_8bit_scaled = cv2.resize(img_8bit_detect, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)

        # Check for blank or extremely low contrast images early
        if img_8bit_scaled.std() < 2.0:
            return {'detected': False, 'reason': 'low_contrast_or_blank', 'sharpness': sharpness}

        # --- Thresholding and Contouring ---
        _, thresh = cv2.threshold(img_8bit_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k = max(1, int(round(3 * scale))) # Kernel size ~3 pixels in original scale
        kernel = np.ones((k, k), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        if not np.any(thresh) or np.all(thresh == 255):
            return {'detected': False, 'reason': 'threshold_all_or_none', 'sharpness': sharpness}

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1] # Handle OpenCV version differences

        if not contours: return {'detected': False, 'reason': 'no_contours', 'sharpness': sharpness}

        # --- Score contours: favor large area near center ---
        h_s, w_s = img_8bit_scaled.shape[:2]
        cx0, cy0 = w_s * 0.5, h_s * 0.5 # Center of scaled image

        def score_contour(cnt):
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M['m00'] == 0: return -np.inf # Use -inf for invalid moments
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            dist_sq = (cx - cx0)**2 + (cy - cy0)**2 # Distance from center squared
            # Score favors area, penalizes distance from center
            return area - (0.005 * dist_sq) # 0.005 is a tunable penalty factor

        best_contour = max(contours, key=score_contour)

        # --- Check best contour properties ---
        area = cv2.contourArea(best_contour)
        scaled_threshold = det_cfg['threshold_area_px'] * (scale * scale) # Adjust threshold for scaled image
        if area < scaled_threshold:
            return {'detected': False, 'reason': 'too_small', 'area': area, 'threshold': scaled_threshold, 'sharpness': sharpness}

        M = cv2.moments(best_contour)
        if M['m00'] == 0: return {'detected': False, 'reason': 'invalid_moments', 'sharpness': sharpness}

        # Centroid of the best contour in *scaled* coordinates
        cx_s = M['m10'] / M['m00']
        cy_s = M['m01'] / M['m00']

        # Convert centroid back to *full resolution* coordinates
        cx_full = int(round(cx_s / scale))
        cy_full = int(round(cy_s / scale))
        # Clamp coordinates to be within original image bounds
        cx_full = max(0, min(orig_w - 1, cx_full))
        cy_full = max(0, min(orig_h - 1, cy_full))

        # --- Calculate confidence ---
        # Rough confidence based on area fraction and sharpness
        area_frac = max(area / float(img_8bit_scaled.size), 1e-7) # Avoid log(0) if area is tiny
        # Combine area fraction (log scale) and normalized sharpness
        # Adjust weights as needed
        confidence = (np.log10(area_frac * 1e4 + 1)) * 0.3 + min(sharpness / 200.0, 1.0) * 0.7
        confidence = np.clip(confidence, 0.0, 1.0) # Ensure confidence is between 0 and 1

        if confidence < det_cfg['confidence_min']:
            return {'detected': False, 'reason': 'low_confidence', 'sharpness': sharpness, 'confidence': confidence}

        return {'detected': True, 'center_px': (cx_full, cy_full), 'confidence': confidence, 'sharpness': sharpness}

    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"Error during aircraft detection: {e}")
        return {'detected': False, 'reason': 'processing_error', 'error': str(e)}

def _save_png_preview_from_data(image_data: np.ndarray, png_path: str) -> str:
    """Creates and saves a PNG preview from float32 NumPy data using ZScale."""
    normed_data = _normalize_zscale(image_data)
    if normed_data is None: return ""
    img_8bit = _normalize_to_8bit(normed_data)
    if img_8bit is None: return ""

    try:
        # Optional: Apply histogram equalization for extra contrast in previews
        img_8bit = cv2.equalizeHist(img_8bit)

        # Ensure directory exists
        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        # Save the image
        if not cv2.imwrite(png_path, img_8bit):
             raise IOError(f"Failed to write PNG file to {png_path}")

        return png_path
    except Exception as e:
        print(f"Error saving PNG preview to {png_path}: {e}")
        return ""

# === Public Functions (operate on file paths) ===

def measure_sharpness(image_path: str) -> float:
    """Measures image sharpness, using ZScale normalization for robustness."""
    if CONFIG['development']['dry_run']:
        if "autofocus" in os.path.basename(image_path):
             # Ensure defaults exist in config for dry run
            return CONFIG.get('capture', {}).get('autofocus', {}).get('sharpness_threshold', 20.0) + 100
        return CONFIG.get('capture', {}).get('detection', {}).get('sharpness_min', 10.0) + 1

    img_data = _load_fits_data(image_path)
    if img_data is None:
        return 0.0
    return _measure_sharpness_from_data(img_data)

def estimate_exposure_adjustment(image_path: str, current_exposure_s: float) -> float:
    """Estimates a pure multiplicative exposure adjustment factor using robust ZScale normalization."""
    if CONFIG['development']['dry_run']:
        return 1.0

    # Basic validation for current exposure
    if not (isinstance(current_exposure_s, (int, float)) and current_exposure_s > 0):
        print(f"Warning: Invalid current_exposure_s ({current_exposure_s}) passed to estimate_exposure_adjustment.")
        return 1.0

    img_data = _load_fits_data(image_path)
    if img_data is None:
        return 1.0 # Return neutral on load error
    return _estimate_exposure_adjustment_from_data(img_data)


def detect_aircraft(image_path: str, sim_initial_error_s: float = 0.0) -> dict:
    """Loads FITS image, detects aircraft-like blobs, and returns results including sharpness."""
    if CONFIG['development']['dry_run']:
        # Keep dry run simulation logic here in the public function
        specs = CONFIG['camera_specs']
        sharpness = CONFIG.get('capture', {}).get('detection', {}).get('sharpness_min', 10.0) + 1

        # Simulate initial pointing error for dry run
        error_px = float(sim_initial_error_s or 0.0) * 20.0 # Arbitrary pixels/sec factor
        max_offset = 150.0 # Limit simulated offset
        dx = min(error_px, max_offset)
        dy = -min(error_px, max_offset) # Simulate diagonal error

        # Calculate center based on potentially missing config keys
        width = specs.get('resolution_width_px', 640) # Default if missing
        height = specs.get('resolution_height_px', 480) # Default if missing
        cx = width  / 2.0 + dx
        cy = height / 2.0 + dy

        return {'detected': True, 'center_px': (cx, cy), 'confidence': 0.95, 'sharpness': sharpness}

    # Load data using the helper
    img_data = _load_fits_data(image_path)
    if img_data is None:
        return {'detected': False, 'reason': 'load_error'}

    # Call the internal detection function
    return _detect_aircraft_from_data(img_data, original_shape=img_data.shape)

def save_png_preview(fits_path: str) -> str:
    """
    Creates a PNG preview from a FITS file using ZScale for good contrast.
    Returns the path to the PNG file, or an empty string on failure.
    """
    # Handle dry run within the public function
    if CONFIG['development']['dry_run']:
        # In dry run, capture_image already created a dummy PNG. Just return its path.
        # Ensure the file actually exists before returning path? Optional.
        png_path = os.path.splitext(fits_path)[0] + ".png"
        # Check if the dummy PNG was actually created by capture_image's dry run
        if os.path.exists(png_path):
             return png_path
        else:
             # If called independently (e.g., from stacker) and file doesn't exist, return empty
             print(f"[DRY RUN] Warning: Dummy PNG not found for {fits_path}, returning empty path.")
             return ""

    if not os.path.exists(fits_path):
         print(f"Warning: FITS file not found for PNG preview: {fits_path}")
         return ""

    # Load data
    img_data = _load_fits_data(fits_path)
    if img_data is None:
        return "" # Return empty if loading failed

    # Construct PNG path
    png_path = os.path.splitext(fits_path)[0] + ".png"

    # Call internal save function
    return _save_png_preview_from_data(img_data, png_path)