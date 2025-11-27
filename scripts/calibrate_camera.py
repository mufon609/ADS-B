# calibrate_camera.py
"""
A script to be run at night to precisely measure camera properties
using plate-solving with ASTAP.
"""
import logging
import math
import os
import subprocess
import time
import sys
from pathlib import Path

# Ensure imports/config work when run from scripts/ by resolving the repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Default config path relative to repo root if env var not set
os.environ.setdefault("ADSB_CONFIG_FILE", str(REPO_ROOT / "config.yaml"))

from config.loader import CONFIG
from hardware_control import IndiController
from utils.logger_config import setup_logging

logger = logging.getLogger(__name__)


def run_calibration():
    """
    Takes a single image of the night sky, plate-solves it, and prints
    the precise camera scale and rotation.
    """
    logger.info("\n--- Camera Calibration Utility ---")
    logger.info(
        "NOTE: Point your telescope at a star-rich area of the night sky before running.")

    try:
        indi = IndiController()

        calib_image_path = os.path.join(
            CONFIG['logging']['log_dir'], 'images', 'calibration_frame.fits')
        logger.info("\nTaking a 10-second calibration image...")
        indi.capture_image(10.0, calib_image_path)
        logger.info(f"Image saved to {calib_image_path}")

        astap_path = "astap"
        wcs_path = os.path.splitext(calib_image_path)[0] + ".wcs"

        logger.info(
            "Attempting to plate-solve the image (this may take up to 60 seconds)...")

        result = subprocess.run(
            [astap_path, "-f", calib_image_path, "-z", "0"],
            timeout=60, check=True, capture_output=True, text=True
        )

        if os.path.exists(wcs_path):
            logger.info("\n--- ✅ Calibration Successful! ---")
            with open(wcs_path, 'r') as f:
                lines = f.readlines()

            cd_matrix = {}
            for line in lines:
                if 'CD1_1' in line:
                    cd_matrix['1_1'] = float(line.split('=')[1])
                if 'CD1_2' in line:
                    cd_matrix['1_2'] = float(line.split('=')[1])
                if 'CD2_1' in line:
                    cd_matrix['2_1'] = float(line.split('=')[1])
                if 'CD2_2' in line:
                    cd_matrix['2_2'] = float(line.split('=')[1])

            # Calculate plate scale and rotation from the WCS CD matrix
            pixel_scale_deg = math.sqrt(
                abs(cd_matrix['1_1'] * cd_matrix['2_2'] - cd_matrix['1_2'] * cd_matrix['2_1']))
            plate_scale_arcsec = pixel_scale_deg * 3600
            rotation_angle_deg = math.degrees(math.atan2(
                cd_matrix['2_1'] - cd_matrix['1_2'], cd_matrix['1_1'] + cd_matrix['2_2']))

            logger.info(
                "\nUpdate your config.yaml 'pointing_calibration' section with these measured values:")
            logger.info("-------------------------------------------------")
            logger.info(f"  plate_scale_arcsec_px: {plate_scale_arcsec:.4f}")
            logger.info(f"  rotation_angle_deg: {rotation_angle_deg:.2f}")
            logger.info("-------------------------------------------------")
        else:
            logger.error(
                "\n--- ❌ Calibration Failed: Plate-solving did not produce a solution. ---")
            logger.error(
                "Try pointing to a different area of the sky or increasing exposure time.")
            logger.error("Solver output:", result.stderr)

    except FileNotFoundError:
        logger.critical(
            "\n--- ❌ Calibration Failed: 'astap' command not found. ---")
        logger.critical(
            "Ensure ASTAP is installed and accessible in your system's PATH.")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.error(
            f"\n--- ❌ Calibration Failed: Plate-solver returned an error. ---")
        logger.error(e.stderr)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}")
    finally:
        if 'indi' in locals() and indi:
            indi.disconnect()


if __name__ == "__main__":
    setup_logging()
    run_calibration()
