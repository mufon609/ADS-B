# ADS-B Aircraft Tracking Program

## Setup
1.  **Install System Dependencies (ASTAP)**: This project requires the ASTAP plate-solver for an optional (but highly recommended) camera calibration step.
    * Download and install the ASTAP program and a star catalog (e.g., H17 or D50) from [the official website](http://www.hnsky.org/astap.htm). Choose the versions for your operating system.

2.  **Install Python Dependencies**: `pip install -r requirements.txt`

3.  **Update `config.yaml`**: Set the `json_file_path`, your observer location, and hardware details.

4.  **Calibrate Camera (One-Time, at Night)**:
    * Point your telescope at a clear, star-rich part of the night sky.
    * Ensure your INDI server is running.
    * Run the calibration script: `python calibrate_camera.py`
    * Copy the `plate_scale_arcsec_px` and `rotation_angle_deg` output and paste them into your `config.yaml` file.

5.  **Calibrate Pointing**: Perform the one-time physical test to determine the correct `az_offset_sign` and `el_offset_sign` values in `config.yaml`.

6.  **Run the Program**:
    * Start the tracker: `python main.py`
    * In a second terminal, start the dashboard: `uvicorn dashboard.server:app --reload --port 8000`

## Operation
The program monitors ADS-B data, predicts flight paths, selects a target using an intelligent scoring and queuing system, and then enters a closed-loop optical guiding mode. The web dashboard provides real-time status, a list of queued targets, and previews of the latest guide and capture images. To stop the program, press `Ctrl+C` in its terminal.

## Evaluation
After a session, you can run `python evaluator.py` to analyze the accuracy of the dead reckoning predictions against the logged flight data.