# ADS-B: Automated Aircraft Tracking & Optical Imaging System

**Real-time ADS-B → predictive slewing → closed-loop optical guiding → high-SNR stacked images**

This system turns any INDI-controlled astronomical mount and camera into a fully autonomous aircraft tracker. It ingests live dump1090-compatible `aircraft.json`, selects optimal targets using a physics-based Expected Value model, slews to intercept, maintains sub-pixel lock via real-time centroid guiding, and produces aligned, sigma-clipped stacked images.

## Features
- Live ADS-B ingestion (`aircraft.json` hot-file polling)
- Expected-Value target scoring with configurable weights and constraints
- Full INDI client implementation (mount, camera, focuser)
- Closed-loop guiding using blob centroid detection and millisecond pulses
- Automatic burst capture with adaptive exposure control
- Background alignment + robust sigma-clipped mean + anomaly map stacking
- FastAPI web dashboard with live guide frames, target queue, gallery, and manual override
- Complete dry-run simulation (no hardware required)
- One-time ASTAP-based plate scale & rotation calibration
- Post-session dead-reckoning accuracy evaluation (`evaluator.py`)

## Quick Start – Simulation Mode
```bash
git clone https://github.com/mufon609/ADS-B.git
cd ADS-B
pip install -r requirements.txt

# Terminal 1 – simulated ADS-B feed
python fake_dump1090_1hz.py --out logs/aircraft.json &

# Terminal 2 – main tracker (dry_run: true by default)
python main.py &

# Terminal 3 – dashboard
uvicorn dashboard.server:app --reload --port 8000
```
Open http://localhost:8000

## Installation

### Prerequisites
- Python 3.10+
- INDI server + drivers (`indi-full`, `indi-3rdparty` on Debian/Ubuntu)
- ASTAP + star database (H17/D50) – required for `calibrate_camera.py`

### Python packages
```bash
pip install -r requirements.txt
```

### Production hardware
```bash
indiserver -v indi_zwo_am5 indi_zwo_asi   # example
```

## Configuration – config.yaml (complete reference)

```yaml
observer:
  latitude_deg: 40.7128
  longitude_deg: -74.0060
  altitude_m: 10

hardware:
  indi_host: localhost
  indi_port: 7624
  mount_device_name: "ZWO AM5"
  camera_device_name: "ZWO Camera 1"
  focuser_device_name: null          # or "ZWO EAF"

camera_specs:
  focal_length_mm: 400
  pixel_size_um: 3.8
  resolution_width_px: 4656
  resolution_height_px: 3520
  binning: 1
  gain: 100
  offset: 10
  cooling_enabled: false
  target_temperature: -10

adsb:
  json_file_path: "logs/aircraft.json"

selection:
  min_elevation_deg: 8
  max_range_nm: 80
  min_sun_separation_deg: 15
  preempt_factor: 1.3
  weights:
    elevation: 0.45
    distance: 0.10
    angular_speed: 0.20
    solar_separation: 0.25

pointing_calibration:
  plate_scale_arcsec_px: null       # filled by calibrate_camera.py
  rotation_angle_deg: null          # filled by calibrate_camera.py
  az_offset_sign: 1                 # ±1 – determined manually
  el_offset_sign: 1                 # ±1 – determined manually

development:
  dry_run: false
```

## One-Time Calibration

1. Point at dense star field at night
2. Start INDI server
3. Run:
   ```bash
   python calibrate_camera.py
   ```
4. Paste `plate_scale_arcsec_px` and `rotation_angle_deg` into `config.yaml`
5. Manually test mount pulses → set `az_offset_sign` / `el_offset_sign` (±1)

## Production Workflow
```bash
# 1. ADS-B feed → logs/aircraft.json (real or fake_dump1090_1hz.py)
# 2. python main.py &
# 3. uvicorn dashboard.server:app --reload --port 8000
```

## Core Operation Loop (main.py)

1. Scan `aircraft.json` every 0.5 s → recompute Expected Value for all aircraft
2. Select highest-EV target → dead-reckon intercept Az/El 5–12 s ahead
3. Slew mount to intercept
4. Begin guide loop:
   - Capture short-exposure frame
   - Detect brightest blob centroid
   - Compute pixel offset from frame center
   - Issue corrective RA/Dec pulse if outside deadzone
5. When centroid stable → trigger burst sequence
6. `stack_orchestrator.py` queues raw frames → `stacker.py` produces aligned stack + anomaly map

## Project Structure
```
.
├── main.py                  # State machine & thread orchestration
├── config.yaml              # Full configuration
├── adsb/
│   ├── aircraft_selector.py # EV scoring & filtering
│   ├── data_reader.py       # ADS-B ingestion + sanitization
│   └── dead_reckoning.py    # Position prediction / extrapolation
├── hardware_control.py      # INDI client wrapper
├── imaging/
│   ├── image_analyzer.py    # Blob detection, sharpness, exposure estimation
│   ├── stacker.py           # Alignment + sigma-clipped stacking
│   └── stack_orchestrator.py# Background job queue
├── utils/
│   ├── logger_config.py     # Logging setup
│   ├── status_writer.py     # Status JSON writer
│   └── storage.py           # JSON append/URL helpers
├── tools/                   # Calib, simulator, cleanup, tests
├── coord_utils.py           # All astrometry / coordinate math
├── dashboard/
│   └── server.py + templates
└── logs/
    └── aircraft.json        # Hot file for ADS-B updates
```

## Dashboard API (read-only)
| Endpoint                     | Description                              |
|------------------------------|------------------------------------------|
| GET /api/status              | Full system state + top-5 candidates     |
| GET /api/latest_stack        | Most recent stacked image metadata       |
| GET /api/aircraft/{icao}     | Historical stacks for specific aircraft  |

## Manual Commands (via dashboard → logs/command.json)
| Command           | JSON payload                     | Effect                              |
|-------------------|----------------------------------|-------------------------------------|
| Track ICAO        | `{"track_icao": "A12345"}`       | Force-track specific aircraft       |
| Abort             | `{"command": "abort_track"}`     | Return to scanning                  |
| Park              | `{"command": "park_mount"}`      | Execute park routine                |

## Troubleshooting
| Symptom                        | Cause                                      | Fix                                            |
|-------------------------------|--------------------------------------------|------------------------------------------------|
| main.py exits immediately     | Invalid/missing aircraft.json              | Verify path and JSON updates                   |
| No mount movement             | Wrong device names or INDI not running     | Match config exactly to INDI device list       |
| Guiding wrong direction      | Incorrect az/el_offset_sign                | Flip ±1 values                                 |
| No stacks appear              | Stacker worker crashed                     | Check console for alignment/OpenCV errors      |
| pip-audit not found           | Missing dependency                         | `pip install pip-audit`                        |

## License
MIT – see [LICENSE](LICENSE)
