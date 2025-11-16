#!/usr/bin/env python3
"""
Fake dump1090 (1 Hz, aligned to whole seconds)

- Writes a realistic aircraft.json once per second (atomic writes).
- Uses accurate dt between ticks for kinematics.
- Ensures output directory exists.
- Includes a "no position" craft to exercise estimator rejections.

Usage:
  python fake_dump1090_1hz.py --out /home/dump/Desktop/gitRepo/dump1090/public_html/data/aircraft.json
"""

import os, time, math, json, random, argparse, logging
from logger_config import setup_logging

logger = logging.getLogger(__name__)

# ---------- helpers ----------
def gc_step(lat_deg, lon_deg, heading_deg, distance_m):
    """Great-circle step from (lat,lon) by distance_m at heading_deg.
       Handles negative distances by flipping heading."""
    if distance_m < 0:
        return gc_step(lat_deg, lon_deg, heading_deg + 180.0, -distance_m)
    R = 6371000.0
    if distance_m == 0:
        return lat_deg, lon_deg
    lat1 = math.radians(lat_deg); lon1 = math.radians(lon_deg)
    brng = math.radians(heading_deg % 360.0)
    ang = distance_m / R
    sin_lat2 = math.sin(lat1)*math.cos(ang) + math.cos(lat1)*math.sin(ang)*math.cos(brng)
    lat2 = math.asin(max(-1.0, min(1.0, sin_lat2)))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(ang)*math.cos(lat1),
                             math.cos(ang) - math.sin(lat1)*math.sin(lat2))
    lon2 = ((math.degrees(lon2) + 540.0) % 360.0) - 180.0
    return math.degrees(lat2), lon2

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x

def atomic_write(path, payload_str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(payload_str)
    os.replace(tmp, path)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Fake dump1090 (1 Hz) aircraft.json generator")
    ap.add_argument("--out", default="data/aircraft.json",
                    help="Output aircraft.json path")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed (for reproducible motion)")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = args.out
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Seed a few aircraft from your snippet (with plausible motion)
    AC = [
        # a81761: add speed/track so it moves
        {"hex":"a81761","lat":40.377596,"lon":-73.589672,
         "alt_ft":31300, "has_geom": False, "gs_knots": 460.0, "track_deg": 222.0,
         "track_rate_dps": 0.0, "rate_fpm": -200.0},

        # a29982: as in snippet, slow mover
        {"hex":"a29982","lat":40.301715,"lon":-74.891676,
         "alt_ft":2950, "has_geom": True, "gs_knots": 73.8, "track_deg": 327.2,
         "track_rate_dps": +0.2, "rate_fpm": +800.0},

        # ad444d (JBU772): cruiser
        {"hex":"ad444d","lat":39.694931,"lon":-75.027366,
         "alt_ft":31650, "has_geom": True, "gs_knots": 500.0, "track_deg": 45.7,
         "track_rate_dps": 0.0, "rate_fpm": 0.0},

        # aa55bd: no position (exercises estimator rejections)
        {"hex":"aa55bd","lat": None,"lon": None,
         "alt_ft":32000, "has_geom": False, "gs_knots": 0.0, "track_deg": 0.0,
         "track_rate_dps": 0.0, "rate_fpm": 0.0},
    ]

    # Per-aircraft static-ish metadata
    for a in AC:
        a["messages"] = random.randint(5,150)
        a["sil_type"] = "perhour" if random.random() < 0.5 else "unknown"
        a["nac_p"] = random.choice([8,10])
        a["nac_v"] = random.choice([1,2])
        a["nic"]   = 8
        a["version"] = random.choice([0,2])
        a["rssi"] = round(random.uniform(-31.0, -23.0), 1)

    total_messages = 8_089_782  # start near your snippet
    logger.info(f"[fake_dump1090_1hz] Writing to {out_path} at 1 Hz. Ctrl+C to stop.")

    # Align to the next whole second
    next_ts = math.floor(time.time()) + 1
    prev_now = next_ts - 1  # model-time of previous tick

    try:
        while True:
            # ---- wait until the exact second ----
            sleep_for = next_ts - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

            now_unix = next_ts
            dt = now_unix - prev_now  # usually 1.0; resilient to hiccups
            prev_now = now_unix

            # ---- advance "true" state by dt seconds ----
            for a in AC:
                if a["lat"] is None or a["lon"] is None:
                    # keep 'alive'
                    a["messages"] += random.randint(0, 2)
                    a["rssi"] = round(clamp(a["rssi"] + random.uniform(-0.2, 0.2), -35.0, -22.0), 1)
                    continue

                # Heading update and motion
                a["track_deg"] = (a["track_deg"] + a["track_rate_dps"] * dt) % 360.0
                gs_ms = max(0.0, float(a["gs_knots"])) * 0.514444
                if gs_ms > 0.0:
                    a["lat"], a["lon"] = gc_step(a["lat"], a["lon"], a["track_deg"], gs_ms * dt)

                # Climb/descend
                a["alt_ft"] = clamp(a["alt_ft"] + float(a["rate_fpm"]) * dt / 60.0, 800.0, 41000.0)

                # Noise in RSSI/messages
                a["messages"] += random.randint(0, 3)
                a["rssi"] = round(clamp(a["rssi"] + random.uniform(-0.2, 0.2), -35.0, -22.0), 1)

            total_messages += random.randint(2, 25)

            # ---- compose dump1090-style aircraft list ----
            aircraft_list = []
            for a in AC:
                base = {
                    "hex": a["hex"],
                    "mlat": [], "tisb": [],
                    "messages": a["messages"],
                    "version": a["version"],
                    "nac_p": a["nac_p"], "nac_v": a["nac_v"],
                    "nic": a["nic"],
                    "sil": random.choice([2,3]),
                    "sil_type": a["sil_type"],
                    "rssi": a["rssi"],
                }
                base["alt_baro"] = int(round(a["alt_ft"]))
                if a["has_geom"]:
                    base["alt_geom"] = int(round(a["alt_ft"] + random.uniform(-250, 250)))

                # Optional callsign/category to mirror snippet
                if a["hex"] in ("ad444d", "ad7ab8"):
                    base["flight"] = "JBU772  " if a["hex"]=="ad444d" else "JBU603  "
                    base["category"] = "A3"

                if a["lat"] is not None and a["lon"] is not None:
                    base["gs"] = round(float(a["gs_knots"]), 1)
                    base["track"] = round(float(a["track_deg"]), 1)
                    # baro/geom rate fields (randomized like real dumps)
                    if random.random() < 0.5:
                        base["baro_rate"] = int(round(a["rate_fpm"]))
                    else:
                        base["geom_rate"] = int(round(a["rate_fpm"]))
                    # sometimes publish track_rate
                    if abs(a["track_rate_dps"]) > 0.01 and random.random() < 0.8:
                        base["track_rate"] = round(float(a["track_rate_dps"]), 3)

                    # Reported position age and back-propagation
                    age = random.uniform(0.0, 0.6)  # seen_pos
                    gs_ms = max(0.0, float(a["gs_knots"])) * 0.514444
                    heading_then = (a["track_deg"] - a["track_rate_dps"] * age) % 360.0
                    # Move back along heading at that time (negative distance handled by gc_step)
                    lat_rep, lon_rep = gc_step(a["lat"], a["lon"], heading_then, -gs_ms * age)
                    base["lat"] = round(lat_rep, 6)
                    base["lon"] = round(lon_rep, 6)
                    base["seen_pos"] = round(age, 1)
                    base["seen"] = round(max(0.0, age + random.uniform(-0.2, 0.4)), 1)
                    base["nic_baro"] = 1
                    base["rc"] = 186
                else:
                    # no-position craft
                    base["seen"] = round(random.uniform(0.8, 3.0), 1)

                aircraft_list.append(base)

            doc = {
                "now": int(now_unix),
                "messages": total_messages,
                "aircraft": aircraft_list,
            }

            # ---- atomic write ----
            atomic_write(out_path, json.dumps(doc, separators=(",", ":")))

            # Next tick
            next_ts += 1

    except KeyboardInterrupt:
        logger.info("\n[fake_dump1090_1hz] Stopped.")

if __name__ == "__main__":
    setup_logging()
    main()
