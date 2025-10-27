"""
FastAPI web server for the real-time dashboard.
"""
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import json
import tempfile
import time
from typing import Optional, Dict, List, Tuple

from config_loader import CONFIG, LOG_DIR

IMAGES_DIR = os.path.join(LOG_DIR, 'images')
STACK_ROOT = os.path.join(LOG_DIR, 'stack')
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI()
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/logs", StaticFiles(directory=LOG_DIR), name="logs")

env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")),
    autoescape=select_autoescape(['html', 'xml'])
)

def read_status():
    """Reads the latest status from the status.json file."""
    path = os.path.join(LOG_DIR, 'status.json')
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {"mode": "initializing"}

def _rel_to_logs_url(path: str) -> Optional[str]:
    """Map an absolute path under LOG_DIR to a /logs/... URL."""
    try:
        abs_path = os.path.abspath(path)
        abs_root = os.path.abspath(LOG_DIR)
        if os.path.commonpath([abs_path, abs_root]) != abs_root:
            return None
        rel = os.path.relpath(abs_path, abs_root).replace(os.sep, "/")
        return f"/logs/{rel}"
    except Exception:
        return None

def _safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _sequence_mtime(seq_dir: str) -> float:
    """Get a reasonable mtime for a sequence directory (manifest.json if present)."""
    manifest = os.path.join(seq_dir, "manifest.json")
    if os.path.exists(manifest):
        try:
            return os.path.getmtime(manifest)
        except Exception:
            pass
    try:
        return os.path.getmtime(seq_dir)
    except Exception:
        return 0.0

def _find_sequence_dirs_for_icao(icao: str) -> List[str]:
    """Return sequence directories for an ICAO, newest first."""
    out: List[str] = []
    icao_dir = os.path.join(STACK_ROOT, icao)
    if not os.path.isdir(icao_dir):
        return out
    for name in os.listdir(icao_dir):
        seq_dir = os.path.join(icao_dir, name)
        if os.path.isdir(seq_dir):
            out.append(seq_dir)
    out.sort(key=_sequence_mtime, reverse=True)
    return out

def _find_latest_sequence_dir() -> Optional[Tuple[str, str, str]]:
    """
    Scan logs/stack/<ICAO>/<sequence_id>/ and pick the newest sequence.
    Returns (icao, sequence_id, seq_dir) or None.
    """
    if not os.path.isdir(STACK_ROOT):
        return None
    newest: Tuple[str, str, str, float] = ("", "", "", 0.0)
    for icao in os.listdir(STACK_ROOT):
        icao_dir = os.path.join(STACK_ROOT, icao)
        if not os.path.isdir(icao_dir):
            continue
        for seq_id in os.listdir(icao_dir):
            seq_dir = os.path.join(icao_dir, seq_id)
            if not os.path.isdir(seq_dir):
                continue
            mt = _sequence_mtime(seq_dir)
            if mt > newest[3]:
                newest = (icao, seq_id, seq_dir, mt)
    if newest[2]:
        return newest[0], newest[1], newest[2]
    return None

def _collect_products(seq_dir: str) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Build a {variant: {png, fits}} dict for the expected variants:
    mean, robust, anomaly.
    Values are returned as /logs/... URLs (or None if missing).
    """
    variants = {
        "mean":    ("stack_mean.png",    "stack_mean.fits"),
        "robust":  ("stack_robust.png",  "stack_robust.fits"),
        "anomaly": ("stack_anomaly.png", "stack_anomaly.fits"),
    }
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for key, (png_name, fits_name) in variants.items():
        png_path = os.path.join(seq_dir, png_name)
        fits_path = os.path.join(seq_dir, fits_name)
        out[key] = {
            "png": _rel_to_logs_url(png_path) if os.path.exists(png_path) else None,
            "fits": _rel_to_logs_url(fits_path) if os.path.exists(fits_path) else None,
        }
    return out

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main dashboard HTML page."""
    template = env.get_template("index.html")
    return template.render(request=request, config=CONFIG)

@app.get("/api/status")
async def api_status():
    """Provides the latest status as a JSON API endpoint, ensuring no caching."""
    data = read_status()
    return JSONResponse(content=data, headers={"Cache-Control": "no-store"})

def write_command(command: dict):
    """Atomically writes a command to the command file using a temp file and rename."""
    COMMAND_FILE = os.path.join(LOG_DIR, CONFIG['logging']['command_file'])
    try:
        os.makedirs(os.path.dirname(COMMAND_FILE), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(COMMAND_FILE), prefix=".cmd.", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(command, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, COMMAND_FILE)
            print(f"Dashboard command sent: {command}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except Exception as e:
        print(f"Error writing command file: {e}")

@app.post("/command/track", response_class=RedirectResponse)
async def command_track(icao: str = Form(...)):
    """Receives and validates a command to track a specific ICAO."""
    icao = (icao or "").strip().lower()
    if icao:
        write_command({"track_icao": icao})
    return RedirectResponse("/", status_code=303)

@app.post("/command/abort", response_class=RedirectResponse)
async def command_abort():
    """Receives a command to abort the current track."""
    write_command({"command": "abort_track"})
    return RedirectResponse("/", status_code=303)

@app.post("/command/park", response_class=RedirectResponse)
async def command_park():
    """Receives a command to park the mount."""
    write_command({"command": "park_mount"})
    return RedirectResponse("/", status_code=303)

# -----------------------
# New read-only APIs to support multi-variant stacking in the dashboard
# -----------------------

@app.get("/api/latest_stack")
async def api_latest_stack():
    """
    Returns the most recent stacked sequence with product URLs and manifest/QC if available.
    {
      "icao": "...",
      "sequence_id": "...",
      "timestamp": 1712345678,
      "products": {
        "mean":    {"png": "/logs/...", "fits": "/logs/..."},
        "robust":  {"png": "/logs/...", "fits": "/logs/..."},
        "anomaly": {"png": "/logs/...", "fits": "/logs/..."}
      },
      "manifest": {...}  # if present
    }
    """
    latest = _find_latest_sequence_dir()
    if not latest:
        return JSONResponse(
            content={"error": "no_stacks_yet"},
            headers={"Cache-Control": "no-store"},
            status_code=404,
        )
    icao, sequence_id, seq_dir = latest
    manifest = _safe_read_json(os.path.join(seq_dir, "manifest.json"))
    ts = int(time.time())
    try:
        ts = int(os.path.getmtime(seq_dir))
    except Exception:
        pass

    payload = {
        "icao": icao,
        "sequence_id": sequence_id,
        "timestamp": ts,
        "products": _collect_products(seq_dir),
        "manifest": manifest,
    }
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})

@app.get("/api/aircraft/{icao}/recent_stacks")
async def api_recent_stacks(icao: str, limit: int = 5):
    """
    List recent sequence directories for a given ICAO (newest first).
    Each item includes sequence_id, timestamp, and product URLs.
    """
    icao = (icao or "").strip().lower()
    seq_dirs = _find_sequence_dirs_for_icao(icao)
    items: List[Dict] = []
    for seq_dir in seq_dirs[: max(1, min(limit, 50))]:
        seq_id = os.path.basename(seq_dir)
        ts = int(_sequence_mtime(seq_dir))
        items.append({
            "sequence_id": seq_id,
            "timestamp": ts,
            "products": _collect_products(seq_dir),
        })
    if not items:
        return JSONResponse(
            content={"icao": icao, "items": [], "message": "no_sequences_for_icao"},
            headers={"Cache-Control": "no-store"},
            status_code=404,
        )
    return JSONResponse(content={"icao": icao, "items": items}, headers={"Cache-Control": "no-store"})

@app.get("/api/stack/sequence/{icao}/{sequence_id}")
async def api_sequence_manifest(icao: str, sequence_id: str):
    """
    Return manifest.json for a specific sequence if available.
    Also include product URLs for convenience.
    """
    icao = (icao or "").strip().lower()
    seq_dir = os.path.join(STACK_ROOT, icao, sequence_id)
    if not os.path.isdir(seq_dir):
        return JSONResponse(
            content={"error": "not_found"},
            headers={"Cache-Control": "no-store"},
            status_code=404,
        )
    manifest = _safe_read_json(os.path.join(seq_dir, "manifest.json"))
    payload = {
        "icao": icao,
        "sequence_id": sequence_id,
        "products": _collect_products(seq_dir),
        "manifest": manifest,
    }
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})
