#dashboard/server.py
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
import asyncio
import uvicorn

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import CONFIG, LOG_DIR

IMAGES_DIR = os.path.join(LOG_DIR, 'images')
STACK_ROOT = os.path.join(LOG_DIR, 'stack')
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(STACK_ROOT, exist_ok=True)

app = FastAPI()
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/logs", StaticFiles(directory=LOG_DIR), name="logs")

env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")),
    autoescape=select_autoescape(['html', 'xml'])
)

# --- Blocking I/O Functions (to be wrapped by asyncio.to_thread) ---

def read_status():
    """Reads the latest status from the status.json file. (Blocking I/O)"""
    path = os.path.join(LOG_DIR, 'status.json')
    try:
        # Ensure file exists before trying to open
        if not os.path.exists(path):
            return {"mode": "initializing", "error": "status_file_missing"}
        with open(path, 'r', encoding='utf-8') as f: # Specify encoding
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"Warning: Error reading status file '{path}': {e}")
        return {"mode": "error", "error_message": f"Could not read status: {e}"}
    except Exception as e: # Catch any other unexpected errors
        print(f"ERROR: Unexpected error reading status file '{path}': {e}")
        return {"mode": "error", "error_message": f"Unexpected error reading status: {e}"}

def _rel_to_logs_url(path: str) -> Optional[str]:
    """Map an absolute path under LOG_DIR to a /logs/... URL."""
    # This function doesn't perform I/O, safe to call directly
    try:
        abs_path = os.path.abspath(path)
        abs_root = os.path.abspath(LOG_DIR)
        # Robust check for path traversal attempts (though StaticFiles handles it too)
        if os.path.commonpath([abs_path, abs_root]) != abs_root:
            print(f"Warning: Attempted to access path outside LOG_DIR: {path}")
            return None
        # Use os.path.normpath to handle separators correctly
        rel = os.path.relpath(abs_path, abs_root)
        # Convert to URL path format (forward slashes)
        url_path = rel.replace(os.sep, "/")
        return f"/logs/{url_path}"
    except Exception as e:
        print(f"Warning: Error creating relative URL for path '{path}': {e}")
        return None

def _safe_read_json(path: str) -> Optional[dict]:
    """Safely reads a JSON file. (Blocking I/O)"""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding='utf-8') as f: # Specify encoding
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        print(f"Warning: Error reading JSON file '{path}': {e}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error reading JSON file '{path}': {e}")
        return None

def _sequence_mtime(seq_dir: str) -> float:
    """Get modification time of manifest or directory. (Blocking I/O)"""
    manifest = os.path.join(seq_dir, "manifest.json")
    path_to_check = manifest if os.path.exists(manifest) else seq_dir
    try:
        return os.path.getmtime(path_to_check)
    except Exception as e:
        # print(f"Warning: Could not get mtime for '{path_to_check}': {e}")
        return 0.0 # Return epoch start on error

def _find_sequence_dirs_for_icao(icao: str) -> List[str]:
    """Finds sequence directories for an ICAO. (Blocking I/O)"""
    out: List[str] = []
    icao_dir = os.path.join(STACK_ROOT, icao)
    try:
        if not os.path.isdir(icao_dir):
            return out
        # Use scandir for potentially better performance on large directories
        with os.scandir(icao_dir) as entries:
            for entry in entries:
                if entry.is_dir():
                    out.append(entry.path)
    except OSError as e:
        print(f"Error listing sequence directories for {icao}: {e}")
        return [] # Return empty list on error

    # Sort by modification time (newest first)
    try:
        out.sort(key=_sequence_mtime, reverse=True)
    except Exception as e:
        print(f"Warning: Error sorting sequence directories for {icao}: {e}")
        # Return unsorted list if sorting fails
    return out

def _find_latest_sequence_dir() -> Optional[Tuple[str, str, str]]:
    """Finds the overall newest sequence directory. (Blocking I/O)"""
    newest_seq_dir: Optional[str] = None
    newest_icao: str = ""
    newest_seq_id: str = ""
    newest_mt: float = 0.0

    try:
        if not os.path.isdir(STACK_ROOT):
            return None
        with os.scandir(STACK_ROOT) as icao_entries:
            for icao_entry in icao_entries:
                if not icao_entry.is_dir(): continue
                icao = icao_entry.name
                icao_dir = icao_entry.path
                try:
                    with os.scandir(icao_dir) as seq_entries:
                        for seq_entry in seq_entries:
                            if not seq_entry.is_dir(): continue
                            seq_dir = seq_entry.path
                            mt = _sequence_mtime(seq_dir)
                            if mt > newest_mt:
                                newest_mt = mt
                                newest_seq_dir = seq_dir
                                newest_icao = icao
                                newest_seq_id = seq_entry.name
                except OSError as e:
                    print(f"Warning: Error scanning directory '{icao_dir}': {e}")
                    continue # Skip problematic ICAO directory

    except OSError as e:
        print(f"Error scanning STACK_ROOT '{STACK_ROOT}': {e}")
        return None # Cannot proceed if root scan fails

    if newest_seq_dir:
        return newest_icao, newest_seq_id, newest_seq_dir
    return None

def _collect_products(seq_dir: str) -> Dict[str, Dict[str, Optional[str]]]:
    """Builds product dictionary with URLs. (Blocking I/O checks paths)"""
    variants = {
        "mean":    ("stack_mean.png",    "stack_mean.fits"),
        "robust":  ("stack_robust.png",  "stack_robust.fits"),
        "anomaly": ("stack_anomaly.png", "stack_anomaly.fits"),
    }
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for key, (png_name, fits_name) in variants.items():
        png_path = os.path.join(seq_dir, png_name)
        fits_path = os.path.join(seq_dir, fits_name)
        # Check existence synchronously, convert to URL (non-blocking)
        png_url = _rel_to_logs_url(png_path) if os.path.exists(png_path) else None
        fits_url = _rel_to_logs_url(fits_path) if os.path.exists(fits_path) else None
        out[key] = {"png": png_url, "fits": fits_url}
    return out

def write_command(command: dict):
    """Atomically writes a command to the command file. (Blocking I/O)"""
    COMMAND_FILE = os.path.join(LOG_DIR, CONFIG['logging']['command_file'])
    try:
        os.makedirs(os.path.dirname(COMMAND_FILE), exist_ok=True)
        # Use simpler temp file naming? tempfile is robust.
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(COMMAND_FILE), prefix=".cmd.", suffix=".json", text=True)
        try:
            with os.fdopen(fd, "w", encoding='utf-8') as f: # Specify encoding
                json.dump(command, f, indent=None) # Compact JSON for commands
                f.flush()
                os.fsync(f.fileno()) # Ensure data hits disk
            os.replace(tmp_path, COMMAND_FILE) # Atomic rename
            print(f"Dashboard command sent: {command}")
        finally:
            # Clean up temp file regardless of success/failure
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except OSError: pass # Ignore cleanup errors
    except Exception as e:
        print(f"Error writing command file '{COMMAND_FILE}': {e}")

# --- FastAPI Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main dashboard HTML page."""
    template = env.get_template("index.html")
    # Pass only necessary config parts to template if desired
    return template.render(request=request, config=CONFIG)

@app.get("/api/status")
async def api_get_status():
    """Serves the latest status.json content."""
    # Move blocking I/O to a thread, as done with other endpoints
    status_data = await asyncio.to_thread(read_status)
    
    status_code = 200
    if status_data.get("error") == "status_file_missing":
        status_code = 404
    elif status_data.get("mode") == "error" or status_data.get("error"):
        status_code = 500
    
    return JSONResponse(
        content=status_data,
        headers={"Cache-Control": "no-store"},
        status_code=status_code
    )

@app.post("/command/track", response_class=RedirectResponse)
async def command_track(icao: str = Form(...)):
    """Receives and validates a command to track a specific ICAO."""
    icao = (icao or "").strip().lower()
    if icao and len(icao) <= 6: # Basic validation
        await asyncio.to_thread(write_command, {"track_icao": icao})
    else:
        print(f"Warning: Invalid ICAO received in track command: '{icao}'")
    return RedirectResponse("/", status_code=303) # Redirect back to dashboard

@app.post("/command/abort", response_class=RedirectResponse)
async def command_abort():
    """Receives a command to abort the current track."""
    await asyncio.to_thread(write_command, {"command": "abort_track"})
    return RedirectResponse("/", status_code=303)

@app.post("/command/park", response_class=RedirectResponse)
async def command_park():
    """Receives a command to park the mount."""
    await asyncio.to_thread(write_command, {"command": "park_mount"})
    return RedirectResponse("/", status_code=303)

# -----------------------
# New read-only APIs for stacking results
# -----------------------

@app.get("/api/latest_stack")
async def api_latest_stack():
    """
    Returns the most recent stacked sequence with product URLs and manifest/QC.
    """
    latest = await asyncio.to_thread(_find_latest_sequence_dir)

    if not latest:
        return JSONResponse(
            content={"error": "no_stacks_yet"},
            headers={"Cache-Control": "no-store"},
            status_code=404,
        )
    icao, sequence_id, seq_dir = latest

    # Run remaining blocking I/O concurrently
    manifest_path = os.path.join(seq_dir, "manifest.json")
    manifest_task = asyncio.to_thread(_safe_read_json, manifest_path)
    products_task = asyncio.to_thread(_collect_products, seq_dir)
    mtime_task = asyncio.to_thread(_sequence_mtime, seq_dir) # Get mtime async too

    manifest, products, ts = await asyncio.gather(manifest_task, products_task, mtime_task)

    payload = {
        "icao": icao,
        "sequence_id": sequence_id,
        "timestamp": int(ts), # Use fetched mtime
        "products": products,
        "manifest": manifest, # Can be None if read failed
    }
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})

@app.get("/api/aircraft/{icao}/recent_stacks")
async def api_recent_stacks(icao: str, limit: int = 5):
    """
    List recent sequence directories for a given ICAO (newest first).
    """
    icao = (icao or "").strip().lower()
    limit = max(1, min(limit, 50)) # Clamp limit

    seq_dirs = await asyncio.to_thread(_find_sequence_dirs_for_icao, icao)

    # Concurrently fetch data for the limited sequence directories
    async def get_item_data(seq_dir):
        seq_id = os.path.basename(seq_dir)
        # Run _sequence_mtime and _collect_products concurrently
        ts_task = asyncio.to_thread(_sequence_mtime, seq_dir)
        products_task = asyncio.to_thread(_collect_products, seq_dir)
        ts, products = await asyncio.gather(ts_task, products_task)
        return {
            "sequence_id": seq_id,
            "timestamp": int(ts),
            "products": products,
        }

    tasks = [get_item_data(seq_dir) for seq_dir in seq_dirs[:limit]]
    items: List[Dict] = []
    if tasks:
        try:
            # Gather results, handle potential errors during concurrent fetching
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions, keep successful results
            items = [res for res in results if not isinstance(res, Exception)]
            # Log exceptions if any occurred
            exceptions = [res for res in results if isinstance(res, Exception)]
            if exceptions:
                print(f"Warning: Errors occurred fetching recent stacks for {icao}: {exceptions}")
        except Exception as e:
            print(f"Error gathering recent stack data for {icao}: {e}")
            # Return potentially empty list if gather fails

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
    """
    icao = (icao or "").strip().lower()
    # Basic validation of sequence_id format? Assuming it's just a name for now.
    sequence_id = (sequence_id or "").strip()

    seq_dir = os.path.join(STACK_ROOT, icao, sequence_id)
    # Check if directory exists synchronously first (fast check)
    if not os.path.isdir(seq_dir):
        return JSONResponse(
            content={"error": "not_found", "message": f"Sequence directory not found: {seq_dir}"},
            headers={"Cache-Control": "no-store"},
            status_code=404,
        )

    # Run potentially slow file reads concurrently
    manifest_path = os.path.join(seq_dir, "manifest.json")
    manifest_task = asyncio.to_thread(_safe_read_json, manifest_path)
    products_task = asyncio.to_thread(_collect_products, seq_dir)

    manifest, products = await asyncio.gather(manifest_task, products_task)

    payload = {
        "icao": icao,
        "sequence_id": sequence_id,
        "products": products,
        "manifest": manifest, # Can be None if manifest file missing/corrupt
    }
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})

if __name__ == "__main__":
    # Get host and port from config, with fallbacks
    host = CONFIG.get('dashboard', {}).get('host', '0.0.0.0')
    port = CONFIG.get('dashboard', {}).get('port', 5001)
    print(f"Starting dashboard server at http://{host}:{port}...")
    uvicorn.run(app, host=host, port=port)

