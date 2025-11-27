# dashboard/server.py
"""
FastAPI web server for the real-time dashboard.
This version implements a high-performance WebSocket push architecture, 
mtime-based caching, and background scanning to minimize file I/O and latency.
"""
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

from utils.logger_config import setup_logging
from config.loader import CONFIG, LOG_DIR

import uvicorn
from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from utils.storage import rel_to_logs_url
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Ensure the parent directory is in the path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

IMAGES_DIR = os.path.join(LOG_DIR, 'images')
STACK_ROOT = os.path.join(LOG_DIR, 'stack')
STATUS_FILE = os.path.join(LOG_DIR, 'status.json')
COMMAND_FILE = os.path.join(LOG_DIR, CONFIG.get('logging', {}).get('command_file', 'command.json'))

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(STACK_ROOT, exist_ok=True)

app = FastAPI()
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/logs", StaticFiles(directory=LOG_DIR), name="logs")

env = Environment(
    loader=FileSystemLoader(os.path.join(
        os.path.dirname(__file__), "templates")),
    autoescape=select_autoescape(['html', 'xml'])
)

# --- GLOBAL CACHE STRUCTURES ---
# _STATUS_CACHE: Stores the last read status and its file mtime
_STATUS_CACHE = {"data": {"mode": "initializing", "error": "cache_empty"}, "mtime": 0.0}
# _STACK_CACHE: Stores pre-scanned, expensive gallery data
_STACK_CACHE = {"latest": None, "recent_icao": {}, "timestamp": 0.0}

# --- WebSocket Connection Manager ---
class ConnectionManager:
    """Tracks active WebSocket clients and broadcasts status updates to them."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket client."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket client from the active set (ignoring missing entries)."""
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass

    async def broadcast(self, data: Dict):
        """
        Send a JSON-serializable payload to all connected clients.

        Exceptions during send are logged and the offending connection is pruned.
        """
        message = json.dumps(data)
        
        send_tasks = []
        # Create a copy of the list to iterate over safely
        connections_to_check = self.active_connections[:] 

        for connection in connections_to_check:
            send_tasks.append(connection.send_text(message))
        
        # Use return_exceptions=True to handle disconnections gracefully
        results = await asyncio.gather(*send_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"WebSocket send failed (client disconnect/error): {result}")
                try:
                    # Remove the failed connection
                    self.active_connections.remove(connections_to_check[i])
                except ValueError:
                    pass

manager = ConnectionManager()

# --- Helper Functions (I/O, Cache) ---

def _get_mtime_for_cache(url: str) -> Optional[int]:
    """Converts a /images/ URL back to a file path and returns its mtime."""
    if not url or not url.startswith("/images/"):
        return None
    
    # Extract filename from URL (stripping query params if any)
    filename = url.split('/')[-1].split('?')[0]
    file_path = os.path.join(IMAGES_DIR, filename)
    
    try:
        if os.path.exists(file_path):
            return int(os.path.getmtime(file_path))
        return None
    except Exception as e:
        logger.warning(f"Could not get mtime for guide image {file_path}: {e}")
        return None


def _read_status_from_disk() -> Tuple[Optional[Dict], Optional[float]]:
    """Reads status from disk and adds guide image mtime for cache busting."""
    try:
        current_mtime = os.path.getmtime(STATUS_FILE)
    except FileNotFoundError:
        return None, None
    except Exception:
        # Cannot stat file, treat as an unknown state
        return None, None
    
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
        
        # CRITICAL: Enhance payload with guide image mtime for client cache busting
        guide_url = status_data.get('last_guide_png', '')
        status_data['last_guide_mtime'] = _get_mtime_for_cache(guide_url)
        
        return status_data, current_mtime
    
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Warning: Error reading status file '{STATUS_FILE}' (JSON or OS error): {e}")
        return None, current_mtime


def read_status_cached() -> Dict:
    """
    Return the most recently cached status payload without hitting disk.

    The cache is refreshed by ``status_poller_task``; this accessor is safe to use
    inside request handlers to avoid filesystem I/O.
    """
    global _STATUS_CACHE
    return _STATUS_CACHE['data']


def _safe_read_json(path: str) -> Optional[dict]:
    """Safely reads a JSON file."""
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.warning(f"Warning: Error reading JSON file '{path}': {e}")
        return None
    except Exception as e:
        logger.error(f"ERROR: Unexpected error reading JSON file '{path}': {e}")
        return None


def _sequence_mtime(seq_dir: str) -> float:
    """Get modification time of manifest or directory."""
    manifest = os.path.join(seq_dir, "manifest.json")
    path_to_check = manifest if os.path.exists(manifest) else seq_dir
    try:
        return os.path.getmtime(path_to_check)
    except Exception:
        return 0.0


def _collect_products(seq_dir: str) -> Dict[str, Dict[str, Optional[str]]]:
    """Builds product dictionary with URLs for a single sequence directory."""
    variants = {
        "mean":    ("stack_mean.png",    "stack_mean.fits"),
        "robust":  ("stack_robust.png",  "stack_robust.fits"),
        "anomaly": ("stack_anomaly.png", "stack_anomaly.fits"),
    }
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for key, (png_name, fits_name) in variants.items():
        png_path = os.path.join(seq_dir, png_name)
        fits_path = os.path.join(seq_dir, fits_name)
        png_url = rel_to_logs_url(
            png_path) if os.path.exists(png_path) else None
        fits_url = rel_to_logs_url(
            fits_path) if os.path.exists(fits_path) else None
        out[key] = {"png": png_url, "fits": fits_url}
    return out


def _scan_stack_root() -> Dict:
    """
    Scans the STACK_ROOT to find the latest sequence and recent sequences per ICAO.
    (Expensive I/O operation).
    """
    latest_seq_data = None
    icao_stacks: Dict[str, List[Dict]] = {}
    newest_mt: float = 0.0
    
    try:
        if not os.path.isdir(STACK_ROOT):
            return {"latest": None, "recent_icao": {}}

        for icao_entry in os.scandir(STACK_ROOT):
            if not icao_entry.is_dir():
                continue
            icao = icao_entry.name
            
            icao_sequences: List[Tuple[float, str, str]] = []

            for seq_entry in os.scandir(icao_entry.path):
                if not seq_entry.is_dir():
                    continue
                seq_dir = seq_entry.path
                mt = _sequence_mtime(seq_dir)
                icao_sequences.append((mt, seq_entry.name, seq_dir))

            icao_sequences.sort(key=lambda x: x[0], reverse=True)
            
            processed_sequences = []
            # Cache the top 5 most recent stacks per ICAO
            for mt, seq_id, seq_dir in icao_sequences[:5]:
                products = _collect_products(seq_dir)
                processed_sequences.append({
                    "sequence_id": seq_id,
                    "timestamp": int(mt),
                    "products": products,
                })
                
                # Check for overall newest stack
                if mt > newest_mt:
                    newest_mt = mt
                    # Read manifest for the absolute latest stack
                    manifest = _safe_read_json(
                        os.path.join(seq_dir, "manifest.json"))
                    latest_seq_data = {
                        "icao": icao,
                        "sequence_id": seq_id,
                        "timestamp": int(mt),
                        "products": products,
                        "manifest": manifest,
                    }
            
            if processed_sequences:
                icao_stacks[icao] = processed_sequences

    except OSError as e:
        logger.error(f"Error scanning stack root: {e}")
        return {"latest": None, "recent_icao": {}}

    return {"latest": latest_seq_data, "recent_icao": icao_stacks}


def write_command(command: dict):
    """
    Atomically persist a command for the backend process to consume.

    Args:
        command: Dict payload (e.g., ``{"track_icao": "abcd12"}``) to be written to
            ``COMMAND_FILE``. File is fsync'd and replaced to avoid partial reads.
    """
    try:
        os.makedirs(os.path.dirname(COMMAND_FILE), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(
            COMMAND_FILE), prefix=".cmd.", suffix=".json", text=True)
        try:
            with os.fdopen(fd, "w", encoding='utf-8') as f:
                json.dump(command, f, indent=None)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, COMMAND_FILE)
            logger.info(f"Dashboard command sent: {command}")
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    except Exception as e:
        logger.error(f"Error writing command file '{COMMAND_FILE}': {e}")


# --- BACKGROUND TASKS ---

async def status_poller_task():
    """
    Background task that monitors status.json mtime and broadcasts changes
    to WebSockets, replacing client polling.
    """
    global _STATUS_CACHE
    while True:
        try:
            # Read from disk using a thread pool executor
            status_data, current_mtime = await asyncio.to_thread(_read_status_from_disk)

            if status_data is not None:
                # Check if the file itself is newer OR if the guide image changed (mtime differs)
                is_file_newer = current_mtime > _STATUS_CACHE['mtime']
                is_guide_newer = (
                    status_data.get('last_guide_mtime')
                    != _STATUS_CACHE['data'].get('last_guide_mtime')
                )

                if is_file_newer or is_guide_newer:
                    # Update cache
                    _STATUS_CACHE['data'] = status_data
                    _STATUS_CACHE['mtime'] = current_mtime
                    
                    # Broadcast the update instantly via WebSocket
                    await manager.broadcast(status_data)
                
            elif (
                _STATUS_CACHE['data'].get("error") != "status_file_missing"
                and current_mtime is None
            ):
                # Handle error state (file missing or stat error)
                logger.warning("Status file disappeared or could not be read; broadcasting error state.")
                error_state = {"mode": "error", "error": "status_file_missing"}
                _STATUS_CACHE['data'] = error_state
                _STATUS_CACHE['mtime'] = time.time() # Update mtime to prevent immediate re-broadcast
                await manager.broadcast(error_state)

        except Exception as e:
            logger.error(f"Error in status poller task: {e}")
        
        # Poll the filesystem every 0.5 seconds for low-latency updates
        await asyncio.sleep(0.5) 


async def cache_refresher_loop():
    """Runs the stack scanning in the background thread periodically."""
    global _STACK_CACHE
    while True:
        try:
            # Run the expensive I/O operation off the main thread
            new_stack_data = await asyncio.to_thread(_scan_stack_root)
            _STACK_CACHE.update(new_stack_data)
            _STACK_CACHE['timestamp'] = time.time()
        except Exception as e:
            logger.error(f"Error during stack cache refresh: {e}")
        
        # Refresh every 5 seconds for gallery/queue data freshness
        await asyncio.sleep(5) 


@app.on_event("startup")
async def startup_event():
    """Starts the background cache refresher and status poller tasks."""
    logger.info("Starting background cache refresher for gallery data.")
    asyncio.create_task(cache_refresher_loop())
    
    logger.info("Starting background status poller for WebSocket push.")
    asyncio.create_task(status_poller_task())


# --- FastAPI Endpoints ---

@app.get("/health")
async def health_check():
    """Returns status of the server and connectivity to critical data."""
    last_update_mtime = _STATUS_CACHE.get('mtime', 0)
    current_time = time.time()
    
    # If the cache hasn't been updated in 30 seconds, something is wrong with the poller/main app
    if (
        (current_time - last_update_mtime) > 30.0
        and last_update_mtime != 0
    ):
        raise HTTPException(status_code=503, detail="Status cache is stale (poller failure or main app frozen).")
    
    if _STATUS_CACHE['data'].get('error') == 'status_file_missing':
         raise HTTPException(status_code=503, detail="Status file is missing.")

    return {"status": "healthy", "last_status_mtime": last_update_mtime}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main dashboard HTML page."""
    template = env.get_template("index.html")
    return template.render(request=request, config=CONFIG)


@app.get("/api/status")
async def api_get_status():
    """Returns the latest status from cache. Used primarily for initial load/fallback."""
    status_data = read_status_cached()

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
    if icao and len(icao) <= 6:
        await asyncio.to_thread(write_command, {"track_icao": icao})
    else:
        logger.warning(
            f"Warning: Invalid ICAO received in track command: '{icao}'")
    return RedirectResponse("/", status_code=303)


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


@app.post("/command/idle", response_class=RedirectResponse)
async def command_idle():
    """Places the system into monitor/idle mode without slewing."""
    await asyncio.to_thread(write_command, {"command": "idle_monitor"})
    return RedirectResponse("/", status_code=303)


@app.post("/command/auto", response_class=RedirectResponse)
async def command_auto():
    """Re-enables automatic target selection/tracking."""
    await asyncio.to_thread(write_command, {"command": "auto_track"})
    return RedirectResponse("/", status_code=303)


# --- WebSocket Endpoint ---

@app.websocket("/ws/status")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections for real-time status push."""
    await manager.connect(websocket)
    try:
        # Push current status immediately upon connection for initialization
        await websocket.send_text(json.dumps(read_status_cached()))
        
        # Keep connection alive, waiting for client disconnect
        while True:
            # We must receive something to keep the connection open, even if it's a ping/pong
            await websocket.receive_text() 
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


# --- Stack Endpoints ---

@app.get("/api/latest_stack")
async def api_latest_stack():
    """Returns the most recent stacked sequence, read from cache."""
    global _STACK_CACHE
    latest_data = _STACK_CACHE['latest']

    if not latest_data:
        return JSONResponse(
            content={"error": "no_stacks_yet"},
            headers={"Cache-Control": "no-store"},
            status_code=404,
        )
    return JSONResponse(content=latest_data, headers={"Cache-Control": "no-store"})


@app.get("/api/aircraft/{icao}/recent_stacks")
async def api_recent_stacks(icao: str, limit: int = 5):
    """
    List recent sequence directories for a given ICAO (newest first), read from cache.

    Args:
        icao: ICAO identifier (case-insensitive).
        limit: Max number of recent sequences to return (default 5).

    Returns:
        JSON payload with ``items`` list and the sanitized ``icao``. Returns 404 with an
        informative message when no sequences are available.
    """
    global _STACK_CACHE
    # Sanitize icao
    icao_clean = os.path.normpath(
        os.path.basename((icao or "").strip().lower()))
    
    recent_stacks = _STACK_CACHE['recent_icao'].get(icao_clean, [])

    if not recent_stacks:
        return JSONResponse(
            content={
                "icao": icao_clean,
                "items": [],
                "message": "no_sequences_for_icao"
            },
            headers={"Cache-Control": "no-store"},
            status_code=404,
        )
    
    return JSONResponse(
        content={"icao": icao_clean, "items": recent_stacks[:limit]},
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/stack/sequence/{icao}/{sequence_id}")
async def api_sequence_manifest(icao: str, sequence_id: str):
    """
    Return manifest.json and products for a specific sequence.
    (Performs disk I/O on manifest and products).
    """
    # Sanitize icao and sequence_id
    icao_clean = os.path.normpath(
        os.path.basename((icao or "").strip().lower()))
    sequence_id_clean = os.path.normpath(
        os.path.basename((sequence_id or "").strip()))

    seq_dir = os.path.abspath(os.path.join(
        STACK_ROOT, icao_clean, sequence_id_clean))

    # Security check: path must be within STACK_ROOT
    is_path_safe = (
        os.path.commonpath([seq_dir, os.path.abspath(STACK_ROOT)])
        == os.path.abspath(STACK_ROOT)
    )
    if not is_path_safe:
        return JSONResponse(
            content={"error": "forbidden", "message": "Invalid path specified."},
            status_code=403,
        )

    if not os.path.isdir(seq_dir):
         return JSONResponse(
            content={"error": "not_found", "message": f"Sequence directory not found: {seq_dir}"},
            headers={"Cache-Control": "no-store"},
            status_code=404,
        )

    manifest_path = os.path.join(seq_dir, "manifest.json")
    
    # Run slow file reads concurrently
    manifest_task = asyncio.to_thread(_safe_read_json, manifest_path)
    products_task = asyncio.to_thread(_collect_products, seq_dir)

    manifest, products = await asyncio.gather(manifest_task, products_task)

    if not manifest:
        return JSONResponse(
            content={
                "error": "not_found",
                "message": f"Sequence manifest not found: {manifest_path}"
            },
            headers={"Cache-Control": "no-store"},
            status_code=404,
        )

    payload = {
        "icao": icao,
        "sequence_id": sequence_id,
        "products": products,
        "manifest": manifest,
    }
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


# --- Main execution block ---
if __name__ == "__main__":
    setup_logging()
    host = CONFIG.get('dashboard', {}).get('host', '0.0.0.0')
    port = CONFIG.get('dashboard', {}).get('port', 8000)
    logger.info(f"Starting dashboard server at http://{host}:{port}...")
    uvicorn.run(app, host=host, port=port)
