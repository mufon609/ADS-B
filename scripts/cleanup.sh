#!/bin/bash

# Resolve repo root (one level up from this script)
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pathImages="$repo_root/logs/images"
pathStack="$repo_root/logs/stack"
pathLogs="$repo_root/logs"
pathAircraft="$pathLogs/aircraft.json"

find "$repo_root" -name '__pycache__' -type d -exec rm -rf {} +

rm -f "$pathImages"/{capture_,snap_}*.{png,fits}
rm -rf "$pathStack"/*
rm "$pathLogs"/gemini.log
rm "$pathLogs"/status.json
rm "$pathLogs"/captures.json
rm "$pathAircraft"
