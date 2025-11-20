#!/bin/bash

pathImages="$PWD/logs/images"
pathStack="$PWD/logs/stack"
pathLogs="$PWD/logs"
pathData="$PWD/data"

find . -name '__pycache__' -type d -exec rm -rf {} +

rm -f "$pathImages"/{capture_,snap_}*.{png,fits}
rm -rf "$pathStack"/*
rm "$pathLogs"/gemini.log
rm "$pathLogs"/status.json
rm "$pathLogs"/captures.json
rm "$pathData"/aircraft.json

