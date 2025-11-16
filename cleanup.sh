#!/bin/bash

pathImages="$PWD/logs/images"
pathStack="$PWD/logs/stack"
pathLogs="$PWD/logs"

rm -f "$pathImages"/{capture_,snap_}*.{png,fits}
find . -name '__pycache__' -type d -exec rm -rf {} +
rm -rf "$pathStack"/*
rm "$pathLogs"/gemini.log
