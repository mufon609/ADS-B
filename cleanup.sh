#!/bin/bash

pathImages="$PWD/logs/images"
pathStack="$PWD/logs/stack"

rm -f "$pathImages"/{capture_,snap_}*.{png,fits}
find . -name '__pycache__' -type d -exec rm -rf {} +
rm -rf "$pathStack"/*
