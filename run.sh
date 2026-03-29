#!/usr/bin/env bash
# Run the SDG pipeline via Blender headless mode.
#
# Usage: ./run.sh [--config config.yaml] [--start 0] [--end 50] [--debug]
#
# Set BLENDER env var if blender is not on your PATH:
#   BLENDER="/path/to/blender-4.2/blender" ./run.sh --config config.yaml

BLENDER=${BLENDER:-blender}

if ! command -v "$BLENDER" &>/dev/null && [ ! -f "$BLENDER" ]; then
    echo "ERROR: blender not found: '$BLENDER'"
    echo "Set the BLENDER env var to the full path, e.g.:"
    echo "  BLENDER=/path/to/blender-4.2/blender $0 $*"
    exit 1
fi

exec "$BLENDER" -b base_scene.blend -P run.py -- "$@"
