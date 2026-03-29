#!/usr/bin/env bash
# Install pipeline dependencies into Blender's bundled Python.
#
# Edit BLENDER_PY to match your Blender 4.2 installation.
# Blender 4.2 ships Python 3.11. Common paths:
#
#   Windows (Git Bash / WSL):
#     BLENDER_PY="C:/Program Files/Blender Foundation/Blender 4.2/4.2/python/bin/python.exe"
#
#   Linux:
#     BLENDER_PY="/path/to/blender-4.2.0-linux-x64/4.2/python/bin/python3.11"
#
#   macOS:
#     BLENDER_PY="/Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11"

BLENDER_PY=/path/to/blender-4.2/4.2/python/bin/python3.11

if [ ! -f "$BLENDER_PY" ]; then
    echo "ERROR: BLENDER_PY not found: $BLENDER_PY"
    echo "Edit this script to set the correct path to Blender 4.2's Python 3.11."
    exit 1
fi

echo "Using: $BLENDER_PY"

"$BLENDER_PY" -m pip install \
  PyYAML \
  numpy \
  Pillow \
  scikit-image \
  shapely \
  pycocotools

echo ""
echo "Verifying install..."
"$BLENDER_PY" -c "import yaml, numpy, PIL; print('OK — all dependencies installed successfully')" \
  || echo "FAILED — one or more imports failed, check the output above"
