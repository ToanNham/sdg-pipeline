#!/usr/bin/env bash
# Install pipeline dependencies into Blender's bundled Python.
# Edit BLENDER_PY to match your Blender installation path.

BLENDER_PY=/path/to/blender/3.6/python/bin/python3.10

$BLENDER_PY -m pip install \
  PyYAML \
  numpy \
  Pillow \
  scikit-image \
  shapely \
  pycocotools \
  OpenEXR
