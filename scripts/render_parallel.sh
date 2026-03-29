#!/bin/bash
# Render images in parallel across multiple GPUs.
#
# Usage: ./scripts/render_parallel.sh <total_images> <num_gpus>
#   Example: ./scripts/render_parallel.sh 2000 4
#
# Set BLENDER env var if blender is not on your PATH:
#   BLENDER="/path/to/blender-4.2/blender" ./scripts/render_parallel.sh 2000 4
#
# Set CONFIG env var to use a different config file (default: config.yaml):
#   CONFIG=examples/multi_category.yaml ./scripts/render_parallel.sh 200 2

TOTAL=${1:-2000}
GPUS=${2:-1}
BLENDER=${BLENDER:-blender}
CONFIG=${CONFIG:-config.yaml}
CHUNK=$((TOTAL / GPUS))

if ! command -v "$BLENDER" &>/dev/null && [ ! -f "$BLENDER" ]; then
    echo "ERROR: blender not found: '$BLENDER'"
    echo "Set the BLENDER env var to the full path, e.g.:"
    echo "  BLENDER=/path/to/blender-4.2/blender $0 $*"
    exit 1
fi

mkdir -p output

for i in $(seq 0 $((GPUS - 1))); do
    START=$((i * CHUNK))
    END=$(((i + 1) * CHUNK))
    if [ $i -eq $((GPUS - 1)) ]; then END=$TOTAL; fi
    echo "GPU $i: images $START to $END"
    CUDA_VISIBLE_DEVICES=$i "$BLENDER" -b base_scene.blend \
        -P run.py -- \
        --config $CONFIG \
        --start $START \
        --end $END \
        > output/log_gpu_$i.txt 2>&1 &
done

wait
echo "All jobs done. Merging COCO annotations..."
python scripts/merge_coco.py output/annotations --out instances.json
python scripts/validate_output.py --output_dir output --config $CONFIG
