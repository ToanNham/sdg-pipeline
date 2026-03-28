#!/bin/bash
# Usage: ./scripts/render_parallel.sh 2000 4
# Args: total_images num_gpus

TOTAL=${1:-2000}
GPUS=${2:-1}
BLENDER=/path/to/blender
CONFIG=config.yaml
CHUNK=$((TOTAL / GPUS))

for i in $(seq 0 $((GPUS - 1))); do
    START=$((i * CHUNK))
    END=$(((i + 1) * CHUNK))
    if [ $i -eq $((GPUS - 1)) ]; then END=$TOTAL; fi
    echo "GPU $i: images $START to $END"
    CUDA_VISIBLE_DEVICES=$i $BLENDER -b base_scene.blend \
        -P run.py -- \
        --config $CONFIG \
        --start $START \
        --end $END \
        > output/log_gpu_$i.txt 2>&1 &
done

wait
echo "All jobs done. Merging COCO annotations..."
python merge_coco.py output/annotations --out instances.json
python validate_output.py --output_dir output --config $CONFIG
