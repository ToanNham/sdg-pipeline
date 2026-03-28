# SDG Pipeline

## Quick Start
1. Install Blender 4.2 LTS
2. Run scripts/install_deps.sh (edit BLENDER_PY path first)
3. Put your .glb model in assets/models/
4. Put background images in assets/backgrounds/
5. Edit config.yaml: set category_name, num_images
6. Run: blender -b base_scene.blend -P run.py -- --config config.yaml
7. Output in output/images/, output/masks/, output/annotations/

## Regenerating base_scene.blend
`base_scene.blend` is derived from the KV example datagen scene. To regenerate it:
```bash
C:/Users/nhamd/miniconda3/envs/sdg-pipeline/python.exe scripts/setup_base_scene.py
```
This strips the product objects from `KV example datagen.blend`, enables the Object Index pass, and saves as `base_scene.blend`.

## Generating 2,000 Images (Multi-GPU)
```bash
./scripts/render_parallel.sh 2000 4
```

## Output Format
- output/images/NNNN_0001.png                  RGB render
- output/masks/NNNN_semantic.png               Grayscale, pixel=category_id
- output/masks_instance/NNNN_inst_K_0001.png   Binary mask per object instance
- output/annotations/instances.json            COCO format

## COCO JSON Schema
Standard COCO instance segmentation.
categories[].id matches pixel values in semantic masks.

## Adding a New Object Class
In config.yaml, add to assets.models:
```yaml
  - path: assets/models/new_class/
    glob: "*.glb"
    category_id: 2
    category_name: new_class
```

## Adding Distractors
Drop .glb files into assets/distractors/.
Set use_primitives: true for random Blender primitives.
Distractor count controlled by scene.distractors_min/max.

## Tuning Render Speed
- Decrease render.samples (20 is the KV baseline; 32 with denoiser is also fine)
- Set render.device: GPU
- Set render.denoiser: OPTIX for NVIDIA cards

## Troubleshooting
Black images: check render.device matches your hardware
Missing masks: check assets/backgrounds/ has at least one image
COCO has 0 annotations: object may be outside camera view —
  tighten camera distance or increase objects_per_scene
Mask PNGs missing: verify view_layer.use_pass_object_index is True and
  obj.pass_index is set (assign_instance_ids does this automatically)
