# SDG Pipeline

A synthetic data generation pipeline that runs in **Blender 4.2 LTS headless mode**: loads `.glb` product models, randomizes scenes, renders RGB images, extracts per-instance segmentation masks via the Object Index pass, and writes COCO JSON annotations. Designed for the KV Challenge (RevolutionUC 2026).

---

## Requirements

- **Blender 4.2 LTS** — [download](https://www.blender.org/download/lts/4-2/) (exact version; other versions untested)
- **Python 3.11** — bundled with Blender 4.2, no separate install needed
- **GPU:** NVIDIA with 6 GB+ VRAM recommended; CPU fallback supported (`render.device: CPU`)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk:** ~2 GB per 1,000 images at 1080×1080
- **OS:** Windows 10+, Ubuntu 20.04+, macOS 12+ (Apple Silicon supported)

---

## Quick Start

1. **Install Blender 4.2 LTS**
2. **Install dependencies** into Blender's bundled Python:
   ```bash
   # Edit BLENDER_PY path in the script first, then:
   ./scripts/install_deps.sh
   ```
   On Windows, run this in Git Bash or WSL. See the script for platform-specific path examples.
3. **Add your `.glb` model** to `assets/models/`
4. **Add background images** to `assets/backgrounds/`
5. **Edit `config.yaml`** — set `category_name`, `num_images`
6. **Run the pipeline:**
   ```bash
   # Linux / macOS
   ./run.sh --config config.yaml

   # Windows
   run.bat --config config.yaml
   ```
   Both wrappers respect a `BLENDER` env var if Blender isn't on your PATH:
   ```bash
   BLENDER="/path/to/blender-4.2/blender" ./run.sh --config config.yaml
   ```
   For a quick smoke test (1 image, no randomization):
   ```bash
   ./run.sh --debug --config config.yaml
   ```
7. **Output** in `output/images/`, `output/masks/`, `output/annotations/`

---

## Example Configs

| Config | Purpose |
|--------|---------|
| `examples/minimal_test.yaml` | 5 images, CPU, 8 samples — smoke test without a GPU |
| `examples/multi_category.yaml` | 200 images, 2 object categories, GPU |

---

## Generating 2,000 Images (Multi-GPU)

```bash
./scripts/render_parallel.sh 2000 4
```

Set `BLENDER` if needed:
```bash
BLENDER=/path/to/blender-4.2/blender ./scripts/render_parallel.sh 2000 4
```

---

## Inspecting Output

Visualize one rendered image with mask overlays and bounding boxes:
```bash
python scripts/visualize.py --output_dir output --idx 0
# Saves output/inspect_0000.png
```

---

## Output Format

```
output/
├── images/           NNNN_0001.png               RGB render
├── masks/            NNNN_semantic.png            Grayscale, pixel=category_id
├── masks_instance/   NNNN_inst_K_0001.png         Binary mask per object instance
├── labels/           NNNN_label.json              Per-image KV-format label
└── annotations/      instances.json               COCO format (merged)
```

---

## COCO JSON Schema

Standard COCO instance segmentation. `categories[].id` matches pixel values in semantic masks.

---

## Extending the Pipeline

The pipeline is built around four injectable classes. Override one method to change one stage — everything else runs unchanged.

| Class | Module | Controls |
|-------|--------|----------|
| `Randomizer` | `pipeline/randomizer.py` | camera, lights, object placement, materials |
| `Renderer` | `pipeline/renderer.py` | Cycles settings, compositor, render trigger |
| `AnnotationWriter` | `pipeline/annotation_writer.py` | label JSON format, finalize hook |
| `SDGPipeline` | `pipeline/pipeline.py` | the loop itself |

### Run with defaults (equivalent to `run.py`)

```python
from pipeline import SDGPipeline

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SDGPipeline(cfg, Path("output")).run(start=0, end=50)
```

### Fix the camera to overhead only

```python
from pipeline import SDGPipeline, Randomizer

class OverheadCam(Randomizer):
    def randomize_camera(self, cam_obj, rng, cfg):
        cam_obj.location = (0.0, 0.0, 3.0)
        cam_obj.rotation_euler = (0.0, 0.0, 0.0)

SDGPipeline(cfg, Path("output"), randomizer=OverheadCam()).run()
```

### Write COCO JSON instead of (or alongside) per-image KV labels

```python
from pipeline import SDGPipeline, AnnotationWriter
from pipeline.annotation_writer import init_coco, write_coco

class CocoWriter(AnnotationWriter):
    def __init__(self, cfg):
        self._coco = init_coco(cfg)

    def write_label(self, label, path):
        super().write_label(label, path)   # keep the KV JSON too
        # build COCO image + annotation entries from label and append to self._coco

    def finalize(self, output_dir):
        write_coco(self._coco, output_dir / "annotations" / "instances.json")

SDGPipeline(cfg, Path("output"), annotation_writer=CocoWriter(cfg)).run()
```

### Use a pre-existing pipeline instance from a script inside Blender

```python
# custom_run.py — invoke via: blender -b base_scene.blend -P custom_run.py -- --config config.yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from pipeline import SDGPipeline, Randomizer

class NoScaleRandomizer(Randomizer):
    """Never rescale target objects — useful when real-world size is calibrated."""
    def place_object(self, obj, rng, placed_aabbs, **kw):
        kw["randomize_scale"] = False
        super().place_object(obj, rng, placed_aabbs, **kw)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SDGPipeline(cfg, Path("output"), randomizer=NoScaleRandomizer()).run(start=0, end=200)
```

> **Camera is intentionally static.** The camera position is fixed in `base_scene.blend` and is never randomized during a run. `Randomizer.randomize_camera` exists for subclasses that need a moving camera but is not called by default.

---

## Object Arrangements (Grouped Bunching)

The pipeline can place multiple instances of the same object class in tight grid formations — 1×1, 2×2, 3×3, or any N×M — matching the style of real retail shelf imagery. Each object in a group still gets its own unique `pass_index`, so per-instance masks and COCO annotations work unchanged.

### How it works

1. **Activation** — `activate_frame_objects_with_groups` decides per class whether to form a group (rolling `group_probability`) and which grid size to use (weighted from `grid_sizes`).
2. **Parenting** — An invisible Blender Empty is created as a group pivot. All member objects are parented to it. Moving or rotating the pivot transforms the entire cluster.
3. **Flush layout** — `layout_group_local` places members in a centered N×M grid at center-to-center spacing derived from `obj.dimensions` (the mesh's actual bounding box × scale). `cell_spacing` adds optional extra clearance beyond flush contact.
4. **Placement** — `place_group_no_overlap` moves the pivot as a unit with AABB collision avoidance against all already-placed objects. If `rotate_as_unit: true` the whole group rotates together as a rigid cluster.
5. **Teardown** — After render, `destroy_group` un-parents all members, resets them to origin, and removes the pivot. The pool objects return to their hidden state for the next frame.

### Config reference

```yaml
scene:
  per_class_max: 9              # must be >= max(rows*cols) across all grid_sizes

  arrangements:
    enabled: true
    group_probability: 0.6      # 0.0 = always solo, 1.0 = always group
    grid_sizes:                 # [rows, cols, weight] — higher weight = sampled more often
      - [1, 1, 1]               # lone object (keeps solo objects in the mix)
      - [2, 2, 3]               # 2×2 pack — most common
      - [3, 3, 1]               # 3×3 pack — occasional
    cell_spacing: 0.0           # extra gap in metres beyond flush contact (0 = meshes touching)
    group_placement_retries: 15 # collision-avoidance retries for the whole group
    rotate_as_unit: true        # true = whole group rotates together; false = only position randomized
```

### Label JSON

Each object in a group gets a `group_id` integer field alongside the standard fields. Objects not in a group have no `group_id` key. All positions are world-space.

```json
{
  "name": "CocaCola.003",
  "position": [1.3256, 0.2162, 0.6720],
  "rotation": [12.4, -3.1, 87.2],
  "mask_color": [255, 128, 0],
  "group_id": 1
}
```

### Customization

#### 1. Config only — tune sizes and frequency

Add a 4×4 size, force CocaCola-like densities, or disable grouping entirely just by editing `config.yaml`. No code changes needed.

#### 2. Override group placement — subclass `Randomizer`

`Randomizer.place_group` is the designed override point. Example: always orient groups to face the camera.

```python
from pipeline.randomizer import Randomizer

class FacingGroupRandomizer(Randomizer):
    def place_group(self, group, rng, placed_aabbs, **kwargs):
        super().place_group(group, rng, placed_aabbs, **kwargs)
        cam_dir = -group.pivot.location.normalized()
        group.pivot.rotation_euler = cam_dir.to_track_quat("X", "Z").to_euler()

SDGPipeline(cfg, Path("output"), randomizer=FacingGroupRandomizer()).run()
```

#### 3. Custom grid geometry — replace `layout_group_local`

The function is pure math. Monkey-patch it for non-rectangular layouts.

```python
import mathutils
from pipeline import scene_builder

def hexagonal_layout(members, rows, cols, spacing_x, spacing_y=None):
    """Honeycomb / offset-row packing."""
    if spacing_y is None:
        spacing_y = spacing_x
    row_h = spacing_y * (3 ** 0.5 / 2)
    for idx, obj in enumerate(members):
        r, c = divmod(idx, cols)
        x = (c - (cols - 1) / 2) * spacing_x + (r % 2) * spacing_x / 2
        y = (r - (rows - 1) / 2) * row_h
        obj.matrix_local = mathutils.Matrix.Translation((x, y, 0.0))

scene_builder.layout_group_local = hexagonal_layout
```

#### 4. Per-class rules — extend `activate_frame_objects_with_groups`

The stem loop in `activate_frame_objects_with_groups` (`pipeline/scene_builder.py`) reads `arr_cfg` once at the top. Add a `per_class` block to your config and merge it over the global defaults inside the loop:

```yaml
arrangements:
  enabled: true
  group_probability: 0.6
  grid_sizes: [[2, 2, 1]]
  per_class:
    CocaCola:
      group_probability: 1.0   # always bunch CocaCola
      grid_sizes: [[3, 3, 1]]
    hero_product:
      group_probability: 0.0   # never bunch hero product
```

Then in the stem loop (line ~`for stem, objs in target_pool.items():`), merge the per-class overrides:

```python
stem_cfg = {**arr_cfg, **arr_cfg.get("per_class", {}).get(stem, {})}
group_prob = float(stem_cfg.get("group_probability", 0.5))
raw_grids  = stem_cfg.get("grid_sizes", [[1, 1, 1]])
```

#### 5. Pool sizing — avoid silent grid downgrades

If `per_class_max < rows * cols` for the largest grid size, the code silently shrinks the grid. To prevent this, set `per_class_max` to at least the largest grid cell count:

```
per_class_max >= max(rows * cols for all grid_sizes)
```

For `grid_sizes` containing a 3×3, set `per_class_max: 9` or higher.

---

## Adding a New Object Class

In `config.yaml`, add to `assets.models`:
```yaml
assets:
  models:
    - path: assets/models/new_class/
      glob: "*.glb"
      category_id: 2
      category_name: new_class
```

---

## Adding Distractors

Drop `.glb` files into `assets/distractors/`.
Set `use_primitives: true` for random Blender primitives.
Distractor count controlled by `scene.distractors_min/max`.

---

## Tuning Render Speed

- Decrease `render.samples` (20 is the baseline; 32 with denoiser is also fine)
- Set `render.device: GPU`
- Set `render.denoiser: OPTIX` for NVIDIA RTX cards

---

## Regenerating base_scene.blend

`base_scene.blend` is derived from the KV example datagen scene. To regenerate:
```bash
# Using the conda dev environment:
conda run -n sdg-pipeline python scripts/setup_base_scene.py
```
This strips product objects from `KV example datagen.blend`, enables the Object Index pass, and saves as `base_scene.blend`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Black images | Check `render.device` matches your hardware |
| Missing masks | Check `assets/backgrounds/` has at least one image |
| COCO has 0 annotations | Object may be outside camera view — tighten `camera.distance_min/max` or increase `objects_per_scene` |
| Mask PNGs missing | Verify `view_layer.use_pass_object_index` is True and `obj.pass_index` is set (done automatically by `assign_instance_ids`) |
| `ConfigError:` at startup | Check the named field in `config.yaml` — the error message says exactly what's wrong |
