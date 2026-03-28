# MASTER DESIGN DOCUMENT
## Synthetic Data Generation Pipeline — Blender + bpy
### Hackathon Edition | 24-Hour Sprint

---

## PROJECT SNAPSHOT

**Goal:** A Python script that runs inside Blender headless mode, loads your 3D model, randomizes a scene, renders RGB images, extracts segmentation masks, and writes paired COCO JSON annotations. Target: 50 images for demo, architected to scale to 2,000.

**Stack:** Blender 4.2 LTS, bpy, Cycles renderer, NumPy, Pillow, scikit-image, pycocotools, PyYAML.

**Launch command:**
```
blender -b base_scene.blend -P run.py -- --config config.yaml --start 0 --end 50
```

---

## REPO STRUCTURE

```
sdg_pipeline/
├── run.py
├── config.yaml
├── merge_coco.py
├── validate_output.py
├── base_scene.blend         ← KV-aligned scene (camera/lights/background/collections)
├── KV example datagen.blend ← reference scene (do not modify)
│
├── pipeline/
│   ├── __init__.py
│   ├── asset_registry.py
│   ├── scene_builder.py
│   ├── randomizer.py
│   ├── renderer.py
│   ├── mask_extractor.py
│   └── annotation_writer.py
│
├── assets/
│   ├── models/
│   ├── distractors/
│   ├── textures/
│   └── backgrounds/
│
├── output/
│   ├── images/
│   ├── masks/
│   ├── masks_instance/
│   └── annotations/
│
└── scripts/
    ├── install_deps.sh
    ├── render_parallel.sh
    └── setup_base_scene.py  ← regenerates base_scene.blend from KV example
```

---

## DEPENDENCY INSTALL

Run once. Installs into Blender's bundled Python, not system Python.

```bash
# install_deps.sh
BLENDER_PY=/path/to/blender/3.6/python/bin/python3.10

$BLENDER_PY -m pip install \
  PyYAML \
  numpy \
  Pillow \
  scikit-image \
  shapely \
  pycocotools
```

---

## PHASE BREAKDOWN

There are 7 phases. Each phase is a self-contained unit of work you can hand to a Claude session with this document as context. Phases 1–5 are the critical path. Phases 6–7 are quality-of-life and scale.

---

## PHASE 1 — Config and Asset Registry

**What it builds:** config.yaml schema + asset_registry.py

**Files touched:** config.yaml, pipeline/asset_registry.py

**config.yaml — full schema:**

```yaml
render:
  engine: CYCLES
  device: GPU                  # GPU or CPU
  samples: 64
  use_denoiser: true
  denoiser: OPENIMAGEDENOISE   # or OPTIX for NVIDIA
  resolution_x: 1920
  resolution_y: 1080

scene:
  objects_per_scene_min: 1
  objects_per_scene_max: 4
  distractors_min: 2
  distractors_max: 6

camera:
  distance_min: 1.5
  distance_max: 4.0
  elevation_min_deg: 15
  elevation_max_deg: 75
  focal_length_min: 35
  focal_length_max: 85

lighting:
  num_lights_min: 1
  num_lights_max: 3
  intensity_min: 500
  intensity_max: 4000
  color_temp_min: 3200
  color_temp_max: 6500

material:
  randomize_color: true
  randomize_roughness: true
  roughness_min: 0.05
  roughness_max: 0.95
  randomize_metallic: true
  use_texture_sets: true

assets:
  models:
    - path: assets/models/
      glob: "*.obj"
      category_id: 1
      category_name: your_object
  distractors:
    meshes: assets/distractors/
    use_primitives: true
  textures: assets/textures/
  backgrounds: assets/backgrounds/

output_dir: output/
num_images: 50
seed: 42
```

**asset_registry.py — full implementation:**

AssetRegistry is a dict of named pools. Each pool is a list of Asset dataclasses. The registry is built once from config at startup and queried each loop iteration.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

@dataclass
class Asset:
    path: Path
    category_id: int = 0
    category_name: str = ""
    tags: list = field(default_factory=list)

class AssetRegistry:
    POOLS = ["models", "distractors", "textures", "backgrounds"]

    def __init__(self):
        self._pools = {p: [] for p in self.POOLS}

    def register(self, pool: str, asset: Asset):
        assert pool in self.POOLS, f"Unknown pool: {pool}"
        self._pools[pool].append(asset)

    def sample(self, pool: str, rng: np.random.Generator,
               n: int = 1) -> list:
        candidates = self._pools[pool]
        if not candidates:
            return []
        idx = rng.choice(len(candidates),
                         size=min(n, len(candidates)),
                         replace=False)
        return [candidates[i] for i in idx]

    def count(self, pool: str) -> int:
        return len(self._pools[pool])

    @classmethod
    def from_config(cls, cfg) -> "AssetRegistry":
        reg = cls()

        for entry in cfg["assets"]["models"]:
            for p in Path(entry["path"]).glob(entry.get("glob", "*.obj")):
                reg.register("models", Asset(
                    path=p,
                    category_id=entry["category_id"],
                    category_name=entry["category_name"]
                ))

        mesh_dir = cfg["assets"]["distractors"].get("meshes")
        if mesh_dir:
            for p in Path(mesh_dir).glob("*.obj"):
                reg.register("distractors", Asset(path=p))

        tex_dir = cfg["assets"].get("textures")
        if tex_dir:
            for p in Path(tex_dir).iterdir():
                if p.is_dir():
                    reg.register("textures", Asset(path=p))

        bg_dir = cfg["assets"].get("backgrounds")
        if bg_dir:
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in Path(bg_dir).glob(ext):
                    reg.register("backgrounds", Asset(path=p))

        return reg
```

**Prompt for Claude session:**
> "Build Phase 1 of this SDG pipeline. Implement config.yaml and pipeline/asset_registry.py exactly as specified in the master design document. Include a small __main__ block in asset_registry.py that loads config.yaml and prints pool counts so I can verify it works."

---

## PHASE 2 — Scene Builder

**What it builds:** pipeline/scene_builder.py

**Responsibilities:**
- Clear the `Randomize`, `Occluders`, and `Distractors` collections (preserves camera, lights, Background plane, Bounds empties)
- Import target models (.glb) into the `Randomize` collection
- Spawn distractor meshes into `Distractors` and primitives into `Occluders`
- Add a shadow catcher plane so objects cast shadows on the background
- Set background image via World environment node
- Assign unique integer `inst_id` and `pass_index` to every target object (required for ID_MASK compositor nodes)
- Tag distractors with `is_distractor=True` to exclude from annotations

**Key bpy APIs used:**

```
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.wm.obj_import(filepath=str(path))
bpy.ops.import_scene.fbx(filepath=str(path))
bpy.data.libraries.load(str(path), link=False)   # for .blend

bpy.ops.mesh.primitive_plane_add(size=20, location=(0,0,0))
obj.is_shadow_catcher = True
obj.cycles.is_shadow_catcher = True

world = bpy.context.scene.world
world.use_nodes = True
tex_node = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
tex_node.image = bpy.data.images.load(str(bg_path))
bg_node = world.node_tree.nodes["Background"]
world.node_tree.links.new(tex_node.outputs["Color"], bg_node.inputs["Color"])

obj["inst_id"] = integer
obj["is_distractor"] = True
```

**Primitive distractor ops:**
```
bpy.ops.mesh.primitive_cube_add()
bpy.ops.mesh.primitive_uv_sphere_add()
bpy.ops.mesh.primitive_cylinder_add()
bpy.ops.mesh.primitive_cone_add()
bpy.ops.mesh.primitive_torus_add()
```

**Function signatures to implement:**

```python
def clear_scene() -> None

def import_model(path: Path) -> list   # returns list of bpy Objects

def spawn_targets(assets: list, rng, cfg) -> list
    # imports each asset, returns flat list of all spawned bpy Objects

def spawn_distractors(registry, rng, cfg) -> list
    # spawns primitives + mesh distractors, tags them, returns list

def set_background(bg_path: Path) -> None
    # sets world environment to the background image

def add_shadow_catcher() -> None
    # adds a large plane, marks it shadow catcher

def assign_instance_ids(objects: list, start: int = 1) -> dict
    # assigns obj["inst_id"] = start+i AND obj.pass_index = start+i, returns {inst_id: obj.name}
```

**Return value contract for main loop:**
```python
{
    "target_objects":     [bpy.types.Object, ...],
    "distractor_objects": [bpy.types.Object, ...],
    "id_map":             {inst_id: obj_name, ...},
    "category_map":       {inst_id: category_id, ...}
}
```

**Prompt for Claude session:**
> "Build Phase 2 of this SDG pipeline. Implement pipeline/scene_builder.py with the exact function signatures in the master design document. Use bpy for all Blender operations. The file must be importable as a module and must not call bpy at import time — only inside function bodies."

---

## PHASE 3 — Randomizer

**What it builds:** pipeline/randomizer.py

**Responsibilities:**
- Randomize camera position (spherical coordinates, always look at origin)
- Randomize object position, rotation, scale
- Randomize lights (delete old, spawn new)
- Randomize material (Principled BSDF scalars and/or PBR texture sets)

**All randomization uses a seeded numpy Generator passed in as a parameter. Never call random.random() or numpy.random directly — always use the passed rng.**

**Camera — spherical coords to cartesian:**
```python
import math

def randomize_camera(cam_obj, rng, cfg):
    c = cfg["camera"]
    r     = rng.uniform(c["distance_min"], c["distance_max"])
    theta = rng.uniform(math.radians(c["elevation_min_deg"]),
                        math.radians(c["elevation_max_deg"]))
    phi   = rng.uniform(0, 2 * math.pi)

    cam_obj.location.x = r * math.sin(theta) * math.cos(phi)
    cam_obj.location.y = r * math.sin(theta) * math.sin(phi)
    cam_obj.location.z = r * math.cos(theta)
    cam_obj.data.lens  = rng.uniform(c["focal_length_min"],
                                     c["focal_length_max"])
    # Track-To constraint points camera at world origin
    if "Track To" not in cam_obj.constraints:
        con = cam_obj.constraints.new(type="TRACK_TO")
        con.target = None   # targets origin implicitly when no object set
        con.track_axis  = "TRACK_NEGATIVE_Z"
        con.up_axis     = "UP_Y"
```

**Object transform:**
```python
def randomize_object_transform(obj, rng, spread: float = 1.5):
    obj.location.x = rng.uniform(-spread, spread)
    obj.location.y = rng.uniform(-spread, spread)
    obj.location.z = 0.0
    obj.rotation_euler = [rng.uniform(0, 2 * math.pi) for _ in range(3)]
    s = rng.uniform(0.7, 1.3)
    obj.scale = (s, s, s)
```

**Lighting:**
```python
LIGHT_TYPES = ["POINT", "SUN", "AREA", "SPOT"]

def randomize_lights(scene, rng, cfg):
    lc = cfg["lighting"]
    # Delete existing lights
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    n = int(rng.integers(lc["num_lights_min"], lc["num_lights_max"] + 1))
    for i in range(n):
        ltype = rng.choice(LIGHT_TYPES)
        light_data = bpy.data.lights.new(name=f"sdg_light_{i}", type=ltype)
        light_data.energy = rng.uniform(lc["intensity_min"], lc["intensity_max"])
        light_data.color  = kelvin_to_rgb(rng.uniform(lc["color_temp_min"],
                                                       lc["color_temp_max"]))
        light_obj = bpy.data.objects.new(name=f"sdg_light_{i}",
                                          object_data=light_data)
        scene.collection.objects.link(light_obj)
        light_obj.location = (rng.uniform(-3, 3),
                               rng.uniform(-3, 3),
                               rng.uniform(1.5, 4.5))
```

**Kelvin to RGB (pure Python, no deps):**
```python
def kelvin_to_rgb(temp: float) -> tuple:
    # Tanner Helland's algorithm, returns (r, g, b) in 0.0-1.0
    temp = temp / 100.0
    if temp <= 66:
        r = 1.0
        g = max(0, min(1, (99.4708025861 * math.log(temp) - 161.1195681661) / 255))
        b = 0.0 if temp <= 19 else max(0, min(1, (138.5177312231 * math.log(temp - 10) - 305.0447927307) / 255))
    else:
        r = max(0, min(1, (329.698727446 * ((temp - 60) ** -0.1332047592)) / 255))
        g = max(0, min(1, (288.1221695283 * ((temp - 60) ** -0.0755148492)) / 255))
        b = 1.0
    return (r, g, b)
```

**Material:**
```python
def randomize_material(obj, rng, cfg, texture_asset=None):
    mc = cfg["material"]
    if not obj.data.materials:
        mat = bpy.data.materials.new(name=f"sdg_mat_{obj.name}")
        obj.data.materials.append(mat)
    mat = obj.data.materials[0]
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    if not bsdf:
        return

    if texture_asset and mc.get("use_texture_sets"):
        _apply_texture_set(nodes, links, bsdf, texture_asset.path)
    else:
        if mc["randomize_color"]:
            bsdf.inputs["Base Color"].default_value = (
                *[float(x) for x in rng.uniform(0, 1, 3)], 1.0)
        if mc["randomize_roughness"]:
            bsdf.inputs["Roughness"].default_value = float(
                rng.uniform(mc["roughness_min"], mc["roughness_max"]))
        if mc["randomize_metallic"]:
            bsdf.inputs["Metallic"].default_value = float(rng.choice([0.0, 1.0]))

def _apply_texture_set(nodes, links, bsdf, tex_dir: Path):
    def load_tex(name, colorspace="sRGB"):
        candidates = list(tex_dir.glob(f"*{name}*"))
        if not candidates:
            return None
        n = nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(str(candidates[0]))
        n.image.colorspace_settings.name = colorspace
        return n

    albedo = load_tex("albedo") or load_tex("color") or load_tex("diffuse")
    rough  = load_tex("roughness")
    normal = load_tex("normal")

    if albedo:
        links.new(albedo.outputs["Color"], bsdf.inputs["Base Color"])
    if rough:
        rough.image.colorspace_settings.name = "Non-Color"
        links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    if normal:
        normal.image.colorspace_settings.name = "Non-Color"
        nmap = nodes.new("ShaderNodeNormalMap")
        links.new(normal.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
```

**Prompt for Claude session:**
> "Build Phase 3 of this SDG pipeline. Implement pipeline/randomizer.py with the exact function signatures and algorithms in the master design document. All random draws must use the numpy Generator passed as rng parameter. Include the kelvin_to_rgb helper and the _apply_texture_set helper."

---

## PHASE 4 — Renderer and Mask Extractor

**What it builds:** pipeline/renderer.py + pipeline/mask_extractor.py

**Masking approach: Object Index (ID_MASK)** — not Cryptomatte. For this pipeline (opaque rigid objects, no depth of field), ID_MASK produces identical results with far less complexity: no EXR, no OpenImageIO, no hashing. Each target object gets `obj.pass_index = inst_id`. The compositor creates one `CompositorNodeIDMask` per object, reads from `IndexOB`, and writes a binary PNG per object directly. See `NOTES.md` for the full decision rationale.

**renderer.py responsibilities:**
- Configure Cycles settings from config
- Enable Object Index pass on the view layer (`use_pass_object_index = True`)
- Set up compositor: one ID_MASK node per target object → binary mask PNG; plus RGB PNG output
- Execute render

**renderer.py key implementation:**

```python
from pathlib import Path

def configure_cycles(scene, cfg):
    r = cfg["render"]
    scene.render.engine       = "CYCLES"
    scene.cycles.device       = r["device"]
    scene.cycles.samples      = r["samples"]
    scene.render.resolution_x = r["resolution_x"]
    scene.render.resolution_y = r["resolution_y"]
    if r.get("use_denoiser"):
        scene.cycles.use_denoising = True
        scene.cycles.denoiser      = r["denoiser"]
    else:
        scene.cycles.use_denoising = False

def enable_object_index_pass(view_layer):
    view_layer.use_pass_object_index = True

def setup_compositor(scene, img_idx: int, output_dir: Path, id_map: dict):
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")

    # RGB PNG output
    png_out = tree.nodes.new("CompositorNodeOutputFile")
    png_out.base_path = str(output_dir / "images")
    png_out.file_slots[0].path = f"{img_idx:04d}_"
    png_out.format.file_format = "PNG"
    png_out.format.color_mode  = "RGB"
    tree.links.new(rl.outputs["Image"], png_out.inputs[0])

    if not id_map:
        return

    # One ID_MASK node + File Output slot per target object
    mask_out = tree.nodes.new("CompositorNodeOutputFile")
    mask_out.base_path = str(output_dir / "masks_instance")
    mask_out.format.file_format = "PNG"
    mask_out.format.color_mode  = "BW"
    mask_out.file_slots.clear()

    for inst_id in sorted(id_map.keys()):
        id_mask = tree.nodes.new("CompositorNodeIDMask")
        id_mask.index = inst_id
        id_mask.use_antialiasing = False
        tree.links.new(rl.outputs["IndexOB"], id_mask.inputs["ID value"])
        slot = mask_out.file_slots.new(f"{img_idx:04d}_inst_{inst_id}_")
        tree.links.new(id_mask.outputs["Alpha"], mask_out.inputs[slot.name])

def render(scene):
    import bpy
    bpy.ops.render.render()
```

**mask_extractor.py responsibilities:**
- Load binary mask PNGs written by the compositor's ID_MASK nodes
- Build semantic mask (pixel value = category_id)

**mask_extractor.py implementation:**

```python
import numpy as np
from pathlib import Path
from PIL import Image

def load_instance_masks(img_idx: int, id_map: dict, output_dir: Path) -> dict:
    """
    id_map: {inst_id: obj_name}
    returns: {inst_id: np.ndarray uint8 (H,W), values 0 or 255}
    Files: output_dir/masks_instance/{img_idx:04d}_inst_{inst_id}_0001.png
    """
    result = {}
    for inst_id in id_map:
        path = output_dir / "masks_instance" / f"{img_idx:04d}_inst_{inst_id}_0001.png"
        if not path.exists():
            continue
        arr = np.array(Image.open(path).convert("L"))
        if arr.any():
            result[inst_id] = (arr > 127).astype(np.uint8) * 255
    return result

def build_semantic_mask(instance_masks: dict, category_map: dict,
                        H: int, W: int) -> np.ndarray:
    semantic = np.zeros((H, W), dtype=np.uint8)
    for inst_id, mask in instance_masks.items():
        semantic[mask > 0] = category_map.get(inst_id, 0)
    return semantic
```

**Prompt for Claude session:**
> "Build Phase 4 of this SDG pipeline. Implement pipeline/renderer.py and pipeline/mask_extractor.py using the ID_MASK approach in the master design document. setup_compositor() takes an id_map parameter and creates one CompositorNodeIDMask per object. mask_extractor loads the PNG files written by the compositor — no EXR, no hashing."

---

## PHASE 5 — Annotation Writer and Main Loop

**What it builds:** pipeline/annotation_writer.py + run.py

**This is the integration phase. Everything comes together here.**

**annotation_writer.py responsibilities:**
- Save semantic mask as 8-bit grayscale PNG (pixel value = category_id)
- Save each instance mask as binary PNG (255 = object, 0 = background)
- Accumulate COCO images + annotations lists across the loop
- Write final instances.json at end of run
- Compute polygon segmentation from binary mask using skimage + shapely
- Compute bounding box from binary mask
- Compute area

```python
import json, numpy as np
from pathlib import Path
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon

def save_semantic_mask(mask: np.ndarray, path: Path):
    Image.fromarray(mask, mode="L").save(str(path))

def save_instance_mask(mask: np.ndarray, path: Path):
    Image.fromarray(mask, mode="L").save(str(path))

def mask_to_polygons(binary_mask: np.ndarray, tolerance: int = 2) -> list:
    contours = measure.find_contours(binary_mask, 0.5)
    polygons = []
    for contour in contours:
        contour = np.flip(contour, axis=1)  # row,col -> x,y
        if len(contour) < 3:
            continue
        poly = Polygon(contour).simplify(tolerance, preserve_topology=False)
        if poly.is_valid and not poly.is_empty and poly.area > 1:
            coords = list(poly.exterior.coords)
            flat   = [v for pt in coords for v in pt]
            polygons.append(flat)
    return polygons

def compute_bbox(binary_mask: np.ndarray) -> list:
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]

def init_coco(cfg) -> dict:
    categories = []
    for entry in cfg["assets"]["models"]:
        categories.append({
            "id":            entry["category_id"],
            "name":          entry["category_name"],
            "supercategory": "object"
        })
    return {
        "info":        {"description": "SDG Pipeline", "version": "1.0"},
        "licenses":    [],
        "categories":  categories,
        "images":      [],
        "annotations": []
    }

def write_coco(coco: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(coco, f, indent=2)
```

**run.py — the main loop:**

```python
import sys, argparse, yaml, logging, time
from pathlib import Path
import numpy as np
import bpy

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.asset_registry   import AssetRegistry
from pipeline.scene_builder     import (clear_scene, spawn_targets,
                                         spawn_distractors, set_background,
                                         add_shadow_catcher, assign_instance_ids)
from pipeline.randomizer        import (randomize_camera, randomize_lights,
                                         randomize_object_transform,
                                         randomize_material)
from pipeline.renderer          import (configure_cycles, enable_object_index_pass,
                                         setup_compositor, render)
from pipeline.mask_extractor    import load_instance_masks, build_semantic_mask
from pipeline.annotation_writer import (save_semantic_mask, save_instance_mask,
                                         mask_to_polygons, compute_bbox,
                                         init_coco, write_coco)

# ── Argument parsing (after the -- separator) ──────────────────────────────
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--start",  type=int, default=0)
parser.add_argument("--end",    type=int, default=None)
args = parser.parse_args(argv)

# ── Setup ───────────────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

output_dir = Path(cfg["output_dir"])
for d in ["images", "masks", "masks_instance", "annotations"]:
    (output_dir / d).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(str(output_dir / "pipeline.log")),
        logging.StreamHandler()
    ]
)

scene      = bpy.context.scene
view_layer = scene.view_layers[0]
registry   = AssetRegistry.from_config(cfg)
end        = args.end or cfg["num_images"]

configure_cycles(scene, cfg)
enable_object_index_pass(view_layer)

coco    = init_coco(cfg)
ann_id  = 0
cam_obj = scene.camera

# ── Main loop ───────────────────────────────────────────────────────────────
for img_idx in range(args.start, end):
    t0  = time.perf_counter()
    rng = np.random.default_rng(cfg["seed"] + img_idx)

    # 1. Scene
    clear_scene()
    add_shadow_catcher()
    bg = registry.sample("backgrounds", rng, n=1)
    if bg:
        set_background(bg[0].path)

    n_targets = int(rng.integers(cfg["scene"]["objects_per_scene_min"],
                                  cfg["scene"]["objects_per_scene_max"] + 1))
    target_assets = registry.sample("models", rng, n=n_targets)
    target_objs   = spawn_targets(target_assets, rng, cfg)
    distractor_objs = spawn_distractors(registry, rng, cfg)

    id_map       = assign_instance_ids(target_objs, start=1)
    category_map = {iid: bpy.data.objects[name].get("category_id", 1)
                    for iid, name in id_map.items()}

    # 2. Randomize
    randomize_camera(cam_obj, rng, cfg)
    randomize_lights(scene, rng, cfg)
    tex = registry.sample("textures", rng, n=1)
    tex = tex[0] if tex else None
    for obj in target_objs + distractor_objs:
        randomize_object_transform(obj, rng)
        randomize_material(obj, rng, cfg, texture_asset=tex)

    # 3. Render — compositor writes RGB PNG + per-object mask PNGs
    setup_compositor(scene, img_idx, output_dir, id_map)
    render(scene)

    # 4. Load masks (written by compositor ID_MASK nodes)
    H = cfg["render"]["resolution_y"]
    W = cfg["render"]["resolution_x"]
    instance_masks = load_instance_masks(img_idx, id_map, output_dir)
    semantic_mask  = build_semantic_mask(instance_masks, category_map, H, W)

    # 5. Save semantic mask
    save_semantic_mask(semantic_mask,
                       output_dir / "masks" / f"{img_idx:04d}_semantic.png")

    # 6. COCO annotations (target objects only, skip distractors)
    img_w, img_h = W, H
    coco["images"].append({
        "id":        img_idx,
        "file_name": f"images/{img_idx:04d}_0001.png",
        "width":     img_w,
        "height":    img_h
    })
    for inst_id, mask in instance_masks.items():
        polygons = mask_to_polygons(mask)
        if not polygons:
            continue
        area = int(np.sum(mask > 0))
        bbox = compute_bbox(mask)
        coco["annotations"].append({
            "id":          ann_id,
            "image_id":    img_idx,
            "category_id": category_map.get(inst_id, 1),
            "segmentation": polygons,
            "area":         area,
            "bbox":         bbox,
            "iscrowd":      0
        })
        ann_id += 1

    elapsed = time.perf_counter() - t0
    logging.info(f"img={img_idx:04d} | objects={len(target_objs)} | "
                 f"distractors={len(distractor_objs)} | "
                 f"annotations={ann_id} | time={elapsed:.1f}s")

# ── Write COCO JSON ──────────────────────────────────────────────────────────
json_path = output_dir / "annotations" / f"instances_{args.start}_{end}.json"
write_coco(coco, json_path)
logging.info(f"Done. Wrote {json_path}")
```

**Prompt for Claude session:**
> "Build Phase 5 of this SDG pipeline. Implement pipeline/annotation_writer.py and run.py exactly as specified in the master design document. The main loop in run.py must import all pipeline modules, follow the 7-step structure (scene, randomize, render, extract, save masks, COCO, cleanup), and write the COCO JSON at the end. Do not add error handling or try/except blocks — we want fast iteration, not defensive code."

---

## PHASE 6 — Validation and COCO Merge

**What it builds:** validate_output.py + merge_coco.py

**validate_output.py — run after generation to catch problems before training:**

Checks: image count matches expected, every image file exists, every mask file exists, image dimensions match config, every image has at least one annotation, no zero-area annotations, no images with annotation count above a sanity threshold (50+), COCO JSON is valid JSON.

Output: PASS with counts, or FAIL with list of specific issues.

```python
# validate_output.py
import json, sys, yaml
from pathlib import Path
from PIL import Image

def validate(output_dir: str, config: str):
    cfg      = yaml.safe_load(open(config))
    out      = Path(output_dir)
    ann_file = next((out / "annotations").glob("instances*.json"), None)
    issues   = []

    if not ann_file:
        print("FAIL: No annotations JSON found.")
        return

    data = json.load(open(ann_file))
    imgs = {img["id"]: img for img in data["images"]}
    anns = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    W = cfg["render"]["resolution_x"]
    H = cfg["render"]["resolution_y"]

    for img in data["images"]:
        iid      = img["id"]
        rgb      = out / img["file_name"]
        sem      = out / "masks" / Path(img["file_name"]).stem.replace("_0001","") + "_semantic.png"

        if not rgb.exists():
            issues.append(f"Missing RGB: {rgb}")
        else:
            iw, ih = Image.open(rgb).size
            if (iw, ih) != (W, H):
                issues.append(f"Wrong size {iw}x{ih} for {rgb}")

        if iid not in anns:
            issues.append(f"No annotations for image_id={iid}")
        else:
            for a in anns[iid]:
                if a["area"] == 0:
                    issues.append(f"Zero-area annotation {a['id']} in image {iid}")

    if issues:
        print(f"FAIL — {len(issues)} issues:")
        for i in issues: print(f"  {i}")
        sys.exit(1)
    else:
        n_img = len(data["images"])
        n_ann = len(data["annotations"])
        print(f"PASS — {n_img} images, {n_ann} annotations, 0 issues.")
```

**merge_coco.py — merge partial JSONs from parallel jobs:**

```python
# merge_coco.py
import json, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("annotations_dir")
parser.add_argument("--out", default="instances.json")
args = parser.parse_args()

parts  = sorted(Path(args.annotations_dir).glob("instances_*.json"))
merged = None
img_offset = 0
ann_offset = 0

for p in parts:
    data = json.load(open(p))
    if merged is None:
        merged = {k: v for k, v in data.items()}
        merged["images"]      = []
        merged["annotations"] = []

    for img in data["images"]:
        img["id"] += img_offset
        merged["images"].append(img)

    for ann in data["annotations"]:
        ann["id"]       += ann_offset
        ann["image_id"] += img_offset
        merged["annotations"].append(ann)

    img_offset += len(data["images"])
    ann_offset += len(data["annotations"])

out_path = Path(args.annotations_dir) / args.out
json.dump(merged, open(out_path, "w"), indent=2)
print(f"Merged {len(parts)} files -> {out_path}")
print(f"Total: {len(merged['images'])} images, {len(merged['annotations'])} annotations")
```

**Prompt for Claude session:**
> "Build Phase 6 of this SDG pipeline. Implement validate_output.py and merge_coco.py exactly as specified in the master design document. validate_output.py takes --output_dir and --config arguments. merge_coco.py takes an annotations directory and --out filename. Both must be runnable as standalone scripts."

---

## PHASE 7 — Parallel Launch Scripts and README

**What it builds:** scripts/render_parallel.sh + README.md

**render_parallel.sh:**

```bash
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
```

**README.md minimum sections:**

```
# SDG Pipeline

## Quick Start
1. Install Blender 3.6 LTS
2. Run scripts/install_deps.sh (edit BLENDER_PY path first)
3. Put your .obj model in assets/models/
4. Put background images in assets/backgrounds/
5. Edit config.yaml: set category_name, num_images
6. Run: blender -b base_scene.blend -P run.py -- --config config.yaml
7. Output in output/images/, output/masks/, output/annotations/

## Generating 2,000 Images (Multi-GPU)
./scripts/render_parallel.sh 2000 4

## Output Format
- output/images/NNNN_0001.png       RGB render
- output/masks/NNNN_semantic.png    Grayscale, pixel=category_id
- output/masks_instance/            One PNG per object instance
- output/annotations/instances.json COCO format

## COCO JSON Schema
Standard COCO instance segmentation.
categories[].id matches pixel values in semantic masks.

## Adding a New Object Class
In config.yaml, add to assets.models:
  - path: assets/models/new_class/
    glob: "*.obj"
    category_id: 2
    category_name: new_class

## Adding Distractors
Drop .obj files into assets/distractors/.
Set use_primitives: true for random Blender primitives.
Distractor count controlled by scene.distractors_min/max.

## Tuning Render Speed
- Decrease render.samples (32 is usable with denoiser on)
- Set render.device: GPU
- Set render.denoiser: OPTIX for NVIDIA cards

## Troubleshooting
Black images: check render.device matches your hardware
Missing masks: check assets/backgrounds/ has at least one image
COCO has 0 annotations: object may be outside camera view —
  tighten camera distance or increase objects_per_scene
```

**Prompt for Claude session:**
> "Build Phase 7 of this SDG pipeline. Write scripts/render_parallel.sh and README.md exactly as specified in the master design document. The shell script must accept total_images and num_gpus as positional arguments, launch Blender jobs in background processes with CUDA_VISIBLE_DEVICES, and call merge_coco.py and validate_output.py after all jobs finish. The README must cover all sections listed."

---

## HACKATHON EXECUTION ORDER

```
Hour 0-1   Phase 1  config.yaml + asset_registry.py
Hour 1-2   Phase 2  scene_builder.py
Hour 2-3   Phase 3  randomizer.py
Hour 3-5   Phase 4  renderer.py + mask_extractor.py  ← hardest, most time
Hour 5-6   Phase 5  annotation_writer.py + run.py
Hour 6-7   INTEGRATION TEST — run 5 images, inspect output
Hour 7-8   Debug mask extraction / compositor issues
Hour 8-9   Phase 6  validate_output.py + merge_coco.py
Hour 9-10  Phase 7  scripts + README
Hour 10+   Run 50-image demo batch, fix whatever breaks
```

---

## KNOWN GOTCHAS

**EXR channel naming:** Blender appends the view layer name to Cryptomatte channel names. They will be named like `ViewLayer.CryptoObject00.R` not just `CryptoObject00.R`. The extractor must strip or handle the prefix. Use `"CryptoObject" in k` as the filter, not an exact match.

**Compositor file output naming:** Blender appends `0001` to the frame number by default in compositor file output nodes. Your image files will be `NNNN_0001.png` not `NNNN.png`. Account for this in the COCO image file_name field.

**bpy import at module level:** Never call `bpy.context` or `bpy.data` at module import time — Blender's context may not be ready. Always call bpy inside function bodies.

**Shadow catcher visibility:** The shadow catcher plane should be on its own collection and excluded from the Cryptomatte pass so it doesn't generate a mask. Set `obj.hide_render = False` but `obj["is_distractor"] = True` to exclude it from annotation.

**GPU not detected:** Set `bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'` and call `bpy.context.preferences.addons['cycles'].preferences.get_devices()` before setting `scene.cycles.device = 'GPU'` in configure_cycles.

**Object import leaves selection state dirty:** After any `bpy.ops.wm.obj_import()` call, the imported objects are selected. Use `bpy.context.selected_objects` immediately after import to capture them before doing anything else that might change the selection.