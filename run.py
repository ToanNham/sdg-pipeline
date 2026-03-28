import sys, site
sys.path.insert(0, site.getusersitepackages())
import argparse, yaml, logging, time
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

output_dir = (Path(args.config).parent / cfg["output_dir"]).resolve()
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

    # 4. Load masks (written by compositor)
    H = cfg["render"]["resolution_y"]
    W = cfg["render"]["resolution_x"]
    instance_masks = load_instance_masks(img_idx, id_map, output_dir)
    semantic_mask  = build_semantic_mask(instance_masks, category_map, H, W)

    # 5. Save semantic mask
    save_semantic_mask(semantic_mask,
                       output_dir / "masks" / f"{img_idx:04d}_semantic.png")

    # 6. COCO annotations (target objects only)
    coco["images"].append({
        "id":        img_idx,
        "file_name": f"images/{img_idx:04d}_0001.png",
        "width":     W,
        "height":    H
    })
    for inst_id, mask in instance_masks.items():
        polygons = mask_to_polygons(mask)
        if not polygons:
            continue
        area = int(np.sum(mask > 0))
        bbox = compute_bbox(mask)
        coco["annotations"].append({
            "id":           ann_id,
            "image_id":     img_idx,
            "category_id":  category_map.get(inst_id, 1),
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
