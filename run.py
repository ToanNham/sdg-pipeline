import sys, site
sys.path.insert(0, site.getusersitepackages())
import argparse, yaml, logging, time, math
from pathlib import Path
import numpy as np
import bpy

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.asset_registry   import AssetRegistry
from pipeline.scene_builder     import (clear_scene, spawn_targets,
                                        spawn_distractors, set_background,
                                        assign_instance_ids)
from pipeline.randomizer        import (randomize_camera, randomize_lights,
                                        randomize_object_transform,
                                        randomize_material,
                                        randomize_background_roughness)
from pipeline.renderer          import (configure_cycles, enable_object_index_pass,
                                        setup_compositor, render)
from pipeline.mask_extractor    import load_instance_masks, build_semantic_mask
from pipeline.annotation_writer import (save_semantic_mask, save_instance_mask,
                                        save_instance_color_mask,
                                        assign_instance_colors,
                                        mask_to_polygons, compute_bbox,
                                        init_coco, write_coco, write_label_json)

# ── Argument parsing (after the -- separator) ──────────────────────────────
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--start",  type=int, default=0)
parser.add_argument("--end",    type=int, default=None)
parser.add_argument("--debug",  action="store_true",
                    help="Render 1 image, skip all randomization, force object visibility")
args = parser.parse_args(argv)

# ── Setup ───────────────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

output_dir = (Path(args.config).parent / cfg["output_dir"]).resolve()
for d in ["images", "masks", "masks_instance", "annotations", "labels", "blends"]:
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
end        = (args.start + 1) if args.debug else (args.end or cfg["num_images"])

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
    inst_colors  = assign_instance_colors(id_map)  # {inst_id: (R,G,B)} — shared by PNG + label

    # 2. Randomize (skipped in --debug mode)
    tex         = None
    light_info  = {}
    bg_roughness = 0.5
    if not args.debug:
        randomize_camera(cam_obj, rng, cfg)
        light_info  = randomize_lights(scene, rng, cfg)
        bg_roughness = randomize_background_roughness(scene, rng, cfg)
        tex = registry.sample("textures", rng, n=1)
        tex = tex[0] if tex else None
        d_scale_min = cfg["scene"].get("distractor_scale_min", 0.05)
        d_scale_max = cfg["scene"].get("distractor_scale_max", 0.15)
        t_spread = cfg["scene"].get("target_spread", 0.5)
        for obj in target_objs:
            randomize_object_transform(obj, rng, spread=t_spread)
            randomize_material(obj, rng, cfg, texture_asset=tex)
        for obj in distractor_objs:
            randomize_object_transform(obj, rng, scale_min=d_scale_min, scale_max=d_scale_max)
            randomize_material(obj, rng, cfg, texture_asset=tex)

    # In debug mode, read light info from whatever is in the scene
    if args.debug:
        for obj in scene.objects:
            if obj.type == "LIGHT":
                tgt = list(obj.get("target_loc", [0.0, 0.0, 0.0]))
                # Fallback: look for a Spot_Target empty
                if tgt == [0.0, 0.0, 0.0]:
                    st = bpy.data.objects.get("Spot_Target")
                    if st:
                        tgt = [round(v, 8) for v in st.location]
                light_info = {
                    "Spot_Light_Location":    [round(v, 8) for v in obj.location],
                    "Light_Target_Location":  tgt,
                    "Spot_Light_Energy":      obj.data.energy,
                    "Spot_Light_Temperature": obj.get("color_temp_K", 6500),
                }
                break
        # Background roughness from scene
        bg_col = bpy.data.collections.get("Background")
        if bg_col:
            for obj in bg_col.objects:
                if obj.type == "MESH" and obj.data.materials:
                    mat = obj.data.materials[0]
                    if mat and mat.use_nodes:
                        bsdf = next((n for n in mat.node_tree.nodes
                                     if n.type == "BSDF_PRINCIPLED"), None)
                        if bsdf:
                            bg_roughness = float(bsdf.inputs["Roughness"].default_value)

    # DEBUG: print camera and object positions
    logging.info(f"  cam_loc={tuple(round(x,3) for x in cam_obj.location)} "
                 f"cam_lens={round(cam_obj.data.lens,1)}mm")
    for iid, name in id_map.items():
        obj = bpy.data.objects[name]
        hide_render = obj.hide_render
        hide_viewport = obj.hide_viewport
        cols = [c.name for c in obj.users_collection]
        col_hide = [bpy.data.collections[c].hide_render for c in cols if c in bpy.data.collections]
        logging.info(f"  target inst_{iid} '{name}' loc={tuple(round(x,3) for x in obj.location)} "
                     f"scale={round(obj.scale[0],3)} pass_index={obj.pass_index} "
                     f"hide_render={hide_render} cols={cols} col_hide_render={col_hide}")

    # 3. Render — compositor writes RGB PNG + per-object mask PNGs
    setup_compositor(scene, img_idx, output_dir, id_map)
    render(scene)
    blend_path = str(output_dir / "blends" / f"{img_idx:04d}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, copy=True)

    # 4. Load masks (written by compositor)
    H = cfg["render"]["resolution_y"]
    W = cfg["render"]["resolution_x"]
    instance_masks = load_instance_masks(img_idx, id_map, output_dir)
    semantic_mask  = build_semantic_mask(instance_masks, category_map, H, W)

    # 5. Save masks
    save_semantic_mask(semantic_mask,
                       output_dir / "masks" / f"{img_idx:04d}_semantic.png")
    save_instance_color_mask(instance_masks, inst_colors,
                             output_dir / "masks" / f"{img_idx:04d}_color.png",
                             H, W)

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

    # 7. Per-image label JSON (KV format)
    label_objects = []
    for inst_id, name in id_map.items():
        obj   = bpy.data.objects[name]
        color = list(inst_colors[inst_id])
        label_objects.append({
            "name":       name,
            "position":   [round(v, 8) for v in obj.location],
            "rotation":   [round(math.degrees(a), 8) for a in obj.rotation_euler],
            "mask_color": color,
        })
    for obj in distractor_objs:
        entry = {
            "name":     obj.name,
            "position": [round(v, 8) for v in obj.location],
            "rotation": [round(math.degrees(a), 8) for a in obj.rotation_euler],
        }
        if tex:
            entry["material_image"] = tex.path.name
        label_objects.append(entry)

    label = {
        "Light_Target_Location":  light_info.get("Light_Target_Location",  [0.0, 0.0, 0.0]),
        "Spot_Light_Location":    light_info.get("Spot_Light_Location",    [0.0, 0.0, 0.0]),
        "Spot_Light_Energy":      light_info.get("Spot_Light_Energy",      100.0),
        "Spot_Light_Temperature": light_info.get("Spot_Light_Temperature", 6500),
        "Background_Roughness":   round(bg_roughness, 8),
        "Wall_Image":             bg[0].path.name if bg else "",
        "Objects":                label_objects,
    }
    write_label_json(label, output_dir / "labels" / f"{img_idx:04d}_label.json")

    elapsed = time.perf_counter() - t0
    logging.info(f"img={img_idx:04d} | objects={len(target_objs)} | "
                 f"distractors={len(distractor_objs)} | "
                 f"annotations={ann_id} | time={elapsed:.1f}s")

# ── Write COCO JSON ──────────────────────────────────────────────────────────
json_path = output_dir / "annotations" / f"instances_{args.start}_{end}.json"
write_coco(coco, json_path)
logging.info(f"Done. Wrote {json_path}")
