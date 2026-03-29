import sys, site
sys.path.insert(0, site.getusersitepackages())
import argparse, yaml, logging, time, math
from pathlib import Path
import numpy as np
import bpy

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.asset_registry   import AssetRegistry
from pipeline.scene_builder     import (set_background,
                                        build_target_pool, build_distractor_pool,
                                        activate_frame_objects)
from pipeline.randomizer        import (randomize_camera, randomize_light_inplace,
                                        randomize_object_transform,
                                        randomize_material,
                                        randomize_background_material)
from pipeline.renderer          import (configure_cycles, enable_object_index_pass,
                                        setup_compositor, render)
from pipeline.annotation_writer import (assign_instance_colors,
                                        write_label_json)

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
for d in ["images", "masks", "labels", "blends"]:
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

cam_obj     = scene.camera
bounds_col  = bpy.data.collections.get("Bounds")
BOUNDS_OBJS = [o for o in bounds_col.objects if o.type == "EMPTY"] if bounds_col else []

# ── Pre-spawn phase (once) ──────────────────────────────────────────────────
target_pool     = build_target_pool(registry, cfg)
distractor_pool = build_distractor_pool(registry, cfg)
pool_light      = next((o for o in scene.objects if o.type == "LIGHT"), None)
static_category_map = {
    obj.pass_index: obj.get("category_id", 1)
    for objs in target_pool.values() for obj in objs
}

# ── Main loop ───────────────────────────────────────────────────────────────
for img_idx in range(args.start, end):
    t0  = time.perf_counter()
    rng = np.random.default_rng(cfg["seed"] + img_idx)

    # 1. Scene — activate objects for this frame
    active_targets, active_distractors, id_map = activate_frame_objects(
        target_pool, distractor_pool, rng, cfg
    )
    bg = registry.sample("backgrounds", rng, n=1)
    if bg:
        set_background(bg[0].path)

    category_map = {pid: static_category_map[pid] for pid in id_map}
    inst_colors  = assign_instance_colors(id_map)  # {inst_id: (R,G,B)} — shared by PNG + label

    # 2. Randomize (skipped in --debug mode)
    tex         = None
    light_info  = {}
    bg_roughness = 0.5
    if not args.debug:
        if pool_light:
            light_info = randomize_light_inplace(pool_light, rng, cfg)
        if bg:
            bg_roughness = randomize_background_material(scene, rng, cfg, bg[0].path)
        tex = registry.sample("textures", rng, n=1)
        tex = tex[0] if tex else None
        t_spread = cfg["scene"].get("target_spread", 0.5)
        for obj in active_targets:
            randomize_object_transform(obj, rng, spread=t_spread,
                                       bounds_objs=BOUNDS_OBJS, randomize_scale=False)
            randomize_material(obj, rng, cfg, texture_asset=None)  # targets have embedded textures
        for obj in active_distractors:
            randomize_object_transform(obj, rng, bounds_objs=BOUNDS_OBJS, randomize_scale=False)
            randomize_material(obj, rng, cfg, texture_asset=tex)

    # In debug mode, read light info from pool_light
    if args.debug:
        if pool_light:
            tgt = list(pool_light.get("target_loc", [0.0, 0.0, 0.0]))
            if tgt == [0.0, 0.0, 0.0]:
                st = bpy.data.objects.get("Spot_Target")
                if st:
                    tgt = [round(v, 8) for v in st.location]
            light_info = {
                "Spot_Light_Location":    [round(v, 8) for v in pool_light.location],
                "Light_Target_Location":  tgt,
                "Spot_Light_Energy":      pool_light.data.energy,
                "Spot_Light_Temperature": pool_light.get("color_temp_K", 6500),
            }
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

    # 3. Render — compositor writes RGB PNG + color instance mask PNG
    setup_compositor(scene, img_idx, output_dir, id_map, inst_colors)
    render(scene)
    blend_path = str(output_dir / "blends" / f"{img_idx:04d}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, copy=True)

    # 6. Per-image label JSON (KV format)
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
    for obj in active_distractors:
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
    logging.info(f"img={img_idx:04d} | objects={len(active_targets)} | "
                 f"distractors={len(active_distractors)} | time={elapsed:.1f}s")

logging.info("Done.")
