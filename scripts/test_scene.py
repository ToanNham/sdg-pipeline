"""Tests for pipeline/scene_builder.py — must run inside Blender (uses bpy).

Run:
    blender -b base_scene.blend -P scripts/test_scene.py

Tests: clear_scene, add_shadow_catcher, set_background, import_model,
assign_instance_ids, spawn_targets, spawn_distractors.
"""
import sys
sys.path.insert(0, ".")

import bpy
from pathlib import Path
from pipeline.scene_builder import (
    clear_scene, add_shadow_catcher, assign_instance_ids,
    import_model, set_background, spawn_targets, spawn_distractors
)
from pipeline.asset_registry import AssetRegistry
import yaml

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

print("=== Initial scene ===")
print("Objects:", [o.name for o in bpy.data.objects])

# --- clear_scene ---
clear_scene()
print("\n=== After clear_scene ===")
print("Objects:", [o.name for o in bpy.data.objects])

# --- add_shadow_catcher ---
add_shadow_catcher()
sc = bpy.data.objects.get("ShadowCatcher")
print("\n=== Shadow catcher ===")
print("Present:", sc is not None)
if sc:
    print("is_shadow_catcher:", sc.is_shadow_catcher)
    print("is_distractor tag:", sc["is_distractor"])

# --- set_background ---
import random
bg_dir = Path(cfg["assets"]["backgrounds"])
bg_files = list(bg_dir.glob("**/*.jpg"))
if bg_files:
    bg_path = bg_files[0]
    set_background(bg_path)
    world = bpy.context.scene.world
    tex_nodes = [n for n in world.node_tree.nodes if n.type == "TEX_ENVIRONMENT"]
    print("\n=== Background ===")
    print("World nodes:", [n.name for n in world.node_tree.nodes])
    print("Tex env nodes:", len(tex_nodes))

# --- import_model (one .glb) ---
glb_path = Path("assets/models/coke.glb")
print("\n=== import_model:", glb_path, "===")
imported = import_model(glb_path)
print("Imported objects:", [o.name for o in imported])

# --- assign_instance_ids ---
id_map = assign_instance_ids(imported, start=1)
print("\n=== Instance IDs ===")
print("id_map:", id_map)
for inst_id, name in id_map.items():
    obj = bpy.data.objects[name]
    print(f"  {name}: inst_id={obj['inst_id']}")

print("\n=== All objects after build ===")
print([o.name for o in bpy.data.objects])
print("\nDone.")
