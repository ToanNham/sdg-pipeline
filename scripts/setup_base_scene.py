"""One-time script to regenerate base_scene.blend from the KV example.

Opens "KV example datagen.blend", strips the product-specific objects and
occluder primitives, configures Cycles + Object Index pass, rebuilds the
compositor with a simple preview node, and saves as base_scene.blend.

Run:
    C:/Users/nhamd/miniconda3/envs/sdg-pipeline/python.exe scripts/setup_base_scene.py
"""

import math
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parent.parent
KV_BLEND = str(ROOT / "KV example datagen.blend")
OUT_BLEND = str(ROOT / "base_scene.blend")

# ---------------------------------------------------------------------------
# Open KV example
# ---------------------------------------------------------------------------
bpy.ops.wm.open_mainfile(filepath=KV_BLEND)
scene = bpy.context.scene

# ---------------------------------------------------------------------------
# Remove product + occluder objects (leave collection shells intact)
# ---------------------------------------------------------------------------
REMOVE_OBJECTS = {
    # Randomize collection — product models
    "CherryCoke", "CocaCola", "CokeZero", "DietCoke", "Fanta", "Sprite",
    # Occluders collection — primitive meshes
    "Cube", "Icosphere", "Sphere", "Suzanne", "Torus",
}

bpy.ops.object.select_all(action="DESELECT")
for name in REMOVE_OBJECTS:
    obj = bpy.data.objects.get(name)
    if obj is not None:
        # Unlink from all collections first
        for col in obj.users_collection:
            col.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)

# ---------------------------------------------------------------------------
# Render settings — match KV example exactly
# ---------------------------------------------------------------------------
scene.render.engine = "CYCLES"
scene.render.resolution_x = 1080
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.use_motion_blur = False

scene.cycles.device = "GPU"
scene.cycles.samples = 20
scene.cycles.preview_samples = 20
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPTIX"
scene.cycles.use_adaptive_sampling = True

# ---------------------------------------------------------------------------
# View layer: enable Object Index pass, disable Cryptomatte
# ---------------------------------------------------------------------------
vl = scene.view_layers[0]
vl.use_pass_object_index = True
vl.use_pass_cryptomatte_object = False
vl.use_pass_cryptomatte_material = False
vl.use_pass_cryptomatte_asset = False

# ---------------------------------------------------------------------------
# Compositor: minimal preview-only setup
# run.py rebuilds this each frame with per-object ID_MASK nodes
# ---------------------------------------------------------------------------
scene.use_nodes = True
tree = scene.node_tree
tree.nodes.clear()

rl = tree.nodes.new("CompositorNodeRLayers")
comp = tree.nodes.new("CompositorNodeComposite")
tree.links.new(rl.outputs["Image"], comp.inputs["Image"])

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND)
print(f"Saved: {OUT_BLEND}")
