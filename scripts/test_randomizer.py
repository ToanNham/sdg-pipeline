import sys
sys.path.insert(0, ".")

import math
import numpy as np
import bpy
import yaml
from pathlib import Path

from pipeline.randomizer import (
    kelvin_to_rgb,
    randomize_camera,
    randomize_object_transform,
    randomize_lights,
    randomize_material,
    _apply_texture_set,
)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

rng = np.random.default_rng(42)

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"{status} {label}{suffix}")
    if not condition:
        raise SystemExit(1)

# ---------------------------------------------------------------------------
# 1. kelvin_to_rgb
# ---------------------------------------------------------------------------
print("\n=== kelvin_to_rgb ===")

r, g, b = kelvin_to_rgb(6500)
check("6500K all channels in [0,1]", all(0.0 <= v <= 1.0 for v in (r, g, b)), f"rgb={r:.3f},{g:.3f},{b:.3f}")
check("6500K is cool (r~g~b near 1)", r > 0.95 and g > 0.95, f"r={r:.3f} g={g:.3f}")

r, g, b = kelvin_to_rgb(3200)
check("3200K is warm (r>b)", r > b, f"r={r:.3f} b={b:.3f}")

r, g, b = kelvin_to_rgb(1000)
check("1000K clamp low (no exception)", True)
r, g, b = kelvin_to_rgb(40000)
check("40000K clamp high (no exception)", True)

# ---------------------------------------------------------------------------
# 2. randomize_camera
# ---------------------------------------------------------------------------
print("\n=== randomize_camera ===")

cam = bpy.context.scene.camera
check("Camera exists in scene", cam is not None, str(cam))

orig_x = cam.location.x
randomize_camera(cam, rng, cfg)

dist = math.sqrt(cam.location.x**2 + cam.location.y**2 + cam.location.z**2)
dmin, dmax = cfg["camera"]["distance_min"], cfg["camera"]["distance_max"]
check("Camera distance in [distance_min, distance_max]", dmin <= dist <= dmax,
      f"dist={dist:.3f}, range=[{dmin},{dmax}]")

z = cam.location.z
check("Camera Z > 0 (above ground due to elevation)", z > 0, f"z={z:.3f}")

fmin, fmax = cfg["camera"]["focal_length_min"], cfg["camera"]["focal_length_max"]
check("Focal length in range", fmin <= cam.data.lens <= fmax,
      f"lens={cam.data.lens:.1f}mm, range=[{fmin},{fmax}]")

# Camera should face roughly toward origin — forward vector dot with -location should be positive
import mathutils
bpy.context.view_layer.update()  # flush matrix_world after rotation_euler assignment
fwd = cam.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
to_origin = -cam.location.normalized()
dot = fwd.dot(to_origin)
check("Camera forward vector points toward origin (dot > 0.99)", dot > 0.99, f"dot={dot:.4f}")

# ---------------------------------------------------------------------------
# 3. randomize_object_transform
# ---------------------------------------------------------------------------
print("\n=== randomize_object_transform ===")

bpy.ops.mesh.primitive_cube_add()
cube = bpy.context.active_object
cube.name = "sdg_test_cube"

randomize_object_transform(cube, rng, spread=1.5)

check("Z location is 0", cube.location.z == 0.0, f"z={cube.location.z}")
check("X location in [-1.5, 1.5]", -1.5 <= cube.location.x <= 1.5, f"x={cube.location.x:.3f}")
check("Y location in [-1.5, 1.5]", -1.5 <= cube.location.y <= 1.5, f"y={cube.location.y:.3f}")
check("Rotation X in [0, 2*pi]", 0 <= cube.rotation_euler[0] <= 2 * math.pi,
      f"rx={cube.rotation_euler[0]:.3f}")
sx, sy, sz = cube.scale
check("Scale is uniform", sx == sy == sz, f"scale={sx:.3f},{sy:.3f},{sz:.3f}")
check("Scale in [0.7, 1.3]", 0.7 <= sx <= 1.3, f"scale={sx:.3f}")

# Determinism: same seed → same result
rng_a = np.random.default_rng(99)
rng_b = np.random.default_rng(99)
bpy.ops.mesh.primitive_cube_add()
obj_a = bpy.context.active_object
bpy.ops.mesh.primitive_cube_add()
obj_b = bpy.context.active_object
randomize_object_transform(obj_a, rng_a)
randomize_object_transform(obj_b, rng_b)
check("Deterministic with same seed",
      obj_a.location.x == obj_b.location.x and obj_a.scale.x == obj_b.scale.x,
      f"x_a={obj_a.location.x:.4f} x_b={obj_b.location.x:.4f}")

# Cleanup test objects
for name in ["sdg_test_cube", obj_a.name, obj_b.name]:
    o = bpy.data.objects.get(name)
    if o:
        bpy.data.objects.remove(o, do_unlink=True)

# ---------------------------------------------------------------------------
# 4. randomize_lights
# ---------------------------------------------------------------------------
print("\n=== randomize_lights ===")

scene = bpy.context.scene
lmin, lmax = cfg["lighting"]["num_lights_min"], cfg["lighting"]["num_lights_max"]

# Add a dummy light to verify deletion
dummy_data = bpy.data.lights.new("dummy_light", type="POINT")
dummy_obj = bpy.data.objects.new("dummy_light", dummy_data)
scene.collection.objects.link(dummy_obj)
pre_lights = [o for o in scene.objects if o.type == "LIGHT"]
check("Pre-existing light is in scene", len(pre_lights) >= 1)

randomize_lights(scene, rng, cfg)

lights_after = [o for o in scene.objects if o.type == "LIGHT"]
check("Old lights deleted and new ones spawned",
      all(o.name != "dummy_light" for o in lights_after),
      f"light names: {[o.name for o in lights_after]}")
check(f"Light count in [{lmin}, {lmax}]", lmin <= len(lights_after) <= lmax,
      f"count={len(lights_after)}")

for lobj in lights_after:
    ld = lobj.data
    emin, emax = cfg["lighting"]["intensity_min"], cfg["lighting"]["intensity_max"]
    check(f"  {lobj.name} energy in range", emin <= ld.energy <= emax,
          f"energy={ld.energy:.1f}")
    check(f"  {lobj.name} color channels in [0,1]",
          all(0.0 <= v <= 1.0 for v in ld.color[:3]),
          f"color={tuple(round(v,3) for v in ld.color[:3])}")
    check(f"  {lobj.name} Z > 0", lobj.location.z > 0, f"z={lobj.location.z:.2f}")
    check(f"  {lobj.name} type valid", ld.type in ("POINT","SUN","AREA","SPOT"),
          f"type={ld.type}")

# ---------------------------------------------------------------------------
# 5. _apply_texture_set  (no texture files on disk → all nodes are None → no-op)
# ---------------------------------------------------------------------------
print("\n=== _apply_texture_set (empty dir, graceful no-op) ===")

bpy.ops.mesh.primitive_uv_sphere_add()
sphere = bpy.context.active_object
sphere.name = "sdg_test_sphere"
mat = bpy.data.materials.new("sdg_test_mat")
mat.use_nodes = True
sphere.data.materials.append(mat)
nodes = mat.node_tree.nodes
links = mat.node_tree.links
bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
check("BSDF node found for texture test", bsdf is not None)

import tempfile, os
empty_dir = Path(tempfile.mkdtemp())
_apply_texture_set(nodes, links, bsdf, empty_dir)
tex_nodes = [n for n in nodes if n.type == "TEX_IMAGE"]
check("No texture nodes added for empty dir", len(tex_nodes) == 0, f"count={len(tex_nodes)}")

# Cleanup
bpy.data.objects.remove(sphere, do_unlink=True)
bpy.data.materials.remove(mat)
os.rmdir(str(empty_dir))

# ---------------------------------------------------------------------------
# 6. randomize_material — procedural branch (no texture asset)
# ---------------------------------------------------------------------------
print("\n=== randomize_material (procedural) ===")

bpy.ops.mesh.primitive_cube_add()
mat_cube = bpy.context.active_object
mat_cube.name = "sdg_mat_cube"

randomize_material(mat_cube, rng, cfg, texture_asset=None)

check("Material was created", len(mat_cube.data.materials) > 0)
mat = mat_cube.data.materials[0]
check("Node materials enabled", mat.use_nodes)
bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
check("Principled BSDF exists", bsdf is not None)

base_color = bsdf.inputs["Base Color"].default_value
check("Base Color is 4-tuple-like with alpha=1", len(base_color) == 4 and base_color[3] == 1.0,
      f"rgba={tuple(round(v,3) for v in base_color)}")
check("Base Color channels in [0,1]",
      all(0.0 <= v <= 1.0 for v in base_color[:3]),
      f"rgb={tuple(round(v,3) for v in base_color[:3])}")

rmin, rmax = cfg["material"]["roughness_min"], cfg["material"]["roughness_max"]
roughness = bsdf.inputs["Roughness"].default_value
check("Roughness in [roughness_min, roughness_max]", rmin <= roughness <= rmax,
      f"roughness={roughness:.3f}, range=[{rmin},{rmax}]")

metallic = bsdf.inputs["Metallic"].default_value
check("Metallic is 0.0 or 1.0", metallic in (0.0, 1.0), f"metallic={metallic}")

# --- object with no material slot (auto-create) ---
bpy.ops.mesh.primitive_cone_add()
bare_obj = bpy.context.active_object
bare_obj.name = "sdg_bare_cone"
bare_obj.data.materials.clear()
check("Pre-condition: no materials", len(bare_obj.data.materials) == 0)
randomize_material(bare_obj, rng, cfg, texture_asset=None)
check("Material auto-created for bare object", len(bare_obj.data.materials) == 1)

# Cleanup
for name in ["sdg_mat_cube", "sdg_bare_cone"]:
    o = bpy.data.objects.get(name)
    if o:
        bpy.data.objects.remove(o, do_unlink=True)

# ---------------------------------------------------------------------------
print("\n=== All tests passed ===")
