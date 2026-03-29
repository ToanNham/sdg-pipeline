import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import bpy       # module-level import is fine; do NOT call bpy.* at module level
import mathutils  # available in Blender's Python and standalone bpy==4.2.0


_LIGHT_TYPES = ["POINT", "SUN", "AREA", "SPOT"]


# ---------------------------------------------------------------------------
# Color temperature
# ---------------------------------------------------------------------------

def kelvin_to_rgb(temp: float) -> Tuple[float, float, float]:
    """Convert color temperature in Kelvin to an RGB tuple (0.0–1.0 each).

    Uses Tanner Helland's algorithm. Valid range: ~1000 K – 40000 K.
    """
    def _clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    temp = max(1000.0, min(40000.0, temp)) / 100.0

    # Red
    if temp <= 66:
        r = 1.0
    else:
        r = _clamp(329.698727446 * (temp - 60) ** -0.1332047592 / 255.0)

    # Green
    if temp <= 66:
        g = _clamp((99.4708025861 * math.log(temp) - 161.1195681661) / 255.0)
    else:
        g = _clamp(288.1221695283 * (temp - 60) ** -0.0755148492 / 255.0)

    # Blue
    if temp >= 66:
        b = 1.0
    elif temp <= 19:
        b = 0.0
    else:
        b = _clamp((138.5177312231 * math.log(temp - 10) - 305.0447927307) / 255.0)

    return (r, g, b)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def randomize_camera(cam_obj, rng: np.random.Generator, cfg: dict) -> None:
    """Randomize camera position (spherical coords) and focal length.

    Camera is always oriented to face the world origin.
    """
    c = cfg["camera"]

    r = float(rng.uniform(c["distance_min"], c["distance_max"]))
    theta = float(rng.uniform(
        math.radians(c["elevation_min_deg"]),
        math.radians(c["elevation_max_deg"]),
    ))
    phi = float(rng.uniform(0.0, 2.0 * math.pi))

    cam_obj.location.x = r * math.sin(theta) * math.cos(phi)
    cam_obj.location.y = r * math.sin(theta) * math.sin(phi)
    cam_obj.location.z = r * math.cos(theta)

    cam_obj.data.lens = float(rng.uniform(c["focal_length_min"], c["focal_length_max"]))

    # Point camera at world origin using mathutils (no constraint state needed)
    direction = mathutils.Vector((0.0, 0.0, 0.0)) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


# ---------------------------------------------------------------------------
# Object transforms
# ---------------------------------------------------------------------------

def randomize_object_transform(
    obj,
    rng: np.random.Generator,
    spread: float = 1.5,
    scale_min: float = 0.7,
    scale_max: float = 1.3,
    bounds_objs: list = None,
    randomize_scale: bool = True,
    rotation_x_min: float = None,
    rotation_x_max: float = None,
    rotation_y_min: float = None,
    rotation_y_max: float = None,
    rotation_z_min: float = None,
    rotation_z_max: float = None,
) -> None:
    """Randomize location, rotation (all axes), and optionally uniform scale.

    If bounds_objs is provided (list of Blender Empty objects), location is
    sampled within the union bounding box of those empties.  Falls back to
    the spread-based placement when bounds_objs is empty or all None.

    Pass randomize_scale=False to preserve the object's imported scale (used
    for target .glb models whose apparent size should vary only with camera
    distance, not explicit scale).
    """
    if bounds_objs:
        # Pick one bounding box at random (equal probability) and place within it
        empty = bounds_objs[int(rng.integers(0, len(bounds_objs)))]
        loc = empty.location
        sc  = empty.scale
        obj.location.x = float(rng.uniform(loc.x - abs(sc.x), loc.x + abs(sc.x)))
        obj.location.y = float(rng.uniform(loc.y - abs(sc.y), loc.y + abs(sc.y)))
        obj.location.z = float(rng.uniform(loc.z - abs(sc.z), loc.z + abs(sc.z)))
    else:
        obj.location.x = float(rng.uniform(-spread, spread))
        obj.location.y = float(rng.uniform(-spread, spread))
        obj.location.z = 0.0

    if rotation_x_min is not None and rotation_x_max is not None:
        rx = float(rng.uniform(math.radians(rotation_x_min), math.radians(rotation_x_max)))
    else:
        rx = float(rng.uniform(0.0, 2.0 * math.pi))
    if rotation_y_min is not None and rotation_y_max is not None:
        ry = float(rng.uniform(math.radians(rotation_y_min), math.radians(rotation_y_max)))
    else:
        ry = float(rng.uniform(0.0, 2.0 * math.pi))
    if rotation_z_min is not None and rotation_z_max is not None:
        rz = float(rng.uniform(math.radians(rotation_z_min), math.radians(rotation_z_max)))
    else:
        rz = float(rng.uniform(0.0, 2.0 * math.pi))
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = (rx, ry, rz)

    if randomize_scale:
        s = float(rng.uniform(scale_min, scale_max))
        obj.scale = (s, s, s)


# ---------------------------------------------------------------------------
# Collision / overlap helpers
# ---------------------------------------------------------------------------

def get_world_aabb(obj, margin: float = 0.0):
    """World-space AABB of obj as (min_co, max_co).

    Call bpy.context.view_layer.update() before this to ensure
    obj.matrix_world reflects any recent location/rotation/scale changes.
    margin: expand AABB outward by this many metres on every side.
    """
    corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
    min_co = mathutils.Vector((
        min(v.x for v in corners) - margin,
        min(v.y for v in corners) - margin,
        min(v.z for v in corners) - margin,
    ))
    max_co = mathutils.Vector((
        max(v.x for v in corners) + margin,
        max(v.y for v in corners) + margin,
        max(v.z for v in corners) + margin,
    ))
    return min_co, max_co


def aabbs_overlap(a_min, a_max, b_min, b_max) -> bool:
    """Return True if two AABBs overlap (touching counts)."""
    return (
        a_min.x <= b_max.x and a_max.x >= b_min.x and
        a_min.y <= b_max.y and a_max.y >= b_min.y and
        a_min.z <= b_max.z and a_max.z >= b_min.z
    )


def place_object_no_overlap(
    obj,
    rng: np.random.Generator,
    placed_aabbs: list,
    spread: float = 1.5,
    scale_min: float = 0.7,
    scale_max: float = 1.3,
    bounds_objs: list = None,
    randomize_scale: bool = True,
    max_retries: int = 10,
    margin: float = 0.0,
    rotation_x_min: float = None,
    rotation_x_max: float = None,
    rotation_y_min: float = None,
    rotation_y_max: float = None,
    rotation_z_min: float = None,
    rotation_z_max: float = None,
) -> None:
    """Place obj without overlapping placed_aabbs; retry up to max_retries times.

    Delegates to randomize_object_transform() for each candidate placement.
    bpy.context.view_layer.update() is called after each placement so
    obj.matrix_world is current before the AABB is read.
    On exit, the object's final AABB is appended to placed_aabbs.
    """
    for _ in range(max_retries + 1):
        randomize_object_transform(
            obj, rng,
            spread=spread,
            scale_min=scale_min,
            scale_max=scale_max,
            bounds_objs=bounds_objs,
            randomize_scale=randomize_scale,
            rotation_x_min=rotation_x_min,
            rotation_x_max=rotation_x_max,
            rotation_y_min=rotation_y_min,
            rotation_y_max=rotation_y_max,
            rotation_z_min=rotation_z_min,
            rotation_z_max=rotation_z_max,
        )
        bpy.context.view_layer.update()
        new_min, new_max = get_world_aabb(obj, margin=margin)
        if not any(aabbs_overlap(new_min, new_max, p_min, p_max)
                   for p_min, p_max in placed_aabbs):
            break  # non-overlapping position found
        # final iteration: accept overlapping position silently

    placed_aabbs.append((new_min, new_max))


# ---------------------------------------------------------------------------
# Group AABB and placement
# ---------------------------------------------------------------------------

def get_group_world_aabb(members: list, margin: float = 0.0):
    """World-space union AABB of all child mesh objects in a group.

    Call bpy.context.view_layer.update() before this to ensure matrix_world
    of each child is current after the pivot was moved.
    margin is applied to the union boundary (not per-child).
    """
    all_mins = []
    all_maxs = []
    for child in members:
        c_min, c_max = get_world_aabb(child, margin=0.0)
        all_mins.append(c_min)
        all_maxs.append(c_max)
    union_min = mathutils.Vector((
        min(v.x for v in all_mins) - margin,
        min(v.y for v in all_mins) - margin,
        min(v.z for v in all_mins) - margin,
    ))
    union_max = mathutils.Vector((
        max(v.x for v in all_maxs) + margin,
        max(v.y for v in all_maxs) + margin,
        max(v.z for v in all_maxs) + margin,
    ))
    return union_min, union_max


def place_group_no_overlap(
    group,
    rng: np.random.Generator,
    placed_aabbs: list,
    spread: float = 1.5,
    bounds_objs: list = None,
    max_retries: int = 15,
    margin: float = 0.0,
    rotation_x_min: float = None,
    rotation_x_max: float = None,
    rotation_y_min: float = None,
    rotation_y_max: float = None,
    rotation_z_min: float = None,
    rotation_z_max: float = None,
    rotate_as_unit: bool = True,
) -> None:
    """Place a GroupArrangement's pivot without overlapping placed_aabbs.

    Moves only the pivot; children follow via parenting.  Calls
    bpy.context.view_layer.update() after each trial so child world transforms
    are current before the union AABB is computed.
    On exit, the group's final union AABB is appended to placed_aabbs.
    """
    pivot = group.pivot

    for _ in range(max_retries + 1):
        # Sample pivot position using same logic as randomize_object_transform
        if bounds_objs:
            empty = bounds_objs[int(rng.integers(0, len(bounds_objs)))]
            loc = empty.location
            sc  = empty.scale
            pivot.location.x = float(rng.uniform(loc.x - abs(sc.x), loc.x + abs(sc.x)))
            pivot.location.y = float(rng.uniform(loc.y - abs(sc.y), loc.y + abs(sc.y)))
            pivot.location.z = float(rng.uniform(loc.z - abs(sc.z), loc.z + abs(sc.z)))
        else:
            pivot.location.x = float(rng.uniform(-spread, spread))
            pivot.location.y = float(rng.uniform(-spread, spread))
            pivot.location.z = 0.0

        if rotate_as_unit:
            pivot.rotation_mode = 'XYZ'
            if rotation_x_min is not None and rotation_x_max is not None:
                rx = float(rng.uniform(math.radians(rotation_x_min), math.radians(rotation_x_max)))
            else:
                rx = float(rng.uniform(0.0, 2.0 * math.pi))
            if rotation_y_min is not None and rotation_y_max is not None:
                ry = float(rng.uniform(math.radians(rotation_y_min), math.radians(rotation_y_max)))
            else:
                ry = float(rng.uniform(0.0, 2.0 * math.pi))
            if rotation_z_min is not None and rotation_z_max is not None:
                rz = float(rng.uniform(math.radians(rotation_z_min), math.radians(rotation_z_max)))
            else:
                rz = float(rng.uniform(0.0, 2.0 * math.pi))
            pivot.rotation_euler = (rx, ry, rz)

        bpy.context.view_layer.update()
        new_min, new_max = get_group_world_aabb(group.members, margin=margin)
        if not any(aabbs_overlap(new_min, new_max, p_min, p_max)
                   for p_min, p_max in placed_aabbs):
            break

    placed_aabbs.append((new_min, new_max))


# ---------------------------------------------------------------------------
# Lights
# ---------------------------------------------------------------------------

def randomize_lights(scene, rng: np.random.Generator, cfg: dict) -> dict:
    """Delete all existing lights and spawn a single randomized SPOT light.

    Returns a dict with the light's scene properties for label generation:
        {
            "Spot_Light_Location":   [x, y, z],
            "Light_Target_Location": [x, y, z],
            "Spot_Light_Energy":     float,
            "Spot_Light_Temperature": int,
        }
    """
    lc = cfg["lighting"]

    # Remove all current light objects
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    ktemp  = float(rng.uniform(lc["color_temp_min"], lc["color_temp_max"]))
    energy = float(rng.uniform(lc["intensity_min"],  lc["intensity_max"]))

    light_data = bpy.data.lights.new(name="sdg_spot", type="SPOT")
    light_data.energy     = energy
    light_data.color      = kelvin_to_rgb(ktemp)
    light_data.spot_size  = float(rng.uniform(math.radians(30), math.radians(90)))

    light_obj = bpy.data.objects.new(name="sdg_spot", object_data=light_data)
    light_obj["color_temp_K"] = int(round(ktemp))
    scene.collection.objects.link(light_obj)

    # Position: sample on sphere of radius [radius_min, radius_max], reject if
    # inside the camera frustum cone (camera fixed at (2.87,0,0) looking at -X).
    CAM_POS  = mathutils.Vector((2.87, 0.0, 0.0))
    VIEW_DIR = mathutils.Vector((-1.0,  0.0, 0.0))
    FOV_HALF = math.radians(32.0)   # 32° exclusion half-angle (covers 35mm FOV + margin)
    FALLBACK = mathutils.Vector((0.0, 3.5, 2.0))  # guaranteed off-screen

    r_min = lc.get("radius_min", 2.0)
    r_max = lc.get("radius_max", 4.5)

    light_pos = None
    for _ in range(20):
        r     = float(rng.uniform(r_min, r_max))
        theta = float(rng.uniform(0.0, math.pi))
        phi   = float(rng.uniform(0.0, 2.0 * math.pi))
        candidate = mathutils.Vector((
            r * math.sin(theta) * math.cos(phi),
            r * math.sin(theta) * math.sin(phi),
            r * math.cos(theta),
        ))
        to_light = (candidate - CAM_POS).normalized()
        cos_a    = max(-1.0, min(1.0, to_light.dot(VIEW_DIR)))
        if math.acos(cos_a) >= FOV_HALF:
            light_pos = candidate
            break
    if light_pos is None:
        light_pos = FALLBACK

    light_obj.location.x = light_pos.x
    light_obj.location.y = light_pos.y
    light_obj.location.z = light_pos.z

    # Target: a point near scene centre along the X-axis
    target_loc = mathutils.Vector((
        float(rng.uniform(0.3, 0.8)),
        float(rng.uniform(-0.2, 0.2)),
        0.0,
    ))
    direction = target_loc - light_obj.location
    light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    light_obj["target_loc"] = list(target_loc)

    return {
        "Spot_Light_Location":    [round(v, 8) for v in light_obj.location],
        "Light_Target_Location":  [round(v, 8) for v in target_loc],
        "Spot_Light_Energy":      energy,
        "Spot_Light_Temperature": int(round(ktemp)),
    }


def randomize_light_inplace(light_obj, rng: np.random.Generator, cfg: dict) -> dict:
    """Randomize an existing SPOT light object in-place (no delete/create).

    Same logic as randomize_lights() but operates on a pre-existing light_obj.
    Returns the same dict format for label JSON compatibility.
    """
    lc = cfg["lighting"]

    ktemp  = float(rng.uniform(lc["color_temp_min"], lc["color_temp_max"]))
    energy = float(rng.uniform(lc["intensity_min"],  lc["intensity_max"]))

    light_obj.data.energy    = energy
    light_obj.data.color     = kelvin_to_rgb(ktemp)
    light_obj.data.spot_size = float(rng.uniform(math.radians(30), math.radians(90)))
    light_obj["color_temp_K"] = int(round(ktemp))

    CAM_POS  = mathutils.Vector((2.87, 0.0, 0.0))
    VIEW_DIR = mathutils.Vector((-1.0,  0.0, 0.0))
    FOV_HALF = math.radians(32.0)
    FALLBACK = mathutils.Vector((0.0, 3.5, 2.0))

    r_min = lc.get("radius_min", 2.0)
    r_max = lc.get("radius_max", 4.5)

    light_pos = None
    for _ in range(20):
        r     = float(rng.uniform(r_min, r_max))
        theta = float(rng.uniform(0.0, math.pi))
        phi   = float(rng.uniform(0.0, 2.0 * math.pi))
        candidate = mathutils.Vector((
            r * math.sin(theta) * math.cos(phi),
            r * math.sin(theta) * math.sin(phi),
            r * math.cos(theta),
        ))
        to_light = (candidate - CAM_POS).normalized()
        cos_a    = max(-1.0, min(1.0, to_light.dot(VIEW_DIR)))
        if math.acos(cos_a) >= FOV_HALF:
            light_pos = candidate
            break
    if light_pos is None:
        light_pos = FALLBACK

    light_obj.location.x = light_pos.x
    light_obj.location.y = light_pos.y
    light_obj.location.z = light_pos.z

    target_loc = mathutils.Vector((
        float(rng.uniform(0.3, 0.8)),
        float(rng.uniform(-0.2, 0.2)),
        0.0,
    ))
    direction = target_loc - light_obj.location
    light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    light_obj["target_loc"] = list(target_loc)

    return {
        "Spot_Light_Location":    [round(v, 8) for v in light_obj.location],
        "Light_Target_Location":  [round(v, 8) for v in target_loc],
        "Spot_Light_Energy":      energy,
        "Spot_Light_Temperature": int(round(ktemp)),
    }


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

def _apply_texture_set(nodes, links, bsdf, tex_dir: Path) -> None:
    """Load PBR textures from tex_dir and wire them into the Principled BSDF."""

    def load_tex(keyword: str, colorspace: str = "sRGB"):
        candidates = list(tex_dir.glob(f"*{keyword}*"))
        if not candidates:
            # Case-insensitive fallback for Linux production
            candidates = list(tex_dir.glob(f"*{keyword.capitalize()}*"))
        if not candidates:
            return None
        node = nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(
            str(candidates[0].resolve()), check_existing=True
        )
        node.image.colorspace_settings.name = colorspace
        return node

    albedo_node = load_tex("albedo") or load_tex("color") or load_tex("diffuse")
    rough_node = load_tex("roughness", colorspace="Non-Color")
    normal_node = load_tex("normal", colorspace="Non-Color")

    if albedo_node:
        links.new(albedo_node.outputs["Color"], bsdf.inputs["Base Color"])
    if rough_node:
        links.new(rough_node.outputs["Color"], bsdf.inputs["Roughness"])
    if normal_node:
        nmap_node = nodes.new("ShaderNodeNormalMap")
        links.new(normal_node.outputs["Color"], nmap_node.inputs["Color"])
        links.new(nmap_node.outputs["Normal"], bsdf.inputs["Normal"])


def randomize_background_roughness(scene, rng: np.random.Generator, cfg: dict) -> float:
    """Set roughness on the background plane material; return the chosen value."""
    sc = cfg["scene"]
    roughness = float(rng.uniform(
        sc.get("background_roughness_min", 0.3),
        sc.get("background_roughness_max", 1.0),
    ))
    bg_mesh = bpy.data.objects.get("Background")
    if bg_mesh and bg_mesh.type == "MESH" and bg_mesh.data.materials:
        mat = bg_mesh.data.materials[0]
        if mat and mat.use_nodes:
            bsdf = next((n for n in mat.node_tree.nodes
                         if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf:
                bsdf.inputs["Roughness"].default_value = roughness
    return roughness


def randomize_background_material(
    scene,
    rng: np.random.Generator,
    cfg: dict,
    bg_image_path: Path,
) -> float:
    """Apply bg_image_path as Base Color texture on the Background plane and
    randomize its roughness.  Returns the roughness value for label logging.
    """
    sc = cfg["scene"]
    roughness = float(rng.uniform(
        sc.get("background_roughness_min", 0.3),
        sc.get("background_roughness_max", 1.0),
    ))

    bg_mesh = bpy.data.objects.get("Background")
    if bg_mesh is None or bg_mesh.type != "MESH":
        return roughness

    if not bg_mesh.data.materials:
        mat = bpy.data.materials.new(name="sdg_bg_mat")
        bg_mesh.data.materials.append(mat)
    mat = bg_mesh.data.materials[0]
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        nodes.clear()
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        out  = nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tex_node = nodes.get("sdg_bg_tex")
    if tex_node is None:
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.name = "sdg_bg_tex"

    tex_node.image = bpy.data.images.load(
        str(Path(bg_image_path).resolve()), check_existing=True
    )
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    return roughness


def randomize_material(
    obj,
    rng: np.random.Generator,
    cfg: dict,
    texture_asset=None,
) -> None:
    """Randomize the material on obj.

    Pass texture_asset=None for .glb target objects (has_embedded_textures=True);
    the caller is responsible for this check since has_embedded_textures lives on
    the Asset dataclass, not on the Blender object.
    """
    if not obj.data.materials:
        mat = bpy.data.materials.new(name=f"sdg_mat_{obj.name}")
        obj.data.materials.append(mat)
    mat = obj.data.materials[0]

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Locale-safe BSDF lookup by type rather than by name
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return

    mc = cfg["material"]
    if texture_asset is not None and mc.get("use_texture_sets", False):
        _apply_texture_set(nodes, links, bsdf, texture_asset.path)
    else:
        if mc.get("randomize_color", True):
            r, g, b = (float(x) for x in rng.uniform(0.0, 1.0, 3))
            bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
        if mc.get("randomize_roughness", True):
            bsdf.inputs["Roughness"].default_value = float(
                rng.uniform(mc["roughness_min"], mc["roughness_max"])
            )
        if mc.get("randomize_metallic", True):
            bsdf.inputs["Metallic"].default_value = float(
                rng.choice(np.array([0.0, 1.0]))
            )


# ---------------------------------------------------------------------------
# Extensibility wrapper
# ---------------------------------------------------------------------------

class Randomizer:
    """Thin wrapper around module-level randomization functions.

    Subclass and override individual methods to customise specific stages
    without touching any other part of the pipeline.
    """

    def randomize_camera(self, cam_obj, rng, cfg):
        randomize_camera(cam_obj, rng, cfg)

    def randomize_light(self, light_obj, rng, cfg):
        return randomize_light_inplace(light_obj, rng, cfg)

    def randomize_background(self, scene, rng, cfg, bg_image_path):
        return randomize_background_material(scene, rng, cfg, bg_image_path)

    def place_object(self, obj, rng, placed_aabbs, spread=1.5,
                     scale_min=0.7, scale_max=1.3, bounds_objs=None,
                     randomize_scale=True, max_retries=10, margin=0.0,
                     rotation_x_min=None, rotation_x_max=None,
                     rotation_y_min=None, rotation_y_max=None,
                     rotation_z_min=None, rotation_z_max=None):
        place_object_no_overlap(
            obj, rng, placed_aabbs,
            spread=spread,
            scale_min=scale_min,
            scale_max=scale_max,
            bounds_objs=bounds_objs,
            randomize_scale=randomize_scale,
            max_retries=max_retries,
            margin=margin,
            rotation_x_min=rotation_x_min,
            rotation_x_max=rotation_x_max,
            rotation_y_min=rotation_y_min,
            rotation_y_max=rotation_y_max,
            rotation_z_min=rotation_z_min,
            rotation_z_max=rotation_z_max,
        )

    def place_group(self, group, rng, placed_aabbs, spread=1.5,
                    bounds_objs=None, max_retries=15, margin=0.0,
                    rotation_x_min=None, rotation_x_max=None,
                    rotation_y_min=None, rotation_y_max=None,
                    rotation_z_min=None, rotation_z_max=None,
                    rotate_as_unit=True):
        place_group_no_overlap(
            group, rng, placed_aabbs,
            spread=spread,
            bounds_objs=bounds_objs,
            max_retries=max_retries,
            margin=margin,
            rotation_x_min=rotation_x_min,
            rotation_x_max=rotation_x_max,
            rotation_y_min=rotation_y_min,
            rotation_y_max=rotation_y_max,
            rotation_z_min=rotation_z_min,
            rotation_z_max=rotation_z_max,
            rotate_as_unit=rotate_as_unit,
        )

    def randomize_material(self, obj, rng, cfg, texture_asset=None):
        randomize_material(obj, rng, cfg, texture_asset)
