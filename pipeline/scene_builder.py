import logging
from dataclasses import dataclass
from pathlib import Path

import bpy  # module-level import is fine; do NOT call bpy.* at module level
import mathutils


# ---------------------------------------------------------------------------
# Group arrangement
# ---------------------------------------------------------------------------

@dataclass
class GroupArrangement:
    """A rigid group of objects parented to a single Empty pivot."""
    pivot:   object        # bpy Empty (the parent object)
    members: list          # list of bpy MESH objects (children)
    rows:    int
    cols:    int
    stem:    str           # model class — all members from same class


def create_group_pivot(name: str, collection_name: str = "Randomize"):
    """Create a named Empty object and link it to collection_name.

    Uses bpy.data (no bpy.ops) so it is safe in headless mode.
    """
    pivot = bpy.data.objects.new(name, None)   # None object_data = Empty
    pivot.empty_display_type = "PLAIN_AXES"
    pivot.empty_display_size = 0.05
    col = bpy.data.collections.get(collection_name)
    if col is not None:
        col.objects.link(pivot)
    return pivot


def parent_to_pivot(child, pivot) -> None:
    """Parent child to pivot without changing child's world position.

    Direct attribute assignment — no bpy.ops, safe in headless mode.
    """
    child.parent = pivot
    child.matrix_parent_inverse = pivot.matrix_world.inverted()


def layout_group_local(
    members: list, rows: int, cols: int,
    spacing_x: float, spacing_y: float = None,
) -> None:
    """Place each member at a grid offset in the pivot's local space.

    The grid is centered on the pivot origin.  Call after parent_to_pivot so
    that matrix_local encodes the grid offset relative to the pivot.
    spacing_x / spacing_y are the center-to-center distances per axis.
    Pass the object's bbox dimension (+ optional gap) to get flush packing.
    """
    if spacing_y is None:
        spacing_y = spacing_x
    offset_x = -(cols - 1) / 2.0 * spacing_x
    offset_y = -(rows - 1) / 2.0 * spacing_y
    for idx, obj in enumerate(members):
        r = idx // cols
        c = idx % cols
        obj.matrix_local = mathutils.Matrix.Translation((
            offset_x + c * spacing_x,
            offset_y + r * spacing_y,
            0.0,
        ))


def destroy_group(group: GroupArrangement) -> None:
    """Restore group members to un-parented state and delete the pivot.

    Call after render, before the next frame's activation.
    """
    for child in group.members:
        child.parent = None
        child.matrix_world = mathutils.Matrix()
        child.hide_render = True
        child.hide_viewport = True
    col = bpy.data.collections.get("Randomize")
    if col is not None and group.pivot.name in col.objects:
        col.objects.unlink(group.pivot)
    bpy.data.objects.remove(group.pivot, do_unlink=True)


# ---------------------------------------------------------------------------
# Scene management
# ---------------------------------------------------------------------------

def clear_scene() -> None:
    """Remove all objects from the Randomize, Occluders, and Distractors collections.

    Preserves persistent objects (Camera, Background plane, Lights, Bounds).
    """
    for col_name in ("Randomize", "Occluders", "Distractors"):
        col = bpy.data.collections.get(col_name)
        if col is None:
            continue
        for obj in list(col.objects):
            col.objects.unlink(obj)
            bpy.data.objects.remove(obj, do_unlink=True)


# ---------------------------------------------------------------------------
# Model import
# ---------------------------------------------------------------------------

def import_model(path: Path, collection_name: str = None) -> list:
    """Import a model file and return newly added MESH objects.

    Args:
        path:            Path to model file (.glb/.gltf/.obj/.fbx/.blend)
        collection_name: If given, link new objects into this named collection
                         instead of the default active collection.
    """
    path = Path(path).resolve()
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()

    if suffix in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == '.obj':
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix == '.fbx':
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix == '.blend':
        with bpy.data.libraries.load(str(path), link=False) as (data_from, data_to):
            data_to.objects = list(data_from.objects)
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported model format: {suffix}")

    after = set(bpy.data.objects)
    new_objs = [obj for obj in (after - before) if obj.type == 'MESH']

    # Apply scale and rotation so random rotations don't cause shearing on
    # meshes that were exported with non-uniform or non-applied transforms.
    for obj in new_objs:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        obj.select_set(False)

    if collection_name:
        col = bpy.data.collections.get(collection_name)
        if col is not None:
            for obj in new_objs:
                # Move from whatever collection the import put it in
                for src_col in list(obj.users_collection):
                    src_col.objects.unlink(obj)
                col.objects.link(obj)

    return new_objs


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------

def spawn_targets(assets: list, rng, cfg) -> list:
    """Import each target asset into the Randomize collection."""
    all_objects = []
    for asset in assets:
        objs = import_model(asset.path, collection_name="Randomize")
        for obj in objs:
            obj["category_id"] = asset.category_id
            obj["category_name"] = asset.category_name
            obj.hide_render = False
            obj.hide_viewport = False
        all_objects.extend(objs)

    col = bpy.data.collections.get("Randomize")
    if col:
        col.hide_render = False

    return all_objects


def spawn_distractors(registry, rng, cfg) -> list:
    """Spawn mesh and/or primitive distractors; tag each is_distractor=True."""
    scene_cfg = cfg["scene"]
    n = int(rng.integers(scene_cfg["distractors_min"],
                         scene_cfg["distractors_max"] + 1))

    use_primitives = cfg["assets"]["distractors"].get("use_primitives", True)
    mesh_assets = registry.sample("distractors", rng, n)

    spawned = []

    # Spawn mesh distractors into Distractors collection
    for asset in mesh_assets:
        objs = import_model(asset.path, collection_name="Distractors")
        for obj in objs:
            obj["is_distractor"] = True
        spawned.extend(objs)

    # Pad with primitives into Occluders collection
    n_prim = max(0, n - len(mesh_assets)) if use_primitives else 0
    prim_size = float(cfg["scene"].get("distractor_primitive_size", 0.2))
    half = prim_size / 2.0
    _PRIMITIVE_OPS = [
        lambda: bpy.ops.mesh.primitive_cube_add(size=prim_size),
        lambda: bpy.ops.mesh.primitive_uv_sphere_add(radius=half),
        lambda: bpy.ops.mesh.primitive_cylinder_add(radius=half, depth=prim_size),
        lambda: bpy.ops.mesh.primitive_cone_add(radius1=half, depth=prim_size),
        lambda: bpy.ops.mesh.primitive_torus_add(major_radius=half, minor_radius=half * 0.3),
    ]
    occluders_col = bpy.data.collections.get("Occluders")
    for _ in range(n_prim):
        before = set(bpy.data.objects)
        _PRIMITIVE_OPS[int(rng.integers(0, len(_PRIMITIVE_OPS)))]()
        after = set(bpy.data.objects)
        new_objs = [o for o in (after - before) if o.type == 'MESH']
        for obj in new_objs:
            obj["is_distractor"] = True
            if occluders_col is not None:
                for src_col in list(obj.users_collection):
                    src_col.objects.unlink(obj)
                occluders_col.objects.link(obj)
        spawned.extend(new_objs)

    return spawned


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------

def set_background(bg_path: Path) -> None:
    """Set the world environment to the given background image."""
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    nodes.clear()

    tex_node = nodes.new("ShaderNodeTexEnvironment")
    tex_node.image = bpy.data.images.load(str(Path(bg_path).resolve()), check_existing=True)

    bg_node = nodes.new("ShaderNodeBackground")
    output_node = nodes.new("ShaderNodeOutputWorld")

    links.new(tex_node.outputs["Color"], bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])


# ---------------------------------------------------------------------------
# Shadow catcher
# ---------------------------------------------------------------------------

def add_shadow_catcher() -> None:
    """Add a large ground plane and mark it as a Cycles shadow catcher."""
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "ShadowCatcher"
    obj.is_shadow_catcher = True
    obj.cycles.is_shadow_catcher = True
    obj["is_distractor"] = True  # exclude from COCO annotations


# ---------------------------------------------------------------------------
# Instance ID assignment
# ---------------------------------------------------------------------------

def assign_instance_ids(objects: list, start: int = 1) -> dict:
    """Assign sequential inst_id and pass_index to each object; return {inst_id: obj.name}."""
    id_map = {}
    for i, obj in enumerate(objects):
        inst_id = start + i
        obj["inst_id"] = inst_id
        obj.pass_index = inst_id  # required for IndexOB / ID_MASK compositor nodes
        id_map[inst_id] = obj.name
    return id_map


# ---------------------------------------------------------------------------
# Pool-based pre-spawn (fast rendering path)
# ---------------------------------------------------------------------------

def build_target_pool(registry, cfg: dict) -> dict:
    """Pre-load all target models once; assign fixed pass_index; hide all.

    Returns {model_stem: [list of bpy obj refs]} with pass_index 1..N assigned
    permanently. Call once before the render loop.
    """
    per_class_max = cfg["scene"].get("per_class_max", 2)
    pool = {}
    next_id = 1

    for asset in registry._pools["models"]:
        stem = asset.path.stem
        pool[stem] = []
        for _ in range(per_class_max):
            objs = import_model(asset.path, collection_name="Randomize")
            for obj in objs:
                obj.pass_index = next_id
                obj["inst_id"] = next_id
                obj["category_id"] = asset.category_id
                obj["category_name"] = asset.category_name
                obj.hide_render = True
                obj.hide_viewport = True
                next_id += 1
            pool[stem].extend(objs)

    col = bpy.data.collections.get("Randomize")
    if col:
        col.hide_render = False

    return pool


def build_distractor_pool(registry, cfg: dict) -> list:
    """Pre-load all distractor mesh assets and primitives once; hide all.

    Returns a flat list of all distractor bpy obj refs.
    Call once before the render loop.
    """
    scene_cfg = cfg["scene"]
    distractors_max = scene_cfg.get("distractors_max", 6)
    prim_size = float(scene_cfg.get("distractor_primitive_size", 0.2))
    half = prim_size / 2.0

    pool = []

    # Load each distractor mesh file once
    for asset in registry._pools.get("distractors", []):
        objs = import_model(asset.path, collection_name="Distractors")
        for obj in objs:
            obj["is_distractor"] = True
            obj.hide_render = True
            obj.hide_viewport = True
        pool.extend(objs)

    # Pre-spawn distractors_max primitives (deterministic type assignment)
    _PRIMITIVE_OPS = [
        lambda: bpy.ops.mesh.primitive_cube_add(size=prim_size),
        lambda: bpy.ops.mesh.primitive_uv_sphere_add(radius=half),
        lambda: bpy.ops.mesh.primitive_cylinder_add(radius=half, depth=prim_size),
        lambda: bpy.ops.mesh.primitive_cone_add(radius1=half, depth=prim_size),
        lambda: bpy.ops.mesh.primitive_torus_add(major_radius=half, minor_radius=half * 0.3),
    ]
    occluders_col = bpy.data.collections.get("Occluders")
    pool_rng = __import__("numpy").random.default_rng(0)
    for _ in range(distractors_max):
        before = set(bpy.data.objects)
        _PRIMITIVE_OPS[int(pool_rng.integers(0, len(_PRIMITIVE_OPS)))]()
        after = set(bpy.data.objects)
        new_objs = [o for o in (after - before) if o.type == "MESH"]
        for obj in new_objs:
            obj["is_distractor"] = True
            obj.hide_render = True
            obj.hide_viewport = True
            if occluders_col is not None:
                for src_col in list(obj.users_collection):
                    src_col.objects.unlink(obj)
                occluders_col.objects.link(obj)
        pool.extend(new_objs)

    return pool


def activate_frame_objects(target_pool: dict, distractor_pool: list,
                            rng, cfg: dict) -> tuple:
    """Hide all pooled objects, then activate a random subset for this frame.

    Returns (active_targets, active_distractors, id_map) where
    id_map = {pass_index: obj.name} for active targets only.
    """
    scene_cfg = cfg["scene"]
    per_class_min = scene_cfg.get("per_class_min", 1)
    per_class_max = scene_cfg.get("per_class_max", 2)
    distractors_min = scene_cfg.get("distractors_min", 0)
    distractors_max = scene_cfg.get("distractors_max", 6)

    # Step 1: hide everything
    for objs in target_pool.values():
        for obj in objs:
            obj.hide_render = True
            obj.hide_viewport = True
    for obj in distractor_pool:
        obj.hide_render = True
        obj.hide_viewport = True

    # Step 2: activate targets
    active_targets = []
    class_appear_prob = float(scene_cfg.get("class_appearance_probability", 1.0))
    for stem, objs in target_pool.items():
        if float(rng.random()) > class_appear_prob:
            continue
        count = int(rng.integers(per_class_min, per_class_max + 1))
        for obj in objs[:count]:
            obj.hide_render = False
            obj.hide_viewport = False
            active_targets.append(obj)

    # Step 3: activate distractors
    n = int(rng.integers(distractors_min, distractors_max + 1))
    shuffled = list(distractor_pool)
    rng.shuffle(shuffled)
    active_distractors = []
    for obj in shuffled[:n]:
        obj.hide_render = False
        obj.hide_viewport = False
        active_distractors.append(obj)

    # Step 4: build id_map from active targets (permanent pass_index values)
    id_map = {obj.pass_index: obj.name for obj in active_targets}

    return active_targets, active_distractors, id_map


def activate_frame_objects_with_groups(
    target_pool: dict,
    distractor_pool: list,
    rng,
    cfg: dict,
    frame_idx: int = 0,
) -> tuple:
    """Hide all pooled objects, then activate a random subset with group support.

    Returns (solo_targets, active_groups, active_distractors, id_map) where:
      solo_targets  — list of ungrouped bpy.Object
      active_groups — list of GroupArrangement
      active_distractors — list of bpy.Object
      id_map        — {pass_index: obj.name} flat dict covering all targets
    """
    scene_cfg = cfg["scene"]
    per_class_min = scene_cfg.get("per_class_min", 1)
    per_class_max = scene_cfg.get("per_class_max", 2)
    distractors_min = scene_cfg.get("distractors_min", 0)
    distractors_max = scene_cfg.get("distractors_max", 6)

    arr_cfg = scene_cfg.get("arrangements", {})
    arr_enabled = arr_cfg.get("enabled", False)
    group_prob = float(arr_cfg.get("group_probability", 0.5))
    raw_grids = arr_cfg.get("grid_sizes", [[1, 1, 1]])
    cell_gap = float(arr_cfg.get("cell_spacing", 0.0))  # extra gap beyond flush contact

    # Parse grid_sizes into (rows, cols, weight) tuples
    grid_defs = []
    for entry in raw_grids:
        r, c, w = int(entry[0]), int(entry[1]), float(entry[2])
        grid_defs.append((r, c, w))
    total_weight = sum(w for _, _, w in grid_defs)

    # Step 1: hide everything
    for objs in target_pool.values():
        for obj in objs:
            obj.hide_render = True
            obj.hide_viewport = True
    for obj in distractor_pool:
        obj.hide_render = True
        obj.hide_viewport = True

    # Step 2: activate targets
    solo_targets: list = []
    active_groups: list = []
    all_group_members: list = []

    class_appear_prob = float(scene_cfg.get("class_appearance_probability", 1.0))
    for stem, objs in target_pool.items():
        if float(rng.random()) > class_appear_prob:
            continue
        available = list(objs)  # all hidden; we pick from them in order

        if arr_enabled and float(rng.random()) < group_prob:
            # Choose grid size by weight
            spin = float(rng.random()) * total_weight
            rows, cols, _ = grid_defs[-1]
            cumulative = 0.0
            for r, c, w in grid_defs:
                cumulative += w
                if spin <= cumulative:
                    rows, cols = r, c
                    break

            needed = rows * cols

            # Fallback: downgrade grid until it fits the pool
            while needed > len(available) and (rows > 1 or cols > 1):
                rows = max(1, rows - 1)
                cols = max(1, cols - 1)
                needed = rows * cols
                logging.warning(
                    f"arrangements: pool for '{stem}' has only {len(available)} objects; "
                    f"downgraded grid to {rows}x{cols}"
                )

            chosen = available[:needed]
            for obj in chosen:
                obj.hide_render = False
                obj.hide_viewport = False

            pivot = create_group_pivot(f"GroupPivot_{stem}_{frame_idx}", "Randomize")
            for obj in chosen:
                parent_to_pivot(obj, pivot)
            # Derive flush spacing from the first member's bounding box dimensions.
            # obj.dimensions = bbox size × scale; available without view_layer.update().
            dim = chosen[0].dimensions
            spacing_x = dim.x + cell_gap
            spacing_y = dim.y + cell_gap
            layout_group_local(chosen, rows, cols, spacing_x, spacing_y)

            active_groups.append(GroupArrangement(
                pivot=pivot, members=chosen, rows=rows, cols=cols, stem=stem
            ))
            all_group_members.extend(chosen)
        else:
            # Solo path: activate per_class_min..per_class_max objects
            count = int(rng.integers(per_class_min, per_class_max + 1))
            for obj in available[:count]:
                obj.hide_render = False
                obj.hide_viewport = False
                solo_targets.append(obj)

    # Step 3: activate distractors (unchanged)
    n = int(rng.integers(distractors_min, distractors_max + 1))
    shuffled = list(distractor_pool)
    rng.shuffle(shuffled)
    active_distractors = []
    for obj in shuffled[:n]:
        obj.hide_render = False
        obj.hide_viewport = False
        active_distractors.append(obj)

    # Step 4: flat id_map covering all active targets
    all_targets = solo_targets + all_group_members
    id_map = {obj.pass_index: obj.name for obj in all_targets}

    return solo_targets, active_groups, active_distractors, id_map
