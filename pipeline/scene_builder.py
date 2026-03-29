from pathlib import Path

import bpy  # module-level import is fine; do NOT call bpy.* at module level


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
    _PRIMITIVE_OPS = [
        bpy.ops.mesh.primitive_cube_add,
        bpy.ops.mesh.primitive_uv_sphere_add,
        bpy.ops.mesh.primitive_cylinder_add,
        bpy.ops.mesh.primitive_cone_add,
        bpy.ops.mesh.primitive_torus_add,
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
