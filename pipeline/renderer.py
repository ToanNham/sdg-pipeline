"""Phase 4 – renderer.py

Configures Blender Cycles, enables the Object Index pass, wires the compositor
to emit an RGB PNG and per-instance binary mask PNGs (via ID_MASK nodes) per
frame, and triggers the render.

All bpy access is deferred to function bodies; no module-level bpy calls.
"""

from pathlib import Path


def configure_cycles(scene, cfg) -> None:
    """Configure Cycles renderer from config dict.

    Args:
        scene: bpy.context.scene
        cfg:   full config dict; reads cfg["render"]
    """
    import bpy  # noqa: F401 – imported for side-effects on scene attrs

    r = cfg["render"]
    scene.render.engine = "CYCLES"
    scene.cycles.device = r["device"]
    scene.cycles.samples = r["samples"]
    scene.render.resolution_x = r["resolution_x"]
    scene.render.resolution_y = r["resolution_y"]

    if r.get("use_denoiser"):
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = r["denoiser"]
    else:
        scene.cycles.use_denoising = False


def enable_object_index_pass(view_layer) -> None:
    """Enable the Object Index (IndexOB) render pass on the given view layer.

    This is required for ID_MASK compositor nodes to read per-object indices.
    Each target object must also have obj.pass_index set to a unique integer
    (done in assign_instance_ids).

    Args:
        view_layer: bpy.context.scene.view_layers[0]
    """
    view_layer.use_pass_object_index = True


def setup_compositor(scene, img_idx: int, output_dir: Path, id_map: dict) -> None:
    """Configure compositor output nodes for one frame.

    Creates:
      - One File Output node for the RGB PNG image
      - One ID_MASK node + File Output slot per target object (binary mask PNGs)

    All outputs are written by the compositor when bpy.ops.render.render() runs.

    Args:
        scene:      bpy.context.scene
        img_idx:    0-based image index; zero-padded to 4 digits
        output_dir: root output directory (Path); must be absolute
        id_map:     {inst_id: obj_name} from assign_instance_ids()

    Output files (Blender appends frame number suffix "_0001" automatically):
        output_dir/images/{img_idx:04d}_0001.png
        output_dir/masks_instance/{img_idx:04d}_inst_{K}_0001.png  (one per inst_id K)
    """
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")

    # RGB PNG output
    png_out = tree.nodes.new("CompositorNodeOutputFile")
    png_out.base_path = str(output_dir / "images")
    png_out.file_slots[0].path = f"{img_idx:04d}_"
    png_out.format.file_format = "PNG"
    png_out.format.color_mode = "RGB"
    tree.links.new(rl.outputs["Image"], png_out.inputs[0])

    if not id_map:
        return

    # Per-object binary mask PNGs via ID_MASK nodes
    mask_out = tree.nodes.new("CompositorNodeOutputFile")
    mask_out.base_path = str(output_dir / "masks_instance")
    mask_out.format.file_format = "PNG"
    mask_out.format.color_mode = "BW"
    # Remove the default slot; we add one per object below
    mask_out.file_slots.clear()

    for inst_id in sorted(id_map.keys()):
        id_mask = tree.nodes.new("CompositorNodeIDMask")
        id_mask.index = inst_id
        id_mask.use_antialiasing = False
        tree.links.new(rl.outputs["IndexOB"], id_mask.inputs["ID value"])
        slot = mask_out.file_slots.new(f"{img_idx:04d}_inst_{inst_id}_")
        tree.links.new(id_mask.outputs["Alpha"], mask_out.inputs[slot.name])


def render(scene) -> None:
    """Trigger a single render pass.

    The compositor runs as a side-effect and writes the RGB PNG and all
    instance mask PNGs.

    Args:
        scene: bpy.context.scene (used implicitly by bpy.ops)
    """
    import bpy
    bpy.ops.render.render()
