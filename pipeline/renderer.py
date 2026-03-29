"""Phase 4 – renderer.py

Configures Blender Cycles, enables the Object Index pass, wires the compositor
to emit an RGB PNG and per-instance binary mask PNGs (via ID_MASK nodes) per
frame, and triggers the render.

All bpy access is deferred to function bodies; no module-level bpy calls.
"""

from pathlib import Path


def _enable_gpu_devices(compute_device_type: str) -> None:
    """Enable GPU devices in Blender Cycles preferences.

    Must be called before scene.cycles.device = "GPU" takes effect.
    Without this Blender silently falls back to CPU.

    Args:
        compute_device_type: "CUDA", "OPTIX", "HIP", or "METAL"
    """
    import bpy
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = compute_device_type
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True


def configure_cycles(scene, cfg) -> None:
    """Configure Cycles renderer from config dict.

    Args:
        scene: bpy.context.scene
        cfg:   full config dict; reads cfg["render"]
    """
    import bpy  # noqa: F401 – imported for side-effects on scene attrs

    r = cfg["render"]
    scene.render.engine = "CYCLES"
    if r["device"] == "GPU":
        _enable_gpu_devices(r.get("compute_device_type", "CUDA"))
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


def setup_compositor(scene, img_idx: int, output_dir: Path, id_map: dict,
                     inst_colors: dict) -> None:
    """Configure compositor output nodes for one frame.

    Creates:
      - One File Output node for the RGB PNG image
      - One File Output node for the color instance mask PNG (built entirely
        in the compositor via ID_MASK → Multiply → Add chain; no Python file I/O)

    All outputs are written by the compositor when bpy.ops.render.render() runs.

    Args:
        scene:       bpy.context.scene
        img_idx:     0-based image index; zero-padded to 4 digits
        output_dir:  root output directory (Path); must be absolute
        id_map:      {inst_id: obj_name} from assign_instance_ids()
        inst_colors: {inst_id: (R, G, B)} uint8 colors from assign_instance_colors()

    Output files (Blender appends frame number suffix "_0001" automatically):
        output_dir/images/{img_idx:04d}_0001.png
        output_dir/masks/{img_idx:04d}_color_0001.png
    """
    scene.use_nodes = True
    scene.render.use_compositing = True  # compositor must execute during render
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

    # Color instance mask — built in compositor, no Python file I/O.
    # For each instance: ID_MASK (0/1 float) × instance color → ADD all together.
    accum = None
    for inst_id in sorted(id_map.keys()):
        r, g, b = [c / 255.0 for c in inst_colors[inst_id]]

        id_mask = tree.nodes.new("CompositorNodeIDMask")
        id_mask.index = inst_id
        id_mask.use_antialiasing = False
        tree.links.new(rl.outputs["IndexOB"], id_mask.inputs["ID value"])

        # Multiply mask (0 or 1) by instance color
        mul = tree.nodes.new("CompositorNodeMixRGB")
        mul.blend_type = 'MULTIPLY'
        mul.inputs[0].default_value = 1.0
        mul.inputs[2].default_value = (r, g, b, 1.0)
        tree.links.new(id_mask.outputs["Alpha"], mul.inputs[1])

        if accum is None:
            accum = mul
        else:
            add = tree.nodes.new("CompositorNodeMixRGB")
            add.blend_type = 'ADD'
            add.inputs[0].default_value = 1.0
            tree.links.new(accum.outputs[0], add.inputs[1])
            tree.links.new(mul.outputs[0], add.inputs[2])
            accum = add

    color_out = tree.nodes.new("CompositorNodeOutputFile")
    color_out.base_path = str(output_dir / "masks")
    color_out.format.file_format = "PNG"
    color_out.format.color_mode = "RGB"
    color_out.file_slots[0].path = f"{img_idx:04d}_color_"
    tree.links.new(accum.outputs[0], color_out.inputs[0])


def render(scene) -> None:
    """Trigger a single render pass.

    The compositor runs as a side-effect and writes the RGB PNG and all
    instance mask PNGs.

    Args:
        scene: bpy.context.scene (used implicitly by bpy.ops)
    """
    import bpy
    scene.frame_current = 1  # ensures compositor appends _0001 suffix
    bpy.ops.render.render()
