"""Phase 4 – renderer.py

Configures Blender Cycles, enables Cryptomatte object passes, wires the
compositor to emit both an RGB PNG and a multilayer EXR per frame, and
triggers the render.

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


def enable_cryptomatte(view_layer) -> None:
    """Enable Cryptomatte Object pass on the given view layer.

    Sets depth=2, which encodes up to 4 objects per pixel across two
    CryptoObjectNN layers (each layer stores 2 (hash, weight) pairs).

    Args:
        view_layer: bpy.context.scene.view_layers[0]
    """
    view_layer.use_pass_cryptomatte_object = True
    view_layer.pass_cryptomatte_depth = 2


def setup_compositor(scene, img_idx: int, output_dir: Path) -> None:
    """Wire compositor to write RGB PNG + multilayer EXR for one frame.

    Clears existing compositor nodes and rebuilds from scratch each call
    so that the output file paths update correctly per image index.

    Args:
        scene:      bpy.context.scene
        img_idx:    0-based image index; zero-padded to 4 digits
        output_dir: root output directory (Path)

    Output files (Blender appends frame suffix "_0001" automatically):
        output_dir/images/{img_idx:04d}_0001.png
        output_dir/tmp/{img_idx:04d}_0001.exr
    """
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    # Render Layers source node
    rl = tree.nodes.new("CompositorNodeRLayers")

    # --- RGB PNG output ---
    png_out = tree.nodes.new("CompositorNodeOutputFile")
    png_out.base_path = str(output_dir / "images")
    png_out.file_slots[0].path = f"{img_idx:04d}_"
    png_out.format.file_format = "PNG"
    png_out.format.color_mode = "RGB"
    tree.links.new(rl.outputs["Image"], png_out.inputs[0])

    # --- Multilayer EXR output (captures all passes incl. Cryptomatte) ---
    exr_out = tree.nodes.new("CompositorNodeOutputFile")
    exr_out.base_path = str(output_dir / "tmp")
    exr_out.file_slots[0].path = f"{img_idx:04d}_"
    exr_out.format.file_format = "OPEN_EXR_MULTILAYER"
    exr_out.format.color_depth = "32"
    tree.links.new(rl.outputs["Image"], exr_out.inputs[0])


def render(scene) -> None:
    """Trigger a single render pass.

    Compositor output nodes write files to disk as a side-effect.
    write_still=False because the compositor handles file writing.

    Args:
        scene: bpy.context.scene (used implicitly by bpy.ops)
    """
    import bpy
    bpy.ops.render.render(write_still=False)
