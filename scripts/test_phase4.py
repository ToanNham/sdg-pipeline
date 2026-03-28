"""Comprehensive tests for Phase 4: renderer.py and mask_extractor.py.

Runs without Blender by:
- Mocking bpy for renderer.py tests
- Building synthetic Cryptomatte EXR files for mask_extractor.py tests
"""

import struct
import sys
import types
from pathlib import Path

import numpy as np
import OpenEXR

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results = []


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    _results.append(cond)


# ---------------------------------------------------------------------------
# 1. mm2_hash_str
# ---------------------------------------------------------------------------

def test_mm2_hash():
    from pipeline.mask_extractor import mm2_hash_str

    print("\n=== mm2_hash_str ===")

    # Output is a Python float (IEEE 754 float32 reinterpreted as Python float)
    h = mm2_hash_str("Cube")
    check("returns float", isinstance(h, float))

    # Deterministic: same name → same hash
    check("deterministic", mm2_hash_str("Cube") == mm2_hash_str("Cube"))

    # Different names → different hashes (collision would be very surprising)
    names = ["Cube", "Sphere", "Camera", "Cube.001", "Cube.002", "Torus", ""]
    hashes = [mm2_hash_str(n) for n in names]
    check("all unique", len(set(hashes)) == len(hashes),
          f"got {len(set(hashes))} unique out of {len(hashes)}")

    # Empty string must not crash
    h_empty = mm2_hash_str("")
    check("empty string ok", isinstance(h_empty, float))

    # Unicode name (Blender allows unicode object names)
    h_uni = mm2_hash_str("Würfel")
    check("unicode ok", isinstance(h_uni, float))

    # Hash is stored as float32; round-tripping through float32 must be stable
    for name in ["Cube", "Sphere", "Light"]:
        h_f64 = mm2_hash_str(name)
        h_f32 = np.float32(h_f64)
        check(f"float32 stable '{name}'", h_f64 == float(h_f32))

    # Known reference value: verify algorithm hasn't drifted
    # Compute expected manually with the same algorithm
    def ref_mm2(name):
        data = name.encode("utf-8")
        m = 0xC6A4A793
        h = (len(data) * m) & 0xFFFFFFFF
        off = 0
        while off + 4 <= len(data):
            k = struct.unpack_from("<I", data, off)[0]
            k = (k * m) & 0xFFFFFFFF
            k ^= k >> 16
            k = (k * m) & 0xFFFFFFFF
            h ^= k
            h = (h * m) & 0xFFFFFFFF
            off += 4
        rem = len(data) - off
        if rem >= 3: h ^= data[off + 2] << 16
        if rem >= 2: h ^= data[off + 1] << 8
        if rem >= 1:
            h ^= data[off]
            h = (h * m) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * m) & 0xFFFFFFFF
        h ^= h >> 15
        h &= 0xFFFFFFFF
        return struct.unpack("f", struct.pack("I", h))[0]

    for name in ["Cube", "Sphere", "Camera", "Cube.001"]:
        check(f"matches reference '{name}'", mm2_hash_str(name) == ref_mm2(name))


# ---------------------------------------------------------------------------
# 2. build_semantic_mask
# ---------------------------------------------------------------------------

def test_build_semantic_mask():
    from pipeline.mask_extractor import build_semantic_mask

    print("\n=== build_semantic_mask ===")

    H, W = 8, 10

    # Single object covers top half
    mask1 = np.zeros((H, W), dtype=np.uint8)
    mask1[:4, :] = 255
    result = build_semantic_mask({1: mask1}, {1: 2}, H, W)
    check("shape correct", result.shape == (H, W))
    check("dtype uint8", result.dtype == np.uint8)
    check("top half = category 2", np.all(result[:4, :] == 2))
    check("bottom half = 0", np.all(result[4:, :] == 0))

    # Two objects; second overwrites first where they overlap
    mask2 = np.zeros((H, W), dtype=np.uint8)
    mask2[3:7, :] = 255   # overlaps rows 3–3 with mask1
    result2 = build_semantic_mask({1: mask1, 2: mask2}, {1: 1, 2: 3}, H, W)
    check("rows 0-2 = cat 1", np.all(result2[:3, :] == 1))
    check("rows 3-6 = cat 3 (obj2 overwrites)", np.all(result2[3:7, :] == 3))
    check("rows 7+ = 0", np.all(result2[7:, :] == 0))

    # Empty instance_masks → all zeros
    result3 = build_semantic_mask({}, {}, H, W)
    check("empty: all zeros", np.all(result3 == 0))

    # Unknown inst_id falls back to category 0
    result4 = build_semantic_mask({99: mask1}, {}, H, W)
    check("missing category_id: 0", np.all(result4 == 0))


# ---------------------------------------------------------------------------
# 3. extract_instance_masks (using synthetic EXR)
# ---------------------------------------------------------------------------

def _make_cryptomatte_exr(path: Path, H: int, W: int, objects: dict) -> None:
    """Write a synthetic multilayer EXR with Cryptomatte channels.

    objects: {inst_id: (obj_name, mask_bool_array)}
    Encodes up to 4 objects across CryptoObject00 and CryptoObject01 layers.
    Each object gets full coverage (weight=1.0) on its mask pixels.
    """
    from pipeline.mask_extractor import mm2_hash_str

    obj_list = list(objects.items())  # [(inst_id, (name, mask)), ...]

    def make_layer(pair0, pair1):
        """Build R,G,B,A arrays for one CryptoObject layer from two object entries."""
        id_r = np.zeros((H, W), dtype=np.float32)
        wt_g = np.zeros((H, W), dtype=np.float32)
        id_b = np.zeros((H, W), dtype=np.float32)
        wt_a = np.zeros((H, W), dtype=np.float32)
        if pair0 is not None:
            _, (name0, mask0) = pair0
            h0 = mm2_hash_str(name0)
            id_r[mask0] = h0
            wt_g[mask0] = 1.0
        if pair1 is not None:
            _, (name1, mask1) = pair1
            h1 = mm2_hash_str(name1)
            id_b[mask1] = h1
            wt_a[mask1] = 1.0
        return id_r, wt_g, id_b, wt_a

    p0 = obj_list[0] if len(obj_list) > 0 else None
    p1 = obj_list[1] if len(obj_list) > 1 else None
    p2 = obj_list[2] if len(obj_list) > 2 else None
    p3 = obj_list[3] if len(obj_list) > 3 else None

    r0, g0, b0, a0 = make_layer(p0, p1)
    r1, g1, b1, a1 = make_layer(p2, p3)

    channels = {
        "ViewLayer.CryptoObject00.R": OpenEXR.Channel(r0),
        "ViewLayer.CryptoObject00.G": OpenEXR.Channel(g0),
        "ViewLayer.CryptoObject00.B": OpenEXR.Channel(b0),
        "ViewLayer.CryptoObject00.A": OpenEXR.Channel(a0),
        "ViewLayer.CryptoObject01.R": OpenEXR.Channel(r1),
        "ViewLayer.CryptoObject01.G": OpenEXR.Channel(g1),
        "ViewLayer.CryptoObject01.B": OpenEXR.Channel(b1),
        "ViewLayer.CryptoObject01.A": OpenEXR.Channel(a1),
    }
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    OpenEXR.File(header, channels).write(str(path))


def test_extract_instance_masks():
    from pipeline.mask_extractor import extract_instance_masks

    print("\n=== extract_instance_masks ===")

    tmp = Path("output/tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    H, W = 16, 20

    # ── Test 1: single object, covers left half ───────────────────────────
    mask_left = np.zeros((H, W), dtype=bool)
    mask_left[:, :W // 2] = True

    exr1 = tmp / "test_single.exr"
    _make_cryptomatte_exr(exr1, H, W, {1: ("Cube", mask_left)})
    result = extract_instance_masks(exr1, {1: "Cube"})

    check("single: inst_id 1 present", 1 in result)
    if 1 in result:
        check("single: shape", result[1].shape == (H, W))
        check("single: dtype uint8", result[1].dtype == np.uint8)
        check("single: left half = 255", np.all(result[1][:, :W // 2] == 255))
        check("single: right half = 0", np.all(result[1][:, W // 2:] == 0))

    # ── Test 2: two objects in CryptoObject00 (both < 2 objs) ────────────
    mask_top = np.zeros((H, W), dtype=bool)
    mask_top[:H // 2, :] = True
    mask_bot = np.zeros((H, W), dtype=bool)
    mask_bot[H // 2:, :] = True

    exr2 = tmp / "test_two.exr"
    _make_cryptomatte_exr(exr2, H, W, {1: ("Cube", mask_top), 2: ("Sphere", mask_bot)})
    id_map2 = {1: "Cube", 2: "Sphere"}
    result2 = extract_instance_masks(exr2, id_map2)

    check("two: both present", 1 in result2 and 2 in result2)
    if 1 in result2 and 2 in result2:
        check("two: obj1 top half = 255", np.all(result2[1][:H // 2, :] == 255))
        check("two: obj1 bottom half = 0", np.all(result2[1][H // 2:, :] == 0))
        check("two: obj2 bottom half = 255", np.all(result2[2][H // 2:, :] == 255))
        check("two: obj2 top half = 0", np.all(result2[2][:H // 2, :] == 0))

    # ── Test 3: four objects across both layers ───────────────────────────
    q = H // 2
    p = W // 2
    masks_4 = {
        1: ("Cube",     np.s_[:q, :p]),
        2: ("Sphere",   np.s_[:q, p:]),
        3: ("Torus",    np.s_[q:, :p]),
        4: ("Cylinder", np.s_[q:, p:]),
    }
    objects_4 = {}
    for iid, (name, slc) in masks_4.items():
        m = np.zeros((H, W), dtype=bool)
        m[slc] = True
        objects_4[iid] = (name, m)

    exr4 = tmp / "test_four.exr"
    _make_cryptomatte_exr(exr4, H, W, objects_4)
    id_map4 = {iid: name for iid, (name, _) in masks_4.items()}
    result4 = extract_instance_masks(exr4, id_map4)

    check("four: all 4 present", all(i in result4 for i in range(1, 5)))
    for iid, (name, slc) in masks_4.items():
        if iid in result4:
            expected = np.zeros((H, W), dtype=np.uint8)
            expected[slc] = 255
            check(f"four: obj{iid} ({name}) mask correct",
                  np.array_equal(result4[iid], expected))

    # ── Test 4: object with zero coverage is absent from result ──────────
    exr5 = tmp / "test_absent.exr"
    _make_cryptomatte_exr(exr5, H, W, {1: ("Cube", mask_left)})
    # id_map includes inst_id=2 ("Ghost") which has no pixels
    result5 = extract_instance_masks(exr5, {1: "Cube", 2: "Ghost"})
    check("absent: visible object present", 1 in result5)
    check("absent: zero-coverage object omitted", 2 not in result5)

    # ── Test 5: unknown name maps to zeros (no crash) ─────────────────────
    exr6 = tmp / "test_unknown.exr"
    _make_cryptomatte_exr(exr6, H, W, {1: ("Cube", mask_left)})
    result6 = extract_instance_masks(exr6, {1: "NotCube"})
    check("unknown name: no crash", True)
    check("unknown name: absent (hash mismatch)", 1 not in result6)


# ---------------------------------------------------------------------------
# 4. renderer.py (logic-only, bpy mocked)
# ---------------------------------------------------------------------------

def _make_bpy_mock():
    """Build a minimal mock of the bpy module sufficient for renderer.py logic."""

    bpy_mod = types.ModuleType("bpy")

    # --- bpy.ops.render ---
    ops = types.SimpleNamespace()
    render_ns = types.SimpleNamespace()
    render_ns.render = lambda write_still=False: None
    ops.render = render_ns
    bpy_mod.ops = ops

    # --- bpy.context (not used directly in renderer but needed for import) ---
    bpy_mod.context = types.SimpleNamespace()

    sys.modules["bpy"] = bpy_mod
    return bpy_mod


def _make_scene_mock(cfg):
    """Build a minimal scene mock for configure_cycles / setup_compositor."""

    # Cycles settings
    cycles_ns = types.SimpleNamespace(
        device=None, samples=None, use_denoising=None, denoiser=None
    )

    # Render settings
    render_ns = types.SimpleNamespace(
        engine=None, resolution_x=None, resolution_y=None
    )

    # Compositor node tree
    nodes_store = {}
    links_store = []

    class NodeTree:
        def __init__(self):
            self.nodes = self
            self.links = self
            self._nodes = []
            self._links = []

        def clear(self):
            self._nodes.clear()

        def new(self, node_type):
            node = types.SimpleNamespace(
                type=node_type,
                base_path=None,
                format=types.SimpleNamespace(
                    file_format=None, color_mode=None, color_depth=None
                ),
                file_slots=[types.SimpleNamespace(path=None)],
                outputs={"Image": types.SimpleNamespace(_connected=[])},
                inputs=[types.SimpleNamespace(_target=None)],
            )
            self._nodes.append(node)
            return node

        def link_new(self, src, dst):
            self._links.append((src, dst))

    node_tree = NodeTree()

    # Patch links.new on the tree
    node_tree.links = types.SimpleNamespace(
        _links=[],
        new=lambda src, dst: node_tree.links._links.append((src, dst)),
    )

    scene = types.SimpleNamespace(
        render=render_ns,
        cycles=cycles_ns,
        use_nodes=False,
        node_tree=node_tree,
    )
    return scene


def test_renderer_configure_cycles():
    from pipeline import renderer as _r_import
    import importlib

    print("\n=== configure_cycles ===")

    _make_bpy_mock()
    import importlib
    # Force re-import so bpy mock is picked up inside the function
    if "pipeline.renderer" in sys.modules:
        del sys.modules["pipeline.renderer"]
    from pipeline.renderer import configure_cycles

    cfg_gpu = {
        "render": {
            "engine": "CYCLES",
            "device": "GPU",
            "samples": 128,
            "resolution_x": 1920,
            "resolution_y": 1080,
            "use_denoiser": True,
            "denoiser": "OPENIMAGEDENOISE",
        }
    }
    scene = _make_scene_mock(cfg_gpu)
    configure_cycles(scene, cfg_gpu)

    check("engine = CYCLES", scene.render.engine == "CYCLES")
    check("device = GPU", scene.cycles.device == "GPU")
    check("samples = 128", scene.cycles.samples == 128)
    check("resolution_x = 1920", scene.render.resolution_x == 1920)
    check("resolution_y = 1080", scene.render.resolution_y == 1080)
    check("denoising enabled", scene.cycles.use_denoising is True)
    check("denoiser = OPENIMAGEDENOISE", scene.cycles.denoiser == "OPENIMAGEDENOISE")

    # CPU, no denoiser
    cfg_cpu = {
        "render": {
            "engine": "CYCLES",
            "device": "CPU",
            "samples": 32,
            "resolution_x": 640,
            "resolution_y": 480,
            "use_denoiser": False,
        }
    }
    scene2 = _make_scene_mock(cfg_cpu)
    configure_cycles(scene2, cfg_cpu)

    check("CPU: device = CPU", scene2.cycles.device == "CPU")
    check("CPU: denoising disabled", scene2.cycles.use_denoising is False)


def test_renderer_enable_cryptomatte():
    print("\n=== enable_cryptomatte ===")

    if "pipeline.renderer" in sys.modules:
        del sys.modules["pipeline.renderer"]
    from pipeline.renderer import enable_cryptomatte

    view_layer = types.SimpleNamespace(
        use_pass_cryptomatte_object=False,
        pass_cryptomatte_depth=0,
    )
    enable_cryptomatte(view_layer)

    check("use_pass_cryptomatte_object = True",
          view_layer.use_pass_cryptomatte_object is True)
    check("pass_cryptomatte_depth = 2", view_layer.pass_cryptomatte_depth == 2)


def test_renderer_setup_compositor():
    print("\n=== setup_compositor ===")

    if "pipeline.renderer" in sys.modules:
        del sys.modules["pipeline.renderer"]
    from pipeline.renderer import setup_compositor

    cfg = {"render": {"device": "CPU", "samples": 4,
                      "resolution_x": 640, "resolution_y": 480,
                      "use_denoiser": False}}
    scene = _make_scene_mock(cfg)
    out_dir = Path("output")

    setup_compositor(scene, img_idx=7, output_dir=out_dir)

    check("use_nodes = True", scene.use_nodes is True)

    nodes = scene.node_tree._nodes
    node_types = [n.type for n in nodes]
    check("RLayers node created", "CompositorNodeRLayers" in node_types)
    check("two OutputFile nodes", node_types.count("CompositorNodeOutputFile") == 2)

    out_nodes = [n for n in nodes if n.type == "CompositorNodeOutputFile"]

    # PNG node
    png = next((n for n in out_nodes if n.format.file_format == "PNG"), None)
    check("PNG node present", png is not None)
    if png:
        check("PNG base_path = output/images",
              png.base_path == str(out_dir / "images"))
        check("PNG slot path = '0007_'", png.file_slots[0].path == "0007_")
        check("PNG color_mode = RGB", png.format.color_mode == "RGB")

    # EXR node
    exr = next((n for n in out_nodes if n.format.file_format == "OPEN_EXR_MULTILAYER"), None)
    check("EXR node present", exr is not None)
    if exr:
        check("EXR base_path = output/tmp",
              exr.base_path == str(out_dir / "tmp"))
        check("EXR slot path = '0007_'", exr.file_slots[0].path == "0007_")
        check("EXR color_depth = 32", exr.format.color_depth == "32")

    # Two compositor links were created
    links = scene.node_tree.links._links
    check("two compositor links", len(links) == 2)

    # Index rolls over correctly at 0
    scene2 = _make_scene_mock(cfg)
    setup_compositor(scene2, img_idx=0, output_dir=out_dir)
    out2 = [n for n in scene2.node_tree._nodes if n.type == "CompositorNodeOutputFile"]
    check("img_idx=0 slot = '0000_'", out2[0].file_slots[0].path == "0000_")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_mm2_hash()
    test_build_semantic_mask()
    test_extract_instance_masks()
    test_renderer_configure_cycles()
    test_renderer_enable_cryptomatte()
    test_renderer_setup_compositor()

    passed = sum(_results)
    total = len(_results)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        sys.exit(1)
    else:
        print(" — all OK")
