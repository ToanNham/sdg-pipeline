"""Comprehensive tests for Phase 5: annotation_writer.py

Runs without Blender by exercising all six functions in annotation_writer.py
plus an end-to-end data-flow integration test that simulates the per-image
loop in run.py (mask extraction -> annotation_writer -> COCO JSON).
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

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
# 1. save_semantic_mask
# ---------------------------------------------------------------------------

def test_save_semantic_mask():
    from pipeline.annotation_writer import save_semantic_mask
    from PIL import Image

    print("\n=== save_semantic_mask ===")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sem.png"

        mask = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        save_semantic_mask(mask, path)

        check("file created", path.exists())
        img = np.array(Image.open(str(path)))
        check("shape preserved", img.shape == (2, 3))
        check("mode L (grayscale)", Image.open(str(path)).mode == "L")
        check("pixel values exact", np.array_equal(img, mask))

        # All-zero mask (background only)
        path2 = Path(tmp) / "sem_zero.png"
        save_semantic_mask(np.zeros((100, 100), dtype=np.uint8), path2)
        check("zero mask: file created", path2.exists())
        check("zero mask: all zero", np.all(np.array(Image.open(str(path2))) == 0))

        # Full-coverage mask (single category)
        path3 = Path(tmp) / "sem_full.png"
        full = np.full((64, 64), 7, dtype=np.uint8)
        save_semantic_mask(full, path3)
        check("full mask: category 7 everywhere",
              np.all(np.array(Image.open(str(path3))) == 7))


# ---------------------------------------------------------------------------
# 2. save_instance_mask
# ---------------------------------------------------------------------------

def test_save_instance_mask():
    from pipeline.annotation_writer import save_instance_mask
    from PIL import Image

    print("\n=== save_instance_mask ===")

    with tempfile.TemporaryDirectory() as tmp:
        # Binary mask: 255 where object, 0 elsewhere
        path = Path(tmp) / "inst.png"
        binary = np.zeros((50, 60), dtype=np.uint8)
        binary[10:40, 15:45] = 255
        save_instance_mask(binary, path)

        check("file created", path.exists())
        img = np.array(Image.open(str(path)))
        check("mode L", Image.open(str(path)).mode == "L")
        check("object region = 255", np.all(img[10:40, 15:45] == 255))
        check("background = 0", np.all(img[:10, :] == 0))

        # All-background
        path2 = Path(tmp) / "inst_empty.png"
        save_instance_mask(np.zeros((32, 32), dtype=np.uint8), path2)
        check("empty mask: all zero",
              np.all(np.array(Image.open(str(path2))) == 0))

        # Full coverage
        path3 = Path(tmp) / "inst_full.png"
        save_instance_mask(np.full((32, 32), 255, dtype=np.uint8), path3)
        check("full mask: all 255",
              np.all(np.array(Image.open(str(path3))) == 255))


# ---------------------------------------------------------------------------
# 3. compute_bbox
# ---------------------------------------------------------------------------

def test_compute_bbox():
    from pipeline.annotation_writer import compute_bbox

    print("\n=== compute_bbox ===")

    # Rectangular object at known position
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[20:60, 30:80] = 255
    bbox = compute_bbox(mask)
    check("returns list", isinstance(bbox, list))
    check("length 4", len(bbox) == 4)
    x, y, w, h = bbox
    check("x_min = 30", x == 30, f"got {x}")
    check("y_min = 20", y == 20, f"got {y}")
    check("width = 49", w == 49, f"got {w}")   # 79 - 30
    check("height = 39", h == 39, f"got {h}")  # 59 - 20

    # Empty mask -> [0, 0, 0, 0]
    empty = np.zeros((50, 50), dtype=np.uint8)
    check("empty mask -> [0,0,0,0]", compute_bbox(empty) == [0, 0, 0, 0])

    # Single pixel
    single = np.zeros((20, 20), dtype=np.uint8)
    single[10, 15] = 255
    sx, sy, sw, sh = compute_bbox(single)
    check("single pixel: x=15", sx == 15, f"got {sx}")
    check("single pixel: y=10", sy == 10, f"got {sy}")
    check("single pixel: w=0", sw == 0, f"got {sw}")
    check("single pixel: h=0", sh == 0, f"got {sh}")

    # Full-frame mask
    full = np.full((40, 50), 255, dtype=np.uint8)
    fx, fy, fw, fh = compute_bbox(full)
    check("full frame: x=0", fx == 0)
    check("full frame: y=0", fy == 0)
    check("full frame: w=49", fw == 49)
    check("full frame: h=39", fh == 39)

    # Top-left corner only
    corner = np.zeros((30, 30), dtype=np.uint8)
    corner[:5, :5] = 255
    cx, cy, cw, ch = compute_bbox(corner)
    check("corner: x=0", cx == 0)
    check("corner: y=0", cy == 0)
    check("corner: w=4", cw == 4)
    check("corner: h=4", ch == 4)

    # Bottom-right corner only
    br = np.zeros((30, 30), dtype=np.uint8)
    br[25:, 25:] = 255
    brx, bry, brw, brh = compute_bbox(br)
    check("bottom-right: x=25", brx == 25)
    check("bottom-right: y=25", bry == 25)


# ---------------------------------------------------------------------------
# 4. mask_to_polygons
# ---------------------------------------------------------------------------

def test_mask_to_polygons():
    from pipeline.annotation_writer import mask_to_polygons

    print("\n=== mask_to_polygons ===")

    # Solid rectangle -> 1 polygon
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 30:70] = 255
    polys = mask_to_polygons(mask)
    check("rectangle -> at least 1 polygon", len(polys) >= 1)
    check("polygon is a list", isinstance(polys[0], list))
    check("polygon has even number of coords (x,y pairs)",
          len(polys[0]) % 2 == 0)
    check("polygon has at least 3 points (6 coords)", len(polys[0]) >= 6)

    # Polygon coords are floats
    check("coords are numeric",
          all(isinstance(v, (int, float)) for v in polys[0]))

    # Empty mask -> no polygons
    empty = np.zeros((50, 50), dtype=np.uint8)
    check("empty mask -> [] ", mask_to_polygons(empty) == [])

    # Tiny 1-pixel region (too small after simplify -> likely empty)
    tiny = np.zeros((50, 50), dtype=np.uint8)
    tiny[25, 25] = 255
    tiny_polys = mask_to_polygons(tiny)
    check("single pixel: no crash", True)
    # A single pixel won't produce a valid polygon; result may be empty
    check("single pixel: 0 or empty polygon list",
          all(len(p) == 0 for p in tiny_polys) or len(tiny_polys) == 0)

    # Large filled circle-like region -> 1 polygon
    Y, X = np.ogrid[:100, :100]
    circle = ((X - 50) ** 2 + (Y - 50) ** 2 < 30 ** 2).astype(np.uint8) * 255
    c_polys = mask_to_polygons(circle)
    check("circle -> at least 1 polygon", len(c_polys) >= 1)

    # tolerance=0 (no simplification) should still work
    p0 = mask_to_polygons(mask, tolerance=0)
    check("tolerance=0: no crash", isinstance(p0, list))

    # Two disconnected blobs -> 2 polygons
    two = np.zeros((100, 200), dtype=np.uint8)
    two[20:60, 10:50]  = 255   # left blob
    two[20:60, 150:190] = 255  # right blob
    t_polys = mask_to_polygons(two)
    check("two blobs -> 2 polygons", len(t_polys) == 2,
          f"got {len(t_polys)}")


# ---------------------------------------------------------------------------
# 5. init_coco
# ---------------------------------------------------------------------------

def test_init_coco():
    from pipeline.annotation_writer import init_coco

    print("\n=== init_coco ===")

    cfg = {
        "assets": {
            "models": [
                {"category_id": 1, "category_name": "widget"},
                {"category_id": 2, "category_name": "gadget"},
            ]
        }
    }
    coco = init_coco(cfg)

    check("returns dict", isinstance(coco, dict))
    for key in ("info", "licenses", "categories", "images", "annotations"):
        check(f"key '{key}' present", key in coco)

    check("images starts empty", coco["images"] == [])
    check("annotations starts empty", coco["annotations"] == [])
    check("licenses is list", isinstance(coco["licenses"], list))

    cats = coco["categories"]
    check("two categories", len(cats) == 2)
    cat1 = next((c for c in cats if c["id"] == 1), None)
    cat2 = next((c for c in cats if c["id"] == 2), None)
    check("category 1 present", cat1 is not None)
    check("category 2 present", cat2 is not None)
    if cat1:
        check("cat1 name = widget", cat1["name"] == "widget")
        check("cat1 supercategory = object", cat1["supercategory"] == "object")
    if cat2:
        check("cat2 name = gadget", cat2["name"] == "gadget")

    # Single category
    cfg_single = {"assets": {"models": [{"category_id": 5, "category_name": "thing"}]}}
    coco2 = init_coco(cfg_single)
    check("single category: length 1", len(coco2["categories"]) == 1)
    check("single category: id=5", coco2["categories"][0]["id"] == 5)

    # info fields
    check("info.description set", coco["info"].get("description") == "SDG Pipeline")
    check("info.version set", coco["info"].get("version") == "1.0")


# ---------------------------------------------------------------------------
# 6. write_coco
# ---------------------------------------------------------------------------

def test_write_coco():
    from pipeline.annotation_writer import write_coco

    print("\n=== write_coco ===")

    coco = {
        "info": {"description": "test", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "obj", "supercategory": "object"}],
        "images": [{"id": 0, "file_name": "images/0000.png", "width": 640, "height": 480}],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 1,
             "segmentation": [[10.0, 20.0, 30.0, 20.0, 30.0, 40.0, 10.0, 40.0]],
             "area": 400, "bbox": [10, 20, 20, 20], "iscrowd": 0}
        ]
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "annotations" / "instances_0_50.json"

        # Parent dir does not yet exist — write_coco must create it
        check("parent dir missing before write", not path.parent.exists())
        write_coco(coco, path)

        check("file created", path.exists())
        check("parent dir created", path.parent.exists())

        loaded = json.loads(path.read_text())
        check("round-trips as JSON", loaded == coco)
        check("images intact", len(loaded["images"]) == 1)
        check("annotations intact", len(loaded["annotations"]) == 1)

        ann = loaded["annotations"][0]
        check("bbox preserved", ann["bbox"] == [10, 20, 20, 20])
        check("segmentation preserved",
              ann["segmentation"] == [[10.0, 20.0, 30.0, 20.0, 30.0, 40.0, 10.0, 40.0]])

        # Overwrite existing file
        coco2 = dict(coco)
        coco2["images"] = []
        write_coco(coco2, path)
        loaded2 = json.loads(path.read_text())
        check("overwrite: images now empty", loaded2["images"] == [])


# ---------------------------------------------------------------------------
# 7. Integration: simulated per-image pipeline loop
# ---------------------------------------------------------------------------

def test_integration_pipeline_loop():
    """Simulate the run.py per-image loop for 3 images without bpy.

    Uses synthetic instance masks and drives annotation_writer end-to-end,
    then validates the resulting COCO JSON structure.
    """
    from pipeline.annotation_writer import (
        save_semantic_mask, save_instance_mask,
        mask_to_polygons, compute_bbox,
        init_coco, write_coco,
    )

    print("\n=== integration: simulated 3-image loop ===")

    cfg = {
        "assets": {
            "models": [{"category_id": 1, "category_name": "widget"}]
        },
        "render": {"resolution_x": 80, "resolution_y": 60},
    }
    H, W = cfg["render"]["resolution_y"], cfg["render"]["resolution_x"]

    with tempfile.TemporaryDirectory() as tmp_root:
        output_dir = Path(tmp_root) / "output"
        for d in ("images", "masks", "masks_instance", "annotations"):
            (output_dir / d).mkdir(parents=True)

        coco = init_coco(cfg)
        ann_id = 0

        # Three images; each has 1 or 2 visible objects
        scenarios = [
            # (img_idx, {inst_id: mask_slice})
            (0, {1: np.s_[5:40, 10:50]}),
            (1, {1: np.s_[10:30, 5:35], 2: np.s_[30:55, 40:70]}),
            (2, {}),  # no visible objects
        ]

        for img_idx, inst_slices in scenarios:
            # Build synthetic instance masks
            instance_masks = {}
            category_map = {iid: 1 for iid in inst_slices}
            for inst_id, slc in inst_slices.items():
                m = np.zeros((H, W), dtype=np.uint8)
                m[slc] = 255
                instance_masks[inst_id] = m

            # Build semantic mask (pixel = category_id)
            semantic = np.zeros((H, W), dtype=np.uint8)
            for inst_id, mask in instance_masks.items():
                semantic[mask > 0] = category_map.get(inst_id, 0)

            # Save masks
            save_semantic_mask(
                semantic,
                output_dir / "masks" / f"{img_idx:04d}_semantic.png"
            )
            for inst_id, mask in instance_masks.items():
                save_instance_mask(
                    mask,
                    output_dir / "masks_instance" / f"{img_idx:04d}_inst_{inst_id}.png"
                )

            # COCO entry
            coco["images"].append({
                "id":        img_idx,
                "file_name": f"images/{img_idx:04d}_0001.png",
                "width":     W,
                "height":    H,
            })
            for inst_id, mask in instance_masks.items():
                polygons = mask_to_polygons(mask)
                if not polygons:
                    continue
                area = int(np.sum(mask > 0))
                bbox = compute_bbox(mask)
                coco["annotations"].append({
                    "id":           ann_id,
                    "image_id":     img_idx,
                    "category_id":  category_map.get(inst_id, 1),
                    "segmentation": polygons,
                    "area":         area,
                    "bbox":         bbox,
                    "iscrowd":      0,
                })
                ann_id += 1

        # Write COCO JSON
        json_path = output_dir / "annotations" / "instances_0_3.json"
        write_coco(coco, json_path)

        check("JSON file written", json_path.exists())
        data = json.loads(json_path.read_text())

        check("3 image entries", len(data["images"]) == 3,
              f"got {len(data['images'])}")

        # img 0: 1 object -> 1 annotation
        anns_0 = [a for a in data["annotations"] if a["image_id"] == 0]
        check("img0: 1 annotation", len(anns_0) == 1, f"got {len(anns_0)}")

        # img 1: 2 objects -> 2 annotations
        anns_1 = [a for a in data["annotations"] if a["image_id"] == 1]
        check("img1: 2 annotations", len(anns_1) == 2, f"got {len(anns_1)}")

        # img 2: no objects -> 0 annotations
        anns_2 = [a for a in data["annotations"] if a["image_id"] == 2]
        check("img2: 0 annotations", len(anns_2) == 0, f"got {len(anns_2)}")

        # total ann_id is sequential
        all_ids = [a["id"] for a in data["annotations"]]
        check("ann_id sequential from 0",
              all_ids == list(range(len(all_ids))), f"ids={all_ids}")

        # every annotation has required COCO fields
        required = {"id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd"}
        for ann in data["annotations"]:
            for field in required:
                check(f"ann id={ann['id']}: has '{field}'", field in ann)

        # bbox sanity: width and height are non-negative
        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]
            check(f"ann id={ann['id']}: bbox w>=0", w >= 0)
            check(f"ann id={ann['id']}: bbox h>=0", h >= 0)

        # area sanity: positive
        for ann in data["annotations"]:
            check(f"ann id={ann['id']}: area>0", ann["area"] > 0,
                  f"area={ann['area']}")

        # segmentation sanity: non-empty list of lists
        for ann in data["annotations"]:
            check(f"ann id={ann['id']}: segmentation non-empty",
                  isinstance(ann["segmentation"], list) and len(ann["segmentation"]) > 0)

        # semantic mask files exist
        for img_idx, inst_slices in scenarios:
            sem_path = output_dir / "masks" / f"{img_idx:04d}_semantic.png"
            check(f"semantic mask img{img_idx} exists", sem_path.exists())

        # instance mask files exist for non-empty scenarios
        for img_idx, inst_slices in scenarios:
            for inst_id in inst_slices:
                inst_path = (output_dir / "masks_instance" /
                             f"{img_idx:04d}_inst_{inst_id}.png")
                check(f"instance mask img{img_idx} inst{inst_id} exists",
                      inst_path.exists())

        # category_id is correct in all annotations
        for ann in data["annotations"]:
            check(f"ann id={ann['id']}: category_id=1",
                  ann["category_id"] == 1, f"got {ann['category_id']}")

        # iscrowd is 0 for all
        for ann in data["annotations"]:
            check(f"ann id={ann['id']}: iscrowd=0", ann["iscrowd"] == 0)


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_save_semantic_mask()
    test_save_instance_mask()
    test_compute_bbox()
    test_mask_to_polygons()
    test_init_coco()
    test_write_coco()
    test_integration_pipeline_loop()

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
