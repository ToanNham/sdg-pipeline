"""Comprehensive tests for Phase 6: merge_coco.py and validate_output.py

Runs without Blender. Creates all fixtures in tempdir.
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results = []

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    _results.append(cond)


def _load_module(path: Path, name: str):
    """Load a top-level script as a module without executing __main__."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


merge_mod = _load_module(ROOT / "merge_coco.py", "merge_coco")
validate_mod = _load_module(ROOT / "validate_output.py", "validate_output")


def _make_coco(n_images: int, n_anns_per_image: int = 2,
               img_id_start: int = 0, ann_id_start: int = 0,
               file_prefix: str = "images/") -> dict:
    """Build a minimal valid COCO dict."""
    coco = {
        "info": {"description": "SDG Pipeline", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "widget", "supercategory": "object"}],
        "images": [],
        "annotations": [],
    }
    ann_id = ann_id_start
    for i in range(n_images):
        img_id = img_id_start + i
        coco["images"].append({
            "id": img_id,
            "file_name": f"{file_prefix}{img_id:04d}.png",
            "width": 640, "height": 480,
        })
        for _ in range(n_anns_per_image):
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [[10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]],
                "area": 1600,
                "bbox": [10, 10, 40, 40],
                "iscrowd": 0,
            })
            ann_id += 1
    return coco


def _write_coco(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _make_rgb(path: Path, size=(640, 480)) -> None:
    """Write a solid-colour PNG at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(128, 64, 32))
    img.save(path)


def _make_config(tmp: Path, resolution_x=640, resolution_y=480) -> Path:
    cfg = {
        "render": {"resolution_x": resolution_x, "resolution_y": resolution_y},
        "assets": {"models": [{"category_id": 1, "category_name": "widget"}]},
    }
    p = tmp / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(cfg, f)
    return p


# ===========================================================================
# merge_coco.py tests
# ===========================================================================

def test_merge_two_files():
    print("\n=== merge: two equal-sized files ===")
    with tempfile.TemporaryDirectory() as tmp:
        ann_dir = Path(tmp) / "annotations"
        ann_dir.mkdir()

        # job 0_25 and job 25_50 — both use IDs starting from 0
        file_a = _make_coco(3, n_anns_per_image=2, img_id_start=0, ann_id_start=0)
        file_b = _make_coco(3, n_anns_per_image=2, img_id_start=0, ann_id_start=0)
        _write_coco(ann_dir / "instances_0_25.json", file_a)
        _write_coco(ann_dir / "instances_25_50.json", file_b)

        out_path = Path(tmp) / "instances.json"
        merge_mod.merge(str(ann_dir), str(out_path))

        check("output file created", out_path.exists())
        with open(out_path) as f:
            merged = json.load(f)

        check("total images = 6", len(merged["images"]) == 6,
              f"got {len(merged['images'])}")
        check("total annotations = 12", len(merged["annotations"]) == 12,
              f"got {len(merged['annotations'])}")

        # image IDs must be unique and sequential
        img_ids = [img["id"] for img in merged["images"]]
        check("image IDs unique", len(img_ids) == len(set(img_ids)), f"ids={img_ids}")
        check("image IDs sequential 0-5",
              sorted(img_ids) == list(range(6)), f"ids={sorted(img_ids)}")

        # annotation IDs must be unique and sequential
        ann_ids = [a["id"] for a in merged["annotations"]]
        check("annotation IDs unique", len(ann_ids) == len(set(ann_ids)))
        check("annotation IDs sequential 0-11",
              sorted(ann_ids) == list(range(12)), f"ids={sorted(ann_ids)}")

        # all image_ids in annotations must reference a valid image
        valid_img_ids = set(img_ids)
        ref_ids = {a["image_id"] for a in merged["annotations"]}
        check("all annotation image_ids valid",
              ref_ids.issubset(valid_img_ids), f"refs={ref_ids}")

        # structure preserved
        check("info preserved", merged["info"]["description"] == "SDG Pipeline")
        check("categories preserved", len(merged["categories"]) == 1)


def test_merge_single_file():
    print("\n=== merge: single file (identity) ===")
    with tempfile.TemporaryDirectory() as tmp:
        ann_dir = Path(tmp) / "annotations"
        ann_dir.mkdir()

        src = _make_coco(4, n_anns_per_image=1, img_id_start=0, ann_id_start=0)
        _write_coco(ann_dir / "instances_0_4.json", src)

        out_path = Path(tmp) / "instances.json"
        merge_mod.merge(str(ann_dir), str(out_path))

        with open(out_path) as f:
            merged = json.load(f)

        check("4 images", len(merged["images"]) == 4)
        check("4 annotations", len(merged["annotations"]) == 4)
        img_ids = [img["id"] for img in merged["images"]]
        check("image IDs 0-3", sorted(img_ids) == list(range(4)))


def test_merge_three_files():
    print("\n=== merge: three files ===")
    with tempfile.TemporaryDirectory() as tmp:
        ann_dir = Path(tmp) / "annotations"
        ann_dir.mkdir()

        for i in range(3):
            data = _make_coco(2, n_anns_per_image=3,
                              img_id_start=0, ann_id_start=0)
            _write_coco(ann_dir / f"instances_{i*2}_{i*2+2}.json", data)

        out_path = Path(tmp) / "instances.json"
        merge_mod.merge(str(ann_dir), str(out_path))

        with open(out_path) as f:
            merged = json.load(f)

        check("6 images total", len(merged["images"]) == 6)
        check("18 annotations total", len(merged["annotations"]) == 18)
        img_ids = sorted(img["id"] for img in merged["images"])
        check("image IDs 0-5", img_ids == list(range(6)), f"ids={img_ids}")

        # Every annotation's image_id must point to a real image
        valid = set(img["id"] for img in merged["images"])
        bad = [a for a in merged["annotations"] if a["image_id"] not in valid]
        check("no dangling image_ids", len(bad) == 0,
              f"{len(bad)} dangling annotations")


def test_merge_no_files_exits():
    print("\n=== merge: no files -> sys.exit(1) ===")
    with tempfile.TemporaryDirectory() as tmp:
        ann_dir = Path(tmp) / "empty"
        ann_dir.mkdir()
        try:
            merge_mod.merge(str(ann_dir), str(Path(tmp) / "out.json"))
            check("should have raised SystemExit", False)
        except SystemExit as e:
            check("exits with code 1", e.code == 1, f"code={e.code}")


def test_merge_offset_correctness():
    """IDs from file B must shift past all IDs in file A."""
    print("\n=== merge: offset correctness ===")
    with tempfile.TemporaryDirectory() as tmp:
        ann_dir = Path(tmp) / "annotations"
        ann_dir.mkdir()

        # File A: 5 images, 2 anns each  → img IDs 0-4, ann IDs 0-9
        # File B: 3 images, 4 anns each  → img IDs 0-2, ann IDs 0-11 (before offset)
        fa = _make_coco(5, n_anns_per_image=2, img_id_start=0, ann_id_start=0)
        fb = _make_coco(3, n_anns_per_image=4, img_id_start=0, ann_id_start=0)
        _write_coco(ann_dir / "instances_0_5.json", fa)
        _write_coco(ann_dir / "instances_5_8.json", fb)

        out_path = Path(tmp) / "out.json"
        merge_mod.merge(str(ann_dir), str(out_path))

        with open(out_path) as f:
            merged = json.load(f)

        img_ids = sorted(img["id"] for img in merged["images"])
        ann_ids = sorted(a["id"] for a in merged["annotations"])
        check("img IDs 0-7", img_ids == list(range(8)), f"ids={img_ids}")
        check("ann IDs 0-21", ann_ids == list(range(22)), f"ids={ann_ids}")

        # Annotations that came from file B (img_id >= 5) must have image_id 5-7
        b_anns = [a for a in merged["annotations"] if a["image_id"] >= 5]
        check("file-B annotations reference shifted image IDs",
              all(5 <= a["image_id"] <= 7 for a in b_anns),
              f"img_ids={[a['image_id'] for a in b_anns]}")


# ===========================================================================
# validate_output.py tests
# ===========================================================================

def test_validate_pass():
    print("\n=== validate: PASS — all correct ===")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        cfg_path = _make_config(Path(tmp), 640, 480)

        # Build COCO with 3 images
        coco = _make_coco(3, n_anns_per_image=1, img_id_start=0, ann_id_start=0)
        _write_coco(out / "annotations" / "instances_0_3.json", coco)

        # Create matching RGB files
        for img in coco["images"]:
            _make_rgb(out / img["file_name"], size=(640, 480))

        # Should not raise
        try:
            validate_mod.validate(str(out), str(cfg_path))
            check("no exception", True)
        except SystemExit as e:
            check("no exception", False, f"exited with {e.code}")


def test_validate_missing_rgb():
    print("\n=== validate: FAIL — missing RGB file ===")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        cfg_path = _make_config(Path(tmp), 640, 480)

        coco = _make_coco(2, n_anns_per_image=1)
        _write_coco(out / "annotations" / "instances_0_2.json", coco)
        # Do NOT create RGB files

        try:
            validate_mod.validate(str(out), str(cfg_path))
            check("should have exited", False)
        except SystemExit as e:
            check("exits 1 on missing RGB", e.code == 1, f"code={e.code}")


def test_validate_wrong_dimensions():
    print("\n=== validate: FAIL — wrong image dimensions ===")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        cfg_path = _make_config(Path(tmp), 640, 480)

        coco = _make_coco(1, n_anns_per_image=1)
        _write_coco(out / "annotations" / "instances_0_1.json", coco)
        # Write wrong-size image
        _make_rgb(out / coco["images"][0]["file_name"], size=(320, 240))

        try:
            validate_mod.validate(str(out), str(cfg_path))
            check("should have exited", False)
        except SystemExit as e:
            check("exits 1 on dimension mismatch", e.code == 1, f"code={e.code}")


def test_validate_no_annotations_for_image():
    print("\n=== validate: FAIL — image with no annotations ===")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        cfg_path = _make_config(Path(tmp), 640, 480)

        coco = _make_coco(2, n_anns_per_image=0)  # no annotations at all
        _write_coco(out / "annotations" / "instances_0_2.json", coco)
        for img in coco["images"]:
            _make_rgb(out / img["file_name"], size=(640, 480))

        try:
            validate_mod.validate(str(out), str(cfg_path))
            check("should have exited", False)
        except SystemExit as e:
            check("exits 1 when image has no annotations", e.code == 1, f"code={e.code}")


def test_validate_zero_area_annotation():
    print("\n=== validate: FAIL — zero-area annotation ===")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        cfg_path = _make_config(Path(tmp), 640, 480)

        coco = _make_coco(1, n_anns_per_image=1)
        coco["annotations"][0]["area"] = 0   # corrupt area
        _write_coco(out / "annotations" / "instances_0_1.json", coco)
        _make_rgb(out / coco["images"][0]["file_name"], size=(640, 480))

        try:
            validate_mod.validate(str(out), str(cfg_path))
            check("should have exited", False)
        except SystemExit as e:
            check("exits 1 on zero-area annotation", e.code == 1, f"code={e.code}")


def test_validate_no_json():
    print("\n=== validate: FAIL — no annotations JSON ===")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        cfg_path = _make_config(Path(tmp), 640, 480)
        (out / "annotations").mkdir(parents=True)  # dir exists but is empty

        try:
            validate_mod.validate(str(out), str(cfg_path))
            check("should have exited", False)
        except SystemExit as e:
            check("exits 1 when no JSON found", e.code == 1, f"code={e.code}")


def test_validate_multiple_issues():
    print("\n=== validate: FAIL — multiple simultaneous issues ===")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        cfg_path = _make_config(Path(tmp), 640, 480)

        coco = _make_coco(3, n_anns_per_image=1)
        # Corrupt annotation for image 0: zero area
        coco["annotations"][0]["area"] = 0
        # Image 1: wrong-size RGB
        # Image 2: missing RGB entirely
        _write_coco(out / "annotations" / "instances.json", coco)
        _make_rgb(out / coco["images"][0]["file_name"], size=(640, 480))
        _make_rgb(out / coco["images"][1]["file_name"], size=(100, 100))
        # image 2 file intentionally absent

        try:
            validate_mod.validate(str(out), str(cfg_path))
            check("should have exited", False)
        except SystemExit as e:
            check("exits 1 with multiple issues", e.code == 1, f"code={e.code}")


def test_validate_counts_printed(capsys_stub=None):
    """Smoke-test: PASS message is printed and contains image/annotation counts."""
    print("\n=== validate: PASS message format ===")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        cfg_path = _make_config(Path(tmp), 640, 480)

        coco = _make_coco(2, n_anns_per_image=3)
        _write_coco(out / "annotations" / "instances_0_2.json", coco)
        for img in coco["images"]:
            _make_rgb(out / img["file_name"], size=(640, 480))

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                validate_mod.validate(str(out), str(cfg_path))
        except SystemExit:
            check("should not fail", False)
            return

        output = buf.getvalue()
        check("output contains PASS", "PASS" in output, f"got: {output!r}")
        check("output contains image count '2'", "2" in output, f"got: {output!r}")
        check("output contains annotation count '6'", "6" in output, f"got: {output!r}")


# ===========================================================================
# Integration: merge then validate
# ===========================================================================

def test_integration_merge_then_validate():
    """Merge two partial jobs, then validate the merged result."""
    print("\n=== integration: merge -> validate ===")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "output"
        cfg_path = _make_config(Path(tmp), 640, 480)
        ann_dir = out / "annotations"
        ann_dir.mkdir(parents=True)

        # Two jobs, each 3 images with IDs starting at 0 (as produced by run.py)
        for job in range(2):
            data = _make_coco(3, n_anns_per_image=2,
                              img_id_start=0, ann_id_start=0,
                              file_prefix="images/")
            _write_coco(ann_dir / f"instances_{job*3}_{job*3+3}.json", data)

        # Merge
        merged_path = ann_dir / "instances.json"
        merge_mod.merge(str(ann_dir), str(merged_path))

        with open(merged_path) as f:
            merged = json.load(f)
        check("merged: 6 images", len(merged["images"]) == 6)
        check("merged: 12 annotations", len(merged["annotations"]) == 12)

        # Create RGB files matching merged image entries
        for img in merged["images"]:
            _make_rgb(out / img["file_name"], size=(640, 480))

        # Now validate against the merged JSON
        # validate_output.py picks the first instances*.json — rename partial files
        # so only the merged file matches
        (ann_dir / "instances_0_3.json").rename(ann_dir / "_partial_0_3.json")
        (ann_dir / "instances_3_6.json").rename(ann_dir / "_partial_3_6.json")

        try:
            validate_mod.validate(str(out), str(cfg_path))
            check("validate PASS after merge", True)
        except SystemExit as e:
            check("validate PASS after merge", False, f"exited {e.code}")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    # merge_coco tests
    test_merge_two_files()
    test_merge_single_file()
    test_merge_three_files()
    test_merge_no_files_exits()
    test_merge_offset_correctness()

    # validate_output tests
    test_validate_pass()
    test_validate_missing_rgb()
    test_validate_wrong_dimensions()
    test_validate_no_annotations_for_image()
    test_validate_zero_area_annotation()
    test_validate_no_json()
    test_validate_multiple_issues()
    test_validate_counts_printed()

    # integration
    test_integration_merge_then_validate()

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
