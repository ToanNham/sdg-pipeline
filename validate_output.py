"""Validate SDG pipeline output against config expectations."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from PIL import Image


def validate(output_dir: str, config: str) -> None:
    out = Path(output_dir)
    issues = []

    # Find annotations JSON
    ann_files = sorted((out / "annotations").glob("instances*.json"))
    if not ann_files:
        print(f"FAIL — no instances*.json found in {out / 'annotations'}")
        sys.exit(1)

    with open(ann_files[0]) as f:
        data = json.load(f)

    imgs = {img["id"]: img for img in data["images"]}
    anns: dict[int, list] = {}
    for a in data["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    W = cfg["render"]["resolution_x"]
    H = cfg["render"]["resolution_y"]

    for img_id, img in imgs.items():
        rgb_path = out / img["file_name"]

        if not rgb_path.exists():
            issues.append(f"Missing RGB: {rgb_path}")
        else:
            with Image.open(rgb_path) as im:
                if im.size != (W, H):
                    issues.append(
                        f"Wrong dimensions for {rgb_path}: expected ({W},{H}), got {im.size}"
                    )

        img_anns = anns.get(img_id, [])
        if not img_anns:
            issues.append(f"No annotations for image_id={img_id} ({img['file_name']})")
        else:
            for a in img_anns:
                if a["area"] == 0:
                    issues.append(
                        f"Zero-area annotation id={a['id']} for image_id={img_id}"
                    )

    n_images = len(imgs)
    n_annotations = len(data["annotations"])

    if issues:
        print(f"FAIL — {len(issues)} issues:")
        for msg in issues:
            print(f"  - {msg}")
        sys.exit(1)
    else:
        print(f"PASS — {n_images} images, {n_annotations} annotations, 0 issues.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SDG pipeline output")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    args = parser.parse_args()
    validate(args.output_dir, args.config)


if __name__ == "__main__":
    main()
