"""Merge partial COCO JSON files from parallel render jobs into one file."""

import argparse
import json
import sys
from pathlib import Path


def merge(annotations_dir: str, out: str) -> None:
    ann_dir = Path(annotations_dir)
    files = sorted(ann_dir.glob("instances_*.json"))

    if not files:
        print(f"No instances_*.json files found in {annotations_dir}")
        sys.exit(1)

    merged = None
    img_offset = 0
    ann_offset = 0

    for path in files:
        with open(path) as f:
            data = json.load(f)

        if merged is None:
            merged = {
                "info": data.get("info", {}),
                "licenses": data.get("licenses", []),
                "categories": data.get("categories", []),
                "images": [],
                "annotations": [],
            }

        for img in data["images"]:
            img = dict(img)
            img["id"] += img_offset
            merged["images"].append(img)

        for ann in data["annotations"]:
            ann = dict(ann)
            ann["id"] += ann_offset
            ann["image_id"] += img_offset
            merged["annotations"].append(ann)

        img_offset += len(data["images"])
        ann_offset += len(data["annotations"])

    out_path = Path(out)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(files)} files -> {out_path}")
    print(f"Total: {len(merged['images'])} images, {len(merged['annotations'])} annotations")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge partial COCO JSON files")
    parser.add_argument("annotations_dir", help="Directory containing instances_*.json files")
    parser.add_argument("--out", default="instances.json", help="Output filename")
    args = parser.parse_args()
    merge(args.annotations_dir, args.out)


if __name__ == "__main__":
    main()
