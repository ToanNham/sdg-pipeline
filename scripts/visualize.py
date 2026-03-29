"""Visualize one pipeline output: RGB image with mask overlays and bounding boxes.

Reads the color instance mask (masks/NNNN_color_0001.png) and the KV label JSON
(labels/NNNN_label.json) to draw per-object contours and name labels.

Usage:
    python visualize.py --output_dir output --idx 0
    python visualize.py --output_dir examples/output --idx 3

Output: prints object summary, saves output/inspect_NNNN.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_mask_contour(draw: ImageDraw.Draw, mask_arr: np.ndarray, color: tuple, width: int = 2):
    """Draw the contour of a binary mask using simple edge detection."""
    m = mask_arr.astype(np.uint8)
    h_edge = np.zeros_like(m)
    h_edge[:, :-1] = np.abs(m[:, 1:].astype(int) - m[:, :-1].astype(int)).astype(np.uint8)
    v_edge = np.zeros_like(m)
    v_edge[:-1, :] = np.abs(m[1:, :].astype(int) - m[:-1, :].astype(int)).astype(np.uint8)
    edge = (h_edge | v_edge) > 0
    ys, xs = np.where(edge)
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.rectangle([x - width // 2, y - width // 2, x + width // 2, y + width // 2], fill=color)


def mask_bbox(mask_arr: np.ndarray):
    """Return (x, y, w, h) bounding box of a binary mask, or None if empty."""
    ys, xs = np.where(mask_arr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())


def main():
    parser = argparse.ArgumentParser(description="Visualize SDG pipeline output")
    parser.add_argument("--output_dir", default="output", help="Pipeline output directory")
    parser.add_argument("--idx", type=int, default=0, help="Image index (0-based)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    idx = args.idx

    # Load RGB image
    img_path = output_dir / "images" / f"{idx:04d}_0001.png"
    if not img_path.exists():
        print(f"ERROR: image not found: {img_path}")
        sys.exit(1)

    rgb = Image.open(img_path).convert("RGBA")
    W, H = rgb.size

    # Load color mask
    color_mask_path = output_dir / "masks" / f"{idx:04d}_color_0001.png"
    color_mask = None
    if color_mask_path.exists():
        color_mask = np.array(Image.open(color_mask_path).convert("RGB"))

    # Load KV label JSON
    label_path = output_dir / "labels" / f"{idx:04d}_label.json"
    label = {}
    if label_path.exists():
        with open(label_path) as f:
            label = json.load(f)

    objects = label.get("Objects", [])
    target_objects = [o for o in objects if "mask_color" in o]
    distractor_objects = [o for o in objects if "mask_color" not in o]

    print(f"\nImage:       {img_path.name}  ({W}x{H})")
    print(f"Color mask:  {color_mask_path.name if color_mask is not None else '(not found)'}")
    print(f"Label:       {label_path.name if label_path.exists() else '(not found)'}")
    print(f"Objects:     {len(target_objects)} target(s), {len(distractor_objects)} distractor(s)")
    if label.get("Wall_Image"):
        print(f"Background:  {label['Wall_Image']}")
    print()

    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    for k, obj in enumerate(target_objects):
        name = obj["name"]
        mc = tuple(obj["mask_color"])  # (R, G, B) used in color mask

        if color_mask is not None:
            # Extract binary mask for this object by matching its exact color
            tol = 5
            binary = (
                (np.abs(color_mask[:, :, 0].astype(int) - mc[0]) <= tol) &
                (np.abs(color_mask[:, :, 1].astype(int) - mc[1]) <= tol) &
                (np.abs(color_mask[:, :, 2].astype(int) - mc[2]) <= tol)
            )
            draw_mask_contour(draw, binary, mc + (220,), width=2)
            bbox = mask_bbox(binary)
        else:
            bbox = None

        if bbox:
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline=mc + (255,), width=2)
            label_text = name
            bbox_text = draw.textbbox((x + 2, y - 18), label_text, font=font)
            draw.rectangle(bbox_text, fill=(0, 0, 0, 160))
            draw.text((x + 2, y - 18), label_text, fill=mc + (255,), font=font)
            px = int(obj["position"][0] * 100) / 100
            py = int(obj["position"][1] * 100) / 100
            pz = int(obj["position"][2] * 100) / 100
            print(f"  [{k}] {name:20s}  color={mc}  bbox=({x},{y},{w},{h})  pos=({px},{py},{pz})")
        else:
            print(f"  [{k}] {name:20s}  color={mc}  (no mask pixels found — may be occluded or out of frame)")

    out_path = output_dir / f"inspect_{idx:04d}.png"
    result = Image.alpha_composite(rgb, overlay).convert("RGB")
    result.save(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
