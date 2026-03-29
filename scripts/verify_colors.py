"""verify_colors.py — Batch-check that mask PNG pixel colors match JSON label mask_color fields.

For each image, loads the RGB color mask and the KV label JSON, then for every target object:
  - Counts exact-match pixels
  - Counts approximate-match pixels (tol=5, same tolerance as visualize.py)
  - Counts gamma-shifted pixels (predicts what Blender writes if it sRGB-encodes linear values)
  - Reports orphan pixels (colored pixels not owned by any label entry)

Exits 0 if all objects have at least one matching pixel (exact or gamma).
Exits 1 if any object has zero matches under both hypotheses.

Usage:
    python verify_colors.py --output_dir output [--start 0] [--end 49] [--tol 5] [--verbose]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def to_srgb(c_uint8: int) -> int:
    """Predict the sRGB-encoded value Blender would write for a linear 0-1 float.

    The compositor receives c/255 as a linear value. If Blender applies the
    standard sRGB transfer function (gamma ~2.2) when writing the PNG, the
    stored byte would be round((c/255)^(1/2.2) * 255).
    """
    v = c_uint8 / 255.0
    if v <= 0.0:
        return 0
    encoded = pow(v, 1.0 / 2.2) * 255.0
    return int(round(min(encoded, 255.0)))


def srgb_color(mc: tuple) -> tuple:
    return tuple(to_srgb(c) for c in mc)


def count_matching_pixels(arr: np.ndarray, color: tuple, tol: int) -> int:
    """Count pixels within tol of color (per-channel L∞ distance)."""
    r, g, b = color
    mask = (
        (np.abs(arr[:, :, 0].astype(int) - r) <= tol) &
        (np.abs(arr[:, :, 1].astype(int) - g) <= tol) &
        (np.abs(arr[:, :, 2].astype(int) - b) <= tol)
    )
    return int(mask.sum())


def main():
    parser = argparse.ArgumentParser(description="Verify mask PNG colors match JSON labels")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=49)
    parser.add_argument("--tol", type=int, default=5, help="Per-channel tolerance for approximate match")
    parser.add_argument("--verbose", action="store_true", help="Print every object row, not just failures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tol = args.tol
    failures = []
    total_objects = 0
    total_exact = 0
    total_approx_only = 0
    total_gamma_only = 0
    total_missing = 0
    total_orphan_px = 0

    print(f"{'idx':>4}  {'object':<22}  {'label_color':>15}  {'exact':>7}  {'approx':>7}  "
          f"{'gamma_c':>15}  {'gamma':>7}  {'status'}")
    print("-" * 105)

    for idx in range(args.start, args.end + 1):
        mask_path = output_dir / "masks" / f"{idx:04d}_color_0001.png"
        label_path = output_dir / "labels" / f"{idx:04d}_label.json"

        if not mask_path.exists() or not label_path.exists():
            print(f"{idx:4d}  (skipped — files missing)")
            continue

        arr = np.array(Image.open(mask_path).convert("RGB"))
        with open(label_path) as f:
            label = json.load(f)

        target_objs = [o for o in label.get("Objects", []) if "mask_color" in o]
        label_colors = {tuple(o["mask_color"]) for o in target_objs}

        # Orphan pixels: non-black pixels not matching any label color (exact)
        non_black = ~((arr[:, :, 0] == 0) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 0))
        orphan_mask = non_black.copy()
        for mc in label_colors:
            owned = (
                (np.abs(arr[:, :, 0].astype(int) - mc[0]) <= tol) &
                (np.abs(arr[:, :, 1].astype(int) - mc[1]) <= tol) &
                (np.abs(arr[:, :, 2].astype(int) - mc[2]) <= tol)
            )
            orphan_mask &= ~owned
        orphan_px = int(orphan_mask.sum())
        total_orphan_px += orphan_px

        for obj in target_objs:
            total_objects += 1
            name = obj["name"]
            mc = tuple(obj["mask_color"])
            gc = srgb_color(mc)

            exact  = count_matching_pixels(arr, mc, 0)
            approx = count_matching_pixels(arr, mc, tol)
            gamma  = count_matching_pixels(arr, gc, tol)

            if exact > 0:
                status = "OK-exact"
                total_exact += 1
            elif approx > 0:
                status = "OK-approx"
                total_approx_only += 1
            elif gamma > 0:
                status = "GAMMA-DRIFT"
                total_gamma_only += 1
                failures.append((idx, name, mc, gc, gamma))
            else:
                status = "MISSING"
                total_missing += 1
                failures.append((idx, name, mc, gc, 0))

            if args.verbose or status not in ("OK-exact", "OK-approx"):
                print(f"{idx:4d}  {name:<22}  {str(mc):>15}  {exact:>7}  {approx:>7}  "
                      f"{str(gc):>15}  {gamma:>7}  {status}")

        if orphan_px > 0:
            print(f"{idx:4d}  {'(orphan pixels)':22}  {'':>15}  {'':>7}  {'':>7}  "
                  f"{'':>15}  {'':>7}  {orphan_px} orphan px")

    print("-" * 105)
    print(f"\nSummary ({args.start}–{args.end}):")
    print(f"  Total objects checked : {total_objects}")
    print(f"  Exact match           : {total_exact}")
    print(f"  Approx match (tol={tol}) : {total_approx_only}")
    print(f"  Gamma-drift match     : {total_gamma_only}  (color-space bug if > 0)")
    print(f"  Missing (no match)    : {total_missing}")
    print(f"  Total orphan pixels   : {total_orphan_px}")

    if failures:
        print(f"\nFailures / warnings ({len(failures)}):")
        for idx, name, mc, gc, gpx in failures:
            tag = "GAMMA-DRIFT" if gpx > 0 else "MISSING"
            print(f"  [{tag}] img={idx:04d}  {name}  label={mc}  gamma_pred={gc}  gamma_px={gpx}")

    if total_missing > 0:
        print(f"\nERROR: {total_missing} object(s) have zero pixels under both exact and gamma hypotheses.")
        sys.exit(1)

    if total_gamma_only > 0:
        print(f"\nWARNING: {total_gamma_only} object(s) only match under gamma hypothesis.")
        print("  The compositor is likely sRGB-encoding the mask PNG (linear→sRGB gamma).")
        print("  Fix: set color_out.format.color_management = 'OVERRIDE' and")
        print("       color_out.format.linear_colorspace_settings.name = 'Linear'")
        print("  in renderer.py setup_compositor(), or pass pre-gamma colors to the compositor.")
        sys.exit(1)

    print("\nAll objects match. Mask colors are correct.")
    sys.exit(0)


if __name__ == "__main__":
    main()
