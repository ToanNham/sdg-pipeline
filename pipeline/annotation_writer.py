import json
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon


# Distinct RGB colors for up to 20 instances (black = background).
# Order matches the KV example palette (blues/greens/yellows/reds).
_INSTANCE_COLORS = [
    (  0,  72, 255), (255, 218,   0), (  0, 145, 255), (  0, 255, 145),
    (  0, 255,  72), ( 72, 255,   0), (255,  72,   0), (218, 255,   0),
    (255, 145,   0), (255,   0,   0), (  0, 218, 255), (145, 255,   0),
    (  0, 255, 218), (  0, 255,   0), (  0,   0, 255), (255,   0, 145),
    (255,   0, 218), (145,   0, 255), (  0, 255, 255), (255, 255,   0),
]


def assign_instance_colors(id_map: dict) -> dict:
    """Return {inst_id: (R, G, B)} using the canonical palette, sorted by inst_id.

    This mapping is the single source of truth — both the color mask PNG and the
    label JSON must use values from this function so they stay in sync.
    """
    return {inst_id: _INSTANCE_COLORS[i % len(_INSTANCE_COLORS)]
            for i, inst_id in enumerate(sorted(id_map.keys()))}


def save_instance_color_mask(instance_masks: dict, inst_colors: dict,
                             path: Path, H: int, W: int) -> None:
    """Save a single RGB PNG with each instance painted in its assigned color.

    Args:
        instance_masks: {inst_id: uint8 array (H, W)} — 255 = object pixel
        inst_colors:    {inst_id: (R, G, B)} from assign_instance_colors()
        path:           output file path
        H, W:           image height and width
    """
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for inst_id, mask in instance_masks.items():
        color = inst_colors.get(inst_id, (255, 255, 255))
        canvas[mask > 0] = color
    Image.fromarray(canvas, mode="RGB").save(str(path))


def write_label_json(label: dict, path: Path) -> None:
    """Write the per-image label JSON (KV format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(label, f, indent=4)


def save_semantic_mask(mask: np.ndarray, path: Path):
    Image.fromarray(mask, mode="L").save(str(path))


def save_instance_mask(mask: np.ndarray, path: Path):
    Image.fromarray(mask, mode="L").save(str(path))


def mask_to_polygons(binary_mask: np.ndarray, tolerance: int = 2) -> list:
    contours = measure.find_contours(binary_mask, 0.5)
    polygons = []
    for contour in contours:
        contour = np.flip(contour, axis=1)  # row,col -> x,y
        if len(contour) < 3:
            continue
        poly = Polygon(contour).simplify(tolerance, preserve_topology=False)
        if poly.is_valid and not poly.is_empty and poly.area > 1:
            coords = list(poly.exterior.coords)
            flat   = [v for pt in coords for v in pt]
            polygons.append(flat)
    return polygons


def compute_bbox(binary_mask: np.ndarray) -> list:
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]


def init_coco(cfg) -> dict:
    categories = []
    for entry in cfg["assets"]["models"]:
        categories.append({
            "id":            entry["category_id"],
            "name":          entry["category_name"],
            "supercategory": "object"
        })
    return {
        "info":        {"description": "SDG Pipeline", "version": "1.0"},
        "licenses":    [],
        "categories":  categories,
        "images":      [],
        "annotations": []
    }


def write_coco(coco: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as f:
        json.dump(coco, f, indent=2)
