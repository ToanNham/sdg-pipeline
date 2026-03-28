import json
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon


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
