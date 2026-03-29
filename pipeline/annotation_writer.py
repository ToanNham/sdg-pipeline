import json
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure

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
                             path: Path, height: int, width: int) -> None:
    """Save a single RGB PNG with each instance painted in its assigned color.

    Args:
        instance_masks: {inst_id: uint8 array (H, W)} — 255 = object pixel
        inst_colors:    {inst_id: (R, G, B)} from assign_instance_colors()
        path:           output file path
        height, width:  image dimensions
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
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
        if not poly.is_valid or poly.is_empty or poly.area <= 1:
            continue
        parts = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
        for part in parts:
            if part.area <= 1:
                continue
            coords = list(part.exterior.coords)
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
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


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


# ---------------------------------------------------------------------------
# Extensibility wrapper
# ---------------------------------------------------------------------------

class AnnotationWriter:
    """Thin wrapper around module-level annotation functions.

    Subclass and override individual methods to customise annotation output
    without touching any other part of the pipeline.  Override ``finalize``
    to write formats (e.g. COCO JSON) that span all frames.
    """

    def __init__(self, cfg=None):
        self._coco   = init_coco(cfg) if cfg is not None else None
        self._ann_id = 1

    def assign_colors(self, id_map):
        return assign_instance_colors(id_map)

    def write_label(self, label, path):
        write_label_json(label, path)

    def add_coco_image_and_annotations(self, img_idx, id_map, category_map,
                                       binary_masks, render_cfg):
        """Accumulate one image + per-instance annotations into the COCO dict.

        Args:
            img_idx:      0-based image index
            id_map:       {inst_id: obj_name}
            category_map: {inst_id: category_id}
            binary_masks: {inst_id: uint8 ndarray (H, W), 255=object pixel}
            render_cfg:   cfg["render"] dict (for resolution)
        """
        if self._coco is None:
            return
        img_w = render_cfg.get("resolution_x", 640)
        img_h = render_cfg.get("resolution_y", 640)
        self._coco["images"].append({
            "id":        img_idx,
            "file_name": f"{img_idx:04d}_0001.png",
            "width":     img_w,
            "height":    img_h,
        })
        for inst_id, binary_mask in binary_masks.items():
            cat_id = category_map.get(inst_id, 1)
            bbox   = compute_bbox(binary_mask)
            if bbox[2] == 0 or bbox[3] == 0:
                continue
            polygons = mask_to_polygons(binary_mask)
            area     = int(np.count_nonzero(binary_mask))
            self._coco["annotations"].append({
                "id":          self._ann_id,
                "image_id":    img_idx,
                "category_id": cat_id,
                "bbox":        bbox,
                "segmentation": polygons,
                "area":        area,
                "iscrowd":     0,
            })
            self._ann_id += 1

    def finalize(self, output_dir):
        """Called once after all frames are rendered.

        Writes COCO JSON when a cfg was supplied at construction time.
        Override this to write additional multi-frame output formats.
        """
        if self._coco is not None:
            write_coco(self._coco, Path(output_dir) / "annotations" / "instances.json")
