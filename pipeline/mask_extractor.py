"""Phase 4 – mask_extractor.py

Loads per-instance binary mask PNGs written directly by the compositor's
ID_MASK nodes.  No EXR, no OpenImageIO, no hashing required.
"""

import numpy as np
from pathlib import Path
from PIL import Image


def load_instance_masks(img_idx: int, id_map: dict, output_dir: Path) -> dict:
    """Load mask PNGs written by the compositor's ID_MASK nodes.

    Args:
        img_idx:    0-based image index
        id_map:     {inst_id: obj_name} from assign_instance_ids()
        output_dir: pipeline output root (same Path passed to setup_compositor)

    Returns:
        {inst_id: np.ndarray(uint8, H×W)} — 255 = object pixel, 0 = background.
        Objects with no visible pixels are omitted.

    File naming convention (Blender appends "_0001" frame suffix):
        output_dir/masks_instance/{img_idx:04d}_inst_{inst_id}_0001.png
    """
    result = {}
    for inst_id in id_map:
        path = output_dir / "masks_instance" / f"{img_idx:04d}_inst_{inst_id}_0001.png"
        if not path.exists():
            continue
        arr = np.array(Image.open(path).convert("L"))
        if arr.any():
            result[inst_id] = (arr > 127).astype(np.uint8) * 255
    return result


def build_semantic_mask(
    instance_masks: dict,
    category_map: dict,
    H: int,
    W: int,
) -> np.ndarray:
    """Combine instance masks into a single semantic segmentation mask.

    Args:
        instance_masks: {inst_id: uint8 array (H, W)} from load_instance_masks()
        category_map:   {inst_id: category_id}
        H:              image height
        W:              image width

    Returns:
        uint8 array (H, W); pixel value = category_id, 0 = background.
    """
    semantic = np.zeros((H, W), dtype=np.uint8)
    for inst_id, mask in instance_masks.items():
        cat = category_map.get(inst_id, 0)
        semantic[mask > 0] = cat
    return semantic
