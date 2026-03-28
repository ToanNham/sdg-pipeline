"""Phase 4 – mask_extractor.py

Decodes Blender's Cryptomatte multilayer EXR output into per-instance binary
masks and a semantic segmentation mask.

Cryptomatte encodes each pixel's object identity as a float32 hash produced
by MurmurHash2. This module re-implements that hash so we can map the float32
values stored in EXR channels back to known instance IDs.
"""

import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# MurmurHash2 – must match Blender's internal Cryptomatte hash exactly
# ---------------------------------------------------------------------------

def mm2_hash_str(name: str) -> float:
    """Blender's Cryptomatte hash for an object name.

    Implements MurmurHash2 over the UTF-8 bytes of *name*, then reinterprets
    the resulting 32-bit unsigned integer as an IEEE 754 float32.

    Args:
        name: bpy.types.Object.name string

    Returns:
        float32 value that Blender writes into EXR Cryptomatte channels.
    """
    data = name.encode("utf-8")
    m = 0xC6A4A793
    h = (len(data) * m) & 0xFFFFFFFF
    off = 0

    # Process 4-byte little-endian chunks
    while off + 4 <= len(data):
        k = struct.unpack_from("<I", data, off)[0]
        k = (k * m) & 0xFFFFFFFF
        k ^= k >> 16
        k = (k * m) & 0xFFFFFFFF
        h ^= k
        h = (h * m) & 0xFFFFFFFF
        off += 4

    # Remaining bytes (0–3)
    remaining = len(data) - off
    if remaining >= 3:
        h ^= data[off + 2] << 16
    if remaining >= 2:
        h ^= data[off + 1] << 8
    if remaining >= 1:
        h ^= data[off]
        h = (h * m) & 0xFFFFFFFF

    # Final avalanche
    h ^= h >> 13
    h = (h * m) & 0xFFFFFFFF
    h ^= h >> 15
    h &= 0xFFFFFFFF

    return struct.unpack("f", struct.pack("I", h))[0]


# ---------------------------------------------------------------------------
# EXR decoding
# ---------------------------------------------------------------------------

def extract_instance_masks(exr_path: Path, id_map: dict) -> dict:
    """Extract per-instance binary masks from a Cryptomatte multilayer EXR.

    Args:
        exr_path: Path to the EXR written by setup_compositor()
                  (e.g. output/tmp/0000_0001.exr)
        id_map:   {inst_id: obj_name} from assign_instance_ids()

    Returns:
        {inst_id: np.ndarray(uint8, shape=(H, W))}
        Pixels where the object contributes >50 % coverage are set to 255;
        all other pixels are 0. Objects with no visible pixels are omitted.
    """
    import OpenEXR

    f = OpenEXR.File(str(exr_path), separate_channels=True)
    channels = f.channels()

    # Derive H, W from the first available channel
    first = next(iter(channels.values()))
    H, W = first.pixels.shape

    # Precompute per-object float32 hashes
    obj_hashes = {inst_id: mm2_hash_str(name) for inst_id, name in id_map.items()}

    # Accumulate weighted coverage per instance
    coverage = {inst_id: np.zeros((H, W), dtype=np.float32) for inst_id in id_map}

    # Cryptomatte depth=2 → two layers: CryptoObject00, CryptoObject01
    for layer_idx in range(2):
        prefix = f"ViewLayer.CryptoObject{layer_idx:02d}"
        r_key = f"{prefix}.R"
        g_key = f"{prefix}.G"
        b_key = f"{prefix}.B"
        a_key = f"{prefix}.A"
        if r_key not in channels:
            # Layer absent (fewer objects than depth slots)
            continue

        id_rg = channels[r_key].pixels
        wt_rg = channels[g_key].pixels
        id_ba = channels[b_key].pixels
        wt_ba = channels[a_key].pixels

        for inst_id, obj_hash in obj_hashes.items():
            coverage[inst_id] += wt_rg * (id_rg == obj_hash)
            coverage[inst_id] += wt_ba * (id_ba == obj_hash)

    result = {}
    for inst_id, cov in coverage.items():
        mask = (cov > 0.5).astype(np.uint8) * 255
        if mask.any():
            result[inst_id] = mask
    return result


def build_semantic_mask(
    instance_masks: dict,
    category_map: dict,
    H: int,
    W: int,
) -> np.ndarray:
    """Combine instance masks into a single semantic segmentation mask.

    Args:
        instance_masks: {inst_id: uint8 array (H, W)} from extract_instance_masks()
        category_map:   {inst_id: category_id}
        H:              image height (cfg["render"]["resolution_y"])
        W:              image width  (cfg["render"]["resolution_x"])

    Returns:
        uint8 array (H, W); pixel value = category_id, 0 = background.
    """
    semantic = np.zeros((H, W), dtype=np.uint8)
    for inst_id, mask in instance_masks.items():
        cat = category_map.get(inst_id, 0)
        semantic[mask > 0] = cat
    return semantic
