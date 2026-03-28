# Engineering Notes

## Masking: ID_MASK vs Cryptomatte

**Current implementation** uses Cryptomatte (`pipeline/mask_extractor.py` + OIIO EXR reading).

**Consider switching to ID_MASK** for a simpler pipeline.

### How ID_MASK would work
Before each render, assign `obj.pass_index = i + 1` to each target object. In `setup_compositor()`, create one `CompositorNodeIDMask` node per object (index = pass_index), wire each to a PNG File Output slot. One render → all instance mask PNGs written directly. No EXR, no OIIO, no MurmurHash.

This would eliminate:
- `pipeline/mask_extractor.py` (EXR reading + hash matching)
- The `write_still=True` / OPEN_EXR_MULTILAYER render path
- OIIO dependency
- The MurmurHash3 implementation (which was initially wrong — was MurmurHash2, caused `annotations=0`)

### Why Cryptomatte was kept
The user chose to keep Cryptomatte for now. It handles semi-transparent pixels and depth-of-field blur more accurately (stores per-pixel coverage weights rather than hard frontmost-object assignment).

### When ID_MASK is sufficient
For **opaque rigid objects with no DoF** (which is this pipeline's current setup — aluminium cans, Cycles, no depth of field), ID_MASK produces identical results with far less complexity. If the scene never uses DoF or transparent materials, ID_MASK is the better choice.

### Known Cryptomatte gotchas (discovered during phase 7 debug)
- Python `openexr` pip package crashes Blender (DLL conflict with Blender's bundled OpenEXR) — must use `OpenImageIO` (bundled with Blender) instead
- Cryptomatte EXR channels use **lowercase** `r/g/b/a` (e.g. `ViewLayer.CryptoObject00.r`), not uppercase
- Blender uses **MurmurHash3 x86 32-bit** (seed=0), not MurmurHash2 — the original implementation was wrong
- The compositor `OPEN_EXR_MULTILAYER` output node crashes on Windows (MSVCP140.dll); workaround is `write_still=True` with `scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'`
