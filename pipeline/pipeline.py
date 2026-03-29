"""SDGPipeline — the central orchestrator for the synthetic data generation loop.

Usage (default — identical behaviour to the old flat run.py):

    SDGPipeline(cfg, output_dir).run(start=0, end=50)

Usage (override camera randomization only):

    from pipeline import SDGPipeline, Randomizer

    class OverheadCamRandomizer(Randomizer):
        def randomize_camera(self, cam_obj, rng, cfg):
            cam_obj.location = (0.0, 0.0, 3.0)

    SDGPipeline(cfg, output_dir, randomizer=OverheadCamRandomizer()).run()

All bpy access is deferred to method bodies; no module-level bpy calls.
"""

import logging
import math
import time
from pathlib import Path

import numpy as np


class SDGPipeline:
    """Dependency-injectable pipeline orchestrator.

    Args:
        cfg:               Full config dict (from config.yaml).
        output_dir:        Absolute Path for all output subdirectories.
        randomizer:        Randomizer instance (or subclass).  Defaults to
                           the base Randomizer which calls the module-level
                           functions directly.
        renderer:          Renderer instance (or subclass).
        annotation_writer: AnnotationWriter instance (or subclass).
    """

    def __init__(self, cfg, output_dir, *,
                 randomizer=None, renderer=None, annotation_writer=None):
        from pipeline.randomizer        import Randomizer
        from pipeline.renderer          import Renderer
        from pipeline.annotation_writer import AnnotationWriter

        self.cfg              = cfg
        self.output_dir       = Path(output_dir)
        self.randomizer       = randomizer       or Randomizer()
        self.renderer         = renderer         or Renderer()
        self.annotation_writer = annotation_writer or AnnotationWriter()

        # Populated during run(); None until then.
        self._scene             = None
        self._cam_obj           = None
        self._BOUNDS_OBJS       = []
        self._target_pool       = None
        self._distractor_pool   = None
        self._pool_light        = None
        self._static_category_map = {}
        self._registry          = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, start: int = 0, end: int = None, debug: bool = False) -> None:
        """Run the pipeline from frame *start* (inclusive) to *end* (exclusive).

        Args:
            start: First image index (0-based).
            end:   One past the last image index.  Defaults to cfg["num_images"].
            debug: If True, renders exactly one frame (start only) and skips
                   all randomization — matches the --debug flag in run.py.
        """
        import bpy
        from pipeline.asset_registry import AssetRegistry
        from pipeline.scene_builder  import build_target_pool, build_distractor_pool

        cfg        = self.cfg
        output_dir = self.output_dir

        # ── Output directories ──────────────────────────────────────────
        dirs = ["images", "masks", "labels"]
        if cfg.get("render", {}).get("save_blends", False):
            dirs.append("blends")
        for d in dirs:
            (output_dir / d).mkdir(parents=True, exist_ok=True)

        # ── Logging ─────────────────────────────────────────────────────
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            handlers=[
                logging.FileHandler(str(output_dir / "pipeline.log")),
                logging.StreamHandler(),
            ],
        )

        # ── One-time Blender setup ───────────────────────────────────────
        self._scene     = bpy.context.scene

        # Prevent Blender from auto-saving during long batch runs.
        # Default is True; after ~2 min it writes <PID>.blend to the temp dir
        # (or cwd if temp is unset), polluting the project during batch runs.
        bpy.context.preferences.filepaths.use_auto_save_temporary_files = False
        view_layer      = self._scene.view_layers[0]
        self._registry  = AssetRegistry.from_config(cfg)

        self.renderer.configure(self._scene, cfg)
        self.renderer.enable_index_pass(view_layer)

        self._cam_obj = self._scene.camera
        bounds_col    = bpy.data.collections.get("Bounds")
        self._BOUNDS_OBJS = (
            [o for o in bounds_col.objects if o.type == "EMPTY"]
            if bounds_col else []
        )

        # ── Pre-spawn phase (once) ───────────────────────────────────────
        self._target_pool     = build_target_pool(self._registry, cfg)
        self._distractor_pool = build_distractor_pool(self._registry, cfg)
        self._pool_light      = next(
            (o for o in self._scene.objects if o.type == "LIGHT"), None
        )
        self._static_category_map = {
            obj.pass_index: obj.get("category_id", 1)
            for objs in self._target_pool.values() for obj in objs
        }

        # ── Frame range ─────────────────────────────────────────────────
        frame_end = (start + 1) if debug else (end or cfg["num_images"])

        # ── Main loop ───────────────────────────────────────────────────
        for img_idx in range(start, frame_end):
            self._run_frame(img_idx, debug=debug)

        self.annotation_writer.finalize(output_dir)
        logging.info("Done.")

    # ------------------------------------------------------------------
    # Per-frame body
    # ------------------------------------------------------------------

    def _run_frame(self, img_idx: int, debug: bool = False) -> None:
        import bpy
        from pipeline.scene_builder import (
            set_background,
            activate_frame_objects,
            activate_frame_objects_with_groups,
            destroy_group,
        )

        cfg        = self.cfg
        output_dir = self.output_dir
        scene      = self._scene

        t0  = time.perf_counter()
        rng = np.random.default_rng(cfg["seed"] + img_idx)

        # 1. Scene — activate objects for this frame
        arr_enabled = cfg["scene"].get("arrangements", {}).get("enabled", False)
        if arr_enabled:
            solo_targets, active_groups, active_distractors, id_map = (
                activate_frame_objects_with_groups(
                    self._target_pool, self._distractor_pool, rng, cfg, frame_idx=img_idx
                )
            )
            active_targets = solo_targets + [obj for g in active_groups for obj in g.members]
        else:
            active_targets, active_distractors, id_map = activate_frame_objects(
                self._target_pool, self._distractor_pool, rng, cfg
            )
            solo_targets, active_groups = active_targets, []
        bg = self._registry.sample("backgrounds", rng, n=1)
        if bg:
            set_background(bg[0].path)

        category_map = {pid: self._static_category_map[pid] for pid in id_map}  # noqa: F841
        inst_colors  = self.annotation_writer.assign_colors(id_map)

        # 2. Randomize (skipped in --debug mode)
        tex          = None
        light_info   = {}
        bg_roughness = 0.5

        if not debug:
            if self._pool_light:
                light_info = self.randomizer.randomize_light(self._pool_light, rng, cfg)
            if bg:
                bg_roughness = self.randomizer.randomize_background(
                    scene, rng, cfg, bg[0].path
                )

            tex_pool = self._registry._pools["textures"]

            t_spread    = cfg["scene"].get("target_spread", 0.5)
            max_retries = cfg["scene"].get("max_placement_retries", 10)
            margin      = float(cfg["scene"].get("placement_margin", 0.0))
            rot_x_min   = cfg["scene"].get("target_rotation_x_min", None)
            rot_x_max   = cfg["scene"].get("target_rotation_x_max", None)
            rot_y_min   = cfg["scene"].get("target_rotation_y_min", None)
            rot_y_max   = cfg["scene"].get("target_rotation_y_max", None)
            rot_z_min   = cfg["scene"].get("target_rotation_z_min", None)
            rot_z_max   = cfg["scene"].get("target_rotation_z_max", None)
            placed_aabbs: list = []

            arr_cfg = cfg["scene"].get("arrangements", {})

            # Place groups first (as rigid units), then solo targets
            for group in active_groups:
                self.randomizer.place_group(
                    group, rng, placed_aabbs,
                    spread=t_spread,
                    bounds_objs=self._BOUNDS_OBJS,
                    max_retries=arr_cfg.get("group_placement_retries", 15),
                    margin=margin,
                    rotation_x_min=rot_x_min,
                    rotation_x_max=rot_x_max,
                    rotation_y_min=rot_y_min,
                    rotation_y_max=rot_y_max,
                    rotation_z_min=rot_z_min,
                    rotation_z_max=rot_z_max,
                    rotate_as_unit=arr_cfg.get("rotate_as_unit", True),
                )
                for obj in group.members:
                    self.randomizer.randomize_material(obj, rng, cfg, texture_asset=None)

            for obj in solo_targets:
                self.randomizer.place_object(
                    obj, rng, placed_aabbs,
                    spread=t_spread,
                    bounds_objs=self._BOUNDS_OBJS,
                    randomize_scale=False,
                    max_retries=max_retries,
                    margin=margin,
                    rotation_x_min=rot_x_min,
                    rotation_x_max=rot_x_max,
                    rotation_y_min=rot_y_min,
                    rotation_y_max=rot_y_max,
                    rotation_z_min=rot_z_min,
                    rotation_z_max=rot_z_max,
                )
                self.randomizer.randomize_material(obj, rng, cfg, texture_asset=None)

            for obj in active_distractors:
                self.randomizer.place_object(
                    obj, rng, placed_aabbs,
                    bounds_objs=self._BOUNDS_OBJS,
                    randomize_scale=False,
                    max_retries=max_retries,
                    margin=margin,
                )
                tex = tex_pool[int(rng.integers(len(tex_pool)))] if tex_pool else None
                self.randomizer.randomize_material(obj, rng, cfg, texture_asset=tex)

        # In debug mode, read light info from the scene
        if debug:
            if self._pool_light:
                tgt = list(self._pool_light.get("target_loc", [0.0, 0.0, 0.0]))
                if tgt == [0.0, 0.0, 0.0]:
                    st = bpy.data.objects.get("Spot_Target")
                    if st:
                        tgt = [round(v, 8) for v in st.location]
                light_info = {
                    "Spot_Light_Location":    [round(v, 8) for v in self._pool_light.location],
                    "Light_Target_Location":  tgt,
                    "Spot_Light_Energy":      self._pool_light.data.energy,
                    "Spot_Light_Temperature": self._pool_light.get("color_temp_K", 6500),
                }
            bg_col = bpy.data.collections.get("Background")
            if bg_col:
                for obj in bg_col.objects:
                    if obj.type == "MESH" and obj.data.materials:
                        mat = obj.data.materials[0]
                        if mat and mat.use_nodes:
                            bsdf = next(
                                (n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"),
                                None,
                            )
                            if bsdf:
                                bg_roughness = float(bsdf.inputs["Roughness"].default_value)

        # Log camera and object state
        if self._cam_obj:
            logging.info(
                f"  cam_loc={tuple(round(x, 3) for x in self._cam_obj.location)} "
                f"cam_lens={round(self._cam_obj.data.lens, 1)}mm"
            )
        for iid, name in id_map.items():
            obj            = bpy.data.objects[name]
            cols           = [c.name for c in obj.users_collection]
            col_hide       = [bpy.data.collections[c].hide_render
                              for c in cols if c in bpy.data.collections]
            world_loc      = obj.matrix_world.translation
            logging.info(
                f"  target inst_{iid} '{name}' "
                f"loc={tuple(round(x, 3) for x in world_loc)} "
                f"scale={round(obj.scale[0], 3)} pass_index={obj.pass_index} "
                f"hide_render={obj.hide_render} cols={cols} col_hide_render={col_hide}"
            )

        # 3. Render — compositor writes RGB PNG + color instance mask PNG
        self.renderer.setup_compositor(scene, img_idx, output_dir, id_map, inst_colors)
        self.renderer.render(scene)
        if cfg.get("render", {}).get("save_blends", False):
            blend_path = str(output_dir / "blends" / f"{img_idx:04d}.blend")
            bpy.ops.wm.save_as_mainfile(filepath=blend_path, copy=True)

        # Build group membership lookup and capture world transforms BEFORE teardown.
        # (After destroy_group the children are reset to origin, so read positions now.)
        group_id_by_name = {}
        for g_idx, group in enumerate(active_groups):
            for obj in group.members:
                group_id_by_name[obj.name] = g_idx

        # 4. Per-image label JSON (KV format) — read world transforms while groups still exist
        label_objects = []
        for inst_id, name in id_map.items():
            obj   = bpy.data.objects[name]
            color = list(inst_colors[inst_id])
            entry = {
                "name":       name,
                # Use world-space transform (obj.location is parent-local for grouped objects)
                "position":   [round(v, 8) for v in obj.matrix_world.translation],
                "rotation":   [round(math.degrees(a), 8) for a in obj.matrix_world.to_euler()],
                "mask_color": color,
            }
            g_id = group_id_by_name.get(name)
            if g_id is not None:
                entry["group_id"] = g_id
            label_objects.append(entry)

        # Tear down groups after reading positions (restores children to hidden/unparented)
        for group in active_groups:
            destroy_group(group)
        active_groups = []

        for obj in active_distractors:
            entry = {
                "name":     obj.name,
                "position": [round(v, 8) for v in obj.matrix_world.translation],
                "rotation": [round(math.degrees(a), 8) for a in obj.matrix_world.to_euler()],
            }
            if tex:
                entry["material_image"] = tex.path.name
            label_objects.append(entry)

        label = {
            "Light_Target_Location":  light_info.get("Light_Target_Location",  [0.0, 0.0, 0.0]),
            "Spot_Light_Location":    light_info.get("Spot_Light_Location",    [0.0, 0.0, 0.0]),
            "Spot_Light_Energy":      light_info.get("Spot_Light_Energy",      100.0),
            "Spot_Light_Temperature": light_info.get("Spot_Light_Temperature", 6500),
            "Background_Roughness":   round(bg_roughness, 8),
            "Wall_Image":             bg[0].path.name if bg else "",
            "Objects":                label_objects,
        }
        self.annotation_writer.write_label(
            label, output_dir / "labels" / f"{img_idx:04d}_label.json"
        )

        elapsed = time.perf_counter() - t0
        logging.info(
            f"img={img_idx:04d} | objects={len(active_targets)} | "
            f"distractors={len(active_distractors)} | time={elapsed:.1f}s"
        )
