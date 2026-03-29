import sys, site
sys.path.insert(0, site.getusersitepackages())
import argparse, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import SDGPipeline


def validate_config(cfg):
    """Check required config fields before any rendering begins."""
    for key in ("render", "scene", "camera", "lighting", "material", "assets", "output_dir", "num_images", "seed"):
        if key not in cfg:
            print(f"ConfigError: missing required key '{key}'")
            sys.exit(1)

    device = cfg["render"].get("device", "")
    if device not in ("GPU", "CPU"):
        print(f"ConfigError: render.device must be 'GPU' or 'CPU', got {device!r}")
        sys.exit(1)

    for field in ("samples", "resolution_x", "resolution_y"):
        val = cfg["render"].get(field)
        if not isinstance(val, int) or val <= 0:
            print(f"ConfigError: render.{field} must be a positive integer, got {val!r}")
            sys.exit(1)

    num_images = cfg.get("num_images")
    if not isinstance(num_images, int) or num_images <= 0:
        print(f"ConfigError: num_images must be a positive integer, got {num_images!r}")
        sys.exit(1)

    models = cfg.get("assets", {}).get("models", [])
    if not models:
        print("ConfigError: assets.models must contain at least one entry")
        sys.exit(1)

    for i, m in enumerate(models):
        for field in ("path", "category_id", "category_name"):
            if field not in m:
                print(f"ConfigError: assets.models[{i}] missing required field '{field}'")
                sys.exit(1)


argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--start",  type=int, default=0)
parser.add_argument("--end",    type=int, default=None)
parser.add_argument("--debug",  action="store_true",
                    help="Render 1 image, skip all randomization, force object visibility")
args = parser.parse_args(argv)

with open(args.config) as f:
    cfg = yaml.safe_load(f)

validate_config(cfg)
output_dir = (Path(args.config).parent / cfg["output_dir"]).resolve()
SDGPipeline(cfg, output_dir).run(start=args.start, end=args.end, debug=args.debug)
