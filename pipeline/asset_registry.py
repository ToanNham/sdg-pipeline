from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class Asset:
    path: Path
    category_id: int = 0
    category_name: str = ""
    has_embedded_textures: bool = False  # True for .glb — skip external texture randomization
    tags: list = field(default_factory=list)


class AssetRegistry:
    POOLS = ["models", "distractors", "textures", "backgrounds"]

    def __init__(self):
        self._pools = {p: [] for p in self.POOLS}

    def register(self, pool: str, asset: Asset):
        assert pool in self.POOLS, f"Unknown pool: {pool}"
        self._pools[pool].append(asset)

    def sample(self, pool: str, rng: np.random.Generator,
               n: int = 1, replace: bool = False) -> list:
        candidates = self._pools[pool]
        if not candidates:
            return []
        n_actual = n if replace else min(n, len(candidates))
        idx = rng.choice(len(candidates), size=n_actual, replace=replace)
        return [candidates[i] for i in idx]

    def count(self, pool: str) -> int:
        return len(self._pools[pool])

    @classmethod
    def from_config(cls, cfg) -> "AssetRegistry":
        reg = cls()

        for entry in cfg["assets"]["models"]:
            for p in Path(entry["path"]).glob(entry.get("glob", "*.glb")):
                reg.register("models", Asset(
                    path=p,
                    category_id=entry["category_id"],
                    category_name=entry["category_name"],
                    has_embedded_textures=p.suffix.lower() == ".glb"
                ))

        mesh_dir = cfg["assets"]["distractors"].get("meshes")
        if mesh_dir:
            for ext in ("*.glb", "*.obj"):
                for p in Path(mesh_dir).glob(ext):
                    reg.register("distractors", Asset(
                        path=p,
                        has_embedded_textures=p.suffix.lower() == ".glb"
                    ))

        tex_dir = cfg["assets"].get("textures")
        if tex_dir:
            for p in Path(tex_dir).iterdir():
                if p.is_dir():
                    reg.register("textures", Asset(path=p))
            for ext in ("**/*.jpg", "**/*.jpeg", "**/*.png"):
                for p in Path(tex_dir).glob(ext):
                    reg.register("textures", Asset(path=p))

        bg_dir = cfg["assets"].get("backgrounds")
        if bg_dir:
            for ext in ("**/*.jpg", "**/*.jpeg", "**/*.png"):
                for p in Path(bg_dir).glob(ext):
                    reg.register("backgrounds", Asset(path=p))

        return reg


if __name__ == "__main__":
    import yaml

    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    registry = AssetRegistry.from_config(cfg)
    for pool in AssetRegistry.POOLS:
        print(f"{pool}: {registry.count(pool)} assets")
