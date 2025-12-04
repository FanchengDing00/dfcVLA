import dataclasses
from typing import Optional

@dataclasses.dataclass(frozen=True)
class VGGTConfig:
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    enable_camera: bool = False
    enable_point: bool = False
    enable_depth: bool = False
    enable_track: bool = False
    feature_only: bool = True
    weight_path: Optional[str] = None