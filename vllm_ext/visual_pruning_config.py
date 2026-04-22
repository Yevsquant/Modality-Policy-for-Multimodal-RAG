from dataclasses import dataclass

@dataclass
class VisualPruningConfig:
    enabled: bool = False
    keep_ratio: float = 0.5
    min_keep: int = 4
    policy: str = "uniform"
    patch_grid_rows: int = 4
    patch_grid_cols: int = 4
    scorer: str = "clip"   # or "uniform"
    keep_cls_token: bool = False