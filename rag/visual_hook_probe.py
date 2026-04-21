from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

@dataclass
class HookProbeStats:
    image_path: str
    tokens_before: int
    tokens_after: int
    hidden_size: int
    keep_indices: List[int]
    cls_kept: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "image_path": self.image_path,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "hidden_size": self.hidden_size,
            "keep_indices": self.keep_indices,
            "cls_kept": self.cls_kept,
        }


class VisualTokenHookProbe:
    """
    Local probe for a true model-side visual token pruning point.

    It runs a vision encoder locally, then prunes patch embeddings *after* the
    encoder output using keep_indices produced by the pruner. This does not
    affect remote server generation, but it does measure the exact hidden-state
    reduction at the hook point.
    """

    def __init__(self, image_model_name: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(image_model_name)
        self.model = CLIPModel.from_pretrained(image_model_name).to(self.device)
        self.model.eval()

    def run_on_image(
        self,
        image_path: str,
        visual_pruning: Dict,
    ) -> Dict:
        path = Path(image_path)
        if not path.exists():
            return {"applied": False, "reason": "missing_image"}

        image = Image.open(path).convert("RGB")
        with torch.no_grad():
            proc = self.processor(images=image, return_tensors="pt")
            pixel_values = proc["pixel_values"].to(self.device)
            out = self.model.vision_model(pixel_values=pixel_values)
            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state  # [1, 1 + n_patches, dim]
            elif isinstance(out, tuple) and len(out) > 0:
                hidden = out[0]
            else:
                raise TypeError("vision_model output missing last_hidden_state")

        cls_tok = hidden[:, :1, :]
        patch_toks = hidden[:, 1:, :]
        n_patches = patch_toks.shape[1]
        patch_grid = int(math.sqrt(n_patches))
        if patch_grid * patch_grid != n_patches:
            return {
                "applied": False,
                "reason": "non_square_patch_grid",
                "tokens_before": int(n_patches),
            }

        coarse_rows = int(visual_pruning.get("grid_rows", 1))
        coarse_cols = int(visual_pruning.get("grid_cols", 1))
        keep_indices = list(visual_pruning.get("keep_indices", []))
        mapped_patch_indices = self._map_coarse_to_patch_indices(
            coarse_rows=coarse_rows,
            coarse_cols=coarse_cols,
            patch_grid=patch_grid,
            keep_indices=keep_indices,
        )
        if not mapped_patch_indices:
            return {
                "applied": False,
                "reason": "empty_keep_indices",
                "tokens_before": int(n_patches),
            }

        keep_tensor = torch.tensor(mapped_patch_indices, device=patch_toks.device, dtype=torch.long)
        pruned_patch_toks = patch_toks.index_select(1, keep_tensor)
        pruned_hidden = torch.cat([cls_tok, pruned_patch_toks], dim=1)

        return {
            "applied": True,
            "hook_point": "after_vision_encoder",
            "tokens_before": int(n_patches),
            "tokens_after": int(pruned_patch_toks.shape[1]),
            "hidden_size": int(hidden.shape[-1]),
            "full_hidden_shape": list(hidden.shape),
            "pruned_hidden_shape": list(pruned_hidden.shape),
            "coarse_keep_indices": keep_indices,
            "mapped_patch_indices": mapped_patch_indices,
        }

    def _map_coarse_to_patch_indices(
        self,
        coarse_rows: int,
        coarse_cols: int,
        patch_grid: int,
        keep_indices: Sequence[int],
    ) -> List[int]:
        patch_ids: List[int] = []
        row_edges = [round(i * patch_grid / coarse_rows) for i in range(coarse_rows + 1)]
        col_edges = [round(i * patch_grid / coarse_cols) for i in range(coarse_cols + 1)]

        for idx in keep_indices:
            r = idx // coarse_cols
            c = idx % coarse_cols
            if r >= coarse_rows or c >= coarse_cols:
                continue
            for pr in range(row_edges[r], row_edges[r + 1]):
                for pc in range(col_edges[c], col_edges[c + 1]):
                    patch_ids.append(pr * patch_grid + pc)

        seen = set()
        out: List[int] = []
        for p in patch_ids:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out
