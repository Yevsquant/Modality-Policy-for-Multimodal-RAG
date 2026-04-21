from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


class LocalHookedVLMRunner:
    """
    Experimental local generation path.

    Goal:
      apply visual-token pruning *inside* a locally loaded HF VLM by monkeypatching
      the vision tower forward pass and pruning patch embeddings after the encoder.

    Caveats:
      - intended for research experiments, not production
      - supports one selected image per request in the hooked path
      - works best with models exposing a recognizable vision tower that returns
        last_hidden_state with [batch, 1 + n_patches, dim]
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: str = "auto",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = self._resolve_dtype(dtype)
        # self.processor = AutoProcessor.from_pretrained(model_name)
        # self.model = AutoModelForVision2Seq.from_pretrained(
        #     model_name,
        #     torch_dtype=self.torch_dtype,
        # ).to(self.device)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            enable_audio_output=False,
            # attn_implementation="flash_attention_2",
        )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        image_path: str,
        visual_pruning: Dict,
        max_new_tokens: int = 128,
    ) -> Dict:
        path = Path(image_path)
        if not path.exists():
            return {"applied": False, "reason": "missing_image"}

        image = Image.open(path).convert("RGB")
        model_inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        model_inputs = {
            k: (v.to(self.device) if hasattr(v, "to") else v)
            for k, v in model_inputs.items()
        }

        tower = self._find_vision_tower()
        if tower is None:
            return {
                "applied": False,
                "reason": "vision_tower_not_found",
                "model_type": type(self.model).__name__,
            }

        hook_stats: Dict = {}
        orig_forward = tower.forward

        def wrapped_forward(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            pruned_out, stats = self._prune_forward_output(out, visual_pruning)
            hook_stats.update(stats)
            return pruned_out

        try:
            tower.forward = wrapped_forward
            with torch.no_grad():
                generated = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        finally:
            tower.forward = orig_forward

        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return {
            "applied": bool(hook_stats.get("applied")),
            "hook_stats": hook_stats,
            "generated_text": text,
        }

    def _prune_forward_output(self, out, visual_pruning: Dict):
        if not hasattr(out, "last_hidden_state"):
            return out, {"applied": False, "reason": "no_last_hidden_state"}

        hidden = out.last_hidden_state
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 2:
            return out, {"applied": False, "reason": "unexpected_hidden_shape", "shape": list(hidden.shape)}

        patch_hidden = hidden[:, 1:, :]
        n_patches = int(patch_hidden.shape[1])
        patch_grid = int(round(n_patches ** 0.5))
        if patch_grid * patch_grid != n_patches:
            return out, {"applied": False, "reason": "non_square_patch_grid", "tokens_before": n_patches}

        mapped = self._map_coarse_to_patch_indices(
            coarse_rows=int(visual_pruning.get("grid_rows", 1)),
            coarse_cols=int(visual_pruning.get("grid_cols", 1)),
            patch_grid=patch_grid,
            keep_indices=list(visual_pruning.get("keep_indices", [])),
        )
        if not mapped:
            return out, {"applied": False, "reason": "empty_keep_indices", "tokens_before": n_patches}

        keep_tensor = torch.tensor(mapped, device=hidden.device, dtype=torch.long)
        pruned = torch.cat(
            [hidden[:, :1, :], patch_hidden.index_select(1, keep_tensor)],
            dim=1,
        )

        out.last_hidden_state = pruned
        if hasattr(out, "hidden_states") and out.hidden_states:
            hs = list(out.hidden_states)
            hs[-1] = pruned
            out.hidden_states = tuple(hs)

        return out, {
            "applied": True,
            "hook_point": "after_vision_encoder",
            "tokens_before": n_patches,
            "tokens_after": int(pruned.shape[1] - 1),
            "mapped_patch_indices": mapped,
        }

    def _find_vision_tower(self):
        candidates = [
            getattr(self.model, "vision_tower", None),
            getattr(getattr(self.model, "model", None), "vision_tower", None),
            getattr(self.model, "visual", None),
            getattr(getattr(self.model, "model", None), "visual", None),
        ]
        get_vision_tower = getattr(self.model, "get_vision_tower", None)
        if callable(get_vision_tower):
            try:
                candidates.insert(0, get_vision_tower())
            except Exception:
                pass

        for c in candidates:
            if c is not None and hasattr(c, "forward"):
                return c
        return None

    def _resolve_dtype(self, dtype: str):
        if dtype == "auto":
            return torch.float16 if self.device.type == "cuda" else torch.float32
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    def _map_coarse_to_patch_indices(
        self,
        coarse_rows: int,
        coarse_cols: int,
        patch_grid: int,
        keep_indices: List[int],
    ) -> List[int]:
        row_edges = [round(i * patch_grid / coarse_rows) for i in range(coarse_rows + 1)]
        col_edges = [round(i * patch_grid / coarse_cols) for i in range(coarse_cols + 1)]
        out: List[int] = []
        seen = set()
        for idx in keep_indices:
            r = idx // coarse_cols
            c = idx % coarse_cols
            if r >= coarse_rows or c >= coarse_cols:
                continue
            for pr in range(row_edges[r], row_edges[r + 1]):
                for pc in range(col_edges[c], col_edges[c + 1]):
                    patch_idx = pr * patch_grid + pc
                    if patch_idx not in seen:
                        seen.add(patch_idx)
                        out.append(patch_idx)
        return out
