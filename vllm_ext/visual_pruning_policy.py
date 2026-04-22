from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ImageKeepSpec:
    keep_indices: torch.Tensor      # (K,)
    tokens_before: int
    tokens_after: int


class VisualTokenPruningPolicy:
    def __init__(self, keep_ratio: float, min_keep: int):
        self.keep_ratio = keep_ratio
        self.min_keep = min_keep

    def _resolve_keep_n(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
        keep_n = max(self.min_keep, int(num_tokens * self.keep_ratio))
        return min(max(1, keep_n), num_tokens)

    def build_uniform_keep_spec(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> ImageKeepSpec:
        keep_n = self._resolve_keep_n(num_tokens)
        if keep_n == num_tokens:
            keep = torch.arange(num_tokens, device=device, dtype=torch.long)
        else:
            # evenly spaced keep
            pos = torch.linspace(0, num_tokens - 1, steps=keep_n, device=device)
            keep = torch.unique(pos.round().long(), sorted=True)
            if keep.numel() < keep_n:
                full = torch.arange(num_tokens, device=device, dtype=torch.long)
                mask = torch.ones(num_tokens, device=device, dtype=torch.bool)
                mask[keep] = False
                extra = full[mask][: keep_n - keep.numel()]
                keep = torch.cat([keep, extra]).sort().values

        return ImageKeepSpec(
            keep_indices=keep,
            tokens_before=num_tokens,
            tokens_after=int(keep.numel()),
        )

    def build_score_keep_spec(
        self,
        token_scores: torch.Tensor,   # (T,)
    ) -> ImageKeepSpec:
        num_tokens = token_scores.numel()
        keep_n = self._resolve_keep_n(num_tokens)

        keep = torch.topk(token_scores, k=keep_n, largest=True).indices.sort().values
        return ImageKeepSpec(
            keep_indices=keep,
            tokens_before=num_tokens,
            tokens_after=int(keep.numel()),
        )