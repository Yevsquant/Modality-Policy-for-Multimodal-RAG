from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ImageKeepSpec:
    keep_indices: torch.Tensor      # (K,)
    tokens_before: int
    tokens_after: int
    scores: torch.Tensor            # (T,)


class VisualTokenPruningPolicy:
    def __init__(
        self,
        keep_ratio: float,
        min_keep: int,
        policy: str = "embedding_l2_norm",
        keep_cls_token: bool = False,
        normalize_scores: bool = True,
    ):
        self.keep_ratio = keep_ratio
        self.min_keep = min_keep
        self.policy = policy
        self.keep_cls_token = keep_cls_token
        self.normalize_scores = normalize_scores

    def _resolve_keep_n(self, num_tokens: int) -> int:
        """the length of the target token seq"""
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
        keep_n = max(self.min_keep, int(num_tokens * self.keep_ratio))
        return min(max(1, keep_n), num_tokens)
    
    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.normalize_scores or scores.numel() == 0:
            return scores
        denom = scores.norm(p=2).clamp_min(1e-12)
        return scores / denom
    
    def score_embeddings(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Score visual token embeddings using their values.

        image_embeds: (T, D)
        returns: (T,)
        """
        if image_embeds.ndim != 2:
            raise ValueError(
                f"Expected image_embeds with shape (T, D), got {tuple(image_embeds.shape)}"
            )

        if self.policy == "embedding_l2_norm":
            scores = image_embeds.float().norm(p=2, dim=-1)
        elif self.policy == "embedding_abs_mean":
            scores = image_embeds.float().abs().mean(dim=-1)
        elif self.policy == "uniform":
            scores = torch.ones(
                image_embeds.shape[0],
                dtype=torch.float32,
                device=image_embeds.device,
            )
        else:
            raise ValueError(f"Unsupported visual pruning policy: {self.policy}")

        return self._normalize_scores(scores)
    
    def build_keep_spec_from_embeddings(self, image_embeds: torch.Tensor) -> ImageKeepSpec:
        scores = self.score_embeddings(image_embeds)
        num_tokens = scores.numel()
        keep_n = self._resolve_keep_n(num_tokens)

        keep = torch.topk(scores, k=keep_n, largest=True).indices.sort().values

        if self.keep_cls_token and num_tokens > 0:
            cls_idx = torch.tensor([0], device=keep.device, dtype=keep.dtype)
            keep = torch.unique(torch.cat([cls_idx, keep]), sorted=True)

        return ImageKeepSpec(
            keep_indices=keep,
            tokens_before=num_tokens,
            tokens_after=int(keep.numel()),
            scores=scores,
        )

    def prune_embeddings(self, image_embeds: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
        keep_indices = keep_indices.to(device=image_embeds.device, dtype=torch.long)
        return image_embeds.index_select(0, keep_indices)

    def soft_mask_embeddings(self, image_embeds: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
        keep_indices = keep_indices.to(device=image_embeds.device, dtype=torch.long)
        masked = torch.zeros_like(image_embeds)
        masked[keep_indices] = image_embeds[keep_indices]
        return masked

def calculate_keep_n(min_keep: int, keep_ratio: float, num_tokens: int) -> int:
    """the length of the target token seq, same as _resolve_keep_n, for external call"""
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be > 0, got {num_tokens}")
    keep_n = max(min_keep, int(num_tokens * keep_ratio))
    return min(max(1, keep_n), num_tokens)
    
def build_keep_spec(
    min_keep: int,
    keep_ratio: float,
    num_tokens: int,
    keep_cls_token: bool = False,
    device: torch.device = torch.device("cpu")
) -> ImageKeepSpec:
    """
    Generate a keep specification without needing embeddings. 
    Used in the Processor to decide the sequence length before 
    GPU allocation occurs.
    """
    if num_tokens <= 0:
        return ImageKeepSpec(
            keep_indices=torch.empty(0, device=device, dtype=torch.long),
            tokens_before=0,
            tokens_after=0,
            scores=torch.empty(0, device=device)
        )

    # 1. Resolve the budget (how many tokens to keep)
    keep_n = calculate_keep_n(min_keep, keep_ratio, num_tokens)
    
    # 2. Strategy: Uniform Sampling
    # This is position-based and requires no embedding values, 
    # making it safe to run in the processor.
    pos = torch.linspace(0, num_tokens - 1, steps=keep_n, device=device)
    keep = torch.unique(pos.round().long(), sorted=True)
    
    # Ensure the count matches keep_n exactly after unique/rounding
    if keep.numel() > keep_n:
        keep = keep[:keep_n]
        
    # 3. Optional: Force keep the CLS token if configured
    if keep_cls_token:
        cls_idx = torch.tensor([0], device=device, dtype=torch.long)
        keep = torch.unique(torch.cat([cls_idx, keep]), sorted=True)

    return ImageKeepSpec(
        keep_indices=keep,
        tokens_before=num_tokens,
        tokens_after=int(keep.numel()),
        # Processor has no embedding-based scores yet
        scores=torch.ones(num_tokens, device=device, dtype=torch.float32)
    )