from dataclasses import dataclass


@dataclass
class VisualPruningConfig:
    enabled: bool = True
    keep_ratio: float = 0.1
    min_keep: int = 4

    # Embedding-based policies run after the vision encoder produces per-token
    # image embeddings. The processor should not choose keep indices anymore.
    policy: str = "embedding_l2_norm"  # {"embedding_l2_norm", "embedding_abs_mean", "uniform"}

    # Optional behavior knobs.
    keep_cls_token: bool = False
    normalize_scores: bool = True

    # Whether to shrink the visual token sequence length after pruning.
    # This is the desired research mode, but exact runtime support still depends
    # on the server path honoring post-embedding feature lengths.
    hard_prune: bool = True

    # Fallback mode if the runtime cannot safely shrink placeholder-aligned
    # multimodal sequence lengths after embedding-time pruning.
    # When True, dropped embeddings are zeroed instead of removed.
    fallback_to_soft_mask: bool = False
