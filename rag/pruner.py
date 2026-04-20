from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PruningStats:
    mode: str
    text_before: int
    text_after: int
    images_before: int
    images_after: int

    def to_dict(self) -> Dict[str, int | str]:
        return {
            "mode": self.mode,
            "text_before": self.text_before,
            "text_after": self.text_after,
            "images_before": self.images_before,
            "images_after": self.images_after,
        }


class RetrievalPruner:
    """
    Simple retrieval-side pruning baselines.

    Supported modes:
      - no_pruning: keep all retrieved text quotes and images
      - uniform_pruning: prune both modalities by the same keep ratio
      - visual_only_pruning: keep all text, prune only images by keep ratio
    """

    SUPPORTED_MODES = {"no_pruning", "uniform_pruning", "visual_only_pruning"}

    def __init__(self, mode: str = "no_pruning", keep_ratio: float = 0.5):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported pruning mode: {mode}. "
                f"Expected one of {sorted(self.SUPPORTED_MODES)}"
            )
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in the range (0, 1].")

        self.mode = mode
        self.keep_ratio = keep_ratio

    def apply(self, retrieval: Dict) -> Dict:
        text_quotes = list(retrieval.get("selected_text_quotes", []))
        img_quotes = list(retrieval.get("selected_img_quotes", []))

        pruned_texts = text_quotes
        pruned_images = img_quotes

        if self.mode == "uniform_pruning":
            pruned_texts = self._prune_list(text_quotes)
            pruned_images = self._prune_list(img_quotes)
        elif self.mode == "visual_only_pruning":
            pruned_images = self._prune_list(img_quotes)

        stats = PruningStats(
            mode=self.mode,
            text_before=len(text_quotes),
            text_after=len(pruned_texts),
            images_before=len(img_quotes),
            images_after=len(pruned_images),
        )

        return {
            "selected_text_quotes": pruned_texts,
            "selected_img_quotes": pruned_images,
            "pruning": stats.to_dict(),
        }

    def _prune_list(self, items: List[Dict]) -> List[Dict]:
        if not items:
            return []
        keep_n = max(1, int(len(items) * self.keep_ratio))
        keep_n = min(keep_n, len(items))
        return items[:keep_n]
