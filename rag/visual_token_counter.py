from __future__ import annotations

import logging

from PIL import Image
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


class VisualTokenCounter:
    """Count the visual tokens an image costs on the *target* generation model.

    Loads only the model's image processor config (not the model weights), so it
    is cheap and CPU-only. Token count = prod(image_grid_thw) // merge_size**2,
    matching how the Qwen vision encoder merges patches.
    """

    def __init__(self, model_name: str):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.image_processor = self.processor.image_processor
        self.merge_size = int(getattr(self.image_processor, "merge_size", 2))

    def count(self, image: Image.Image) -> int:
        rgb = image.convert("RGB")
        out = self.image_processor(images=[rgb], return_tensors="pt")
        grid = out.get("image_grid_thw") if hasattr(out, "get") else out["image_grid_thw"]
        if grid is None:
            # Fallback: estimate from pixel area when the processor does not
            # expose a Qwen-style grid (e.g. some omni variants).
            patch = 14 * self.merge_size
            tokens = (rgb.width // patch) * (rgb.height // patch)
            logger.warning(
                "image_grid_thw unavailable; estimating %d tokens from pixel area", tokens
            )
            return max(1, int(tokens))
        t, h, w = grid[0].tolist()
        return int(t * h * w // (self.merge_size ** 2))

    def count_path(self, path: str) -> int:
        return self.count(Image.open(path))
