from __future__ import annotations

import math
import re
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw, ImageChops
from transformers import CLIPModel, CLIPProcessor
import pandas as pd
import matplotlib.pyplot as plt

from rag.retriever import _clip_features_to_tensor


def _bbox_union(boxes: List[List[int]]) -> List[int]:
    """Axis-aligned bounding box [l,t,r,b] enclosing all selected tile boxes."""
    lefts = [b[0] for b in boxes]
    tops = [b[1] for b in boxes]
    rights = [b[2] for b in boxes]
    bottoms = [b[3] for b in boxes]
    return [min(lefts), min(tops), max(rights), max(bottoms)]


def _area_budget_factor(crop_area: float, full_area: float, keep_ratio: float) -> float:
    """Linear resize factor (<=1) so the crop's area is downscaled to a token
    budget of keep_ratio * full_area. Returns 1.0 when the crop is already within
    budget (never upscales). Visual tokens scale ~ pixel area."""
    target = keep_ratio * full_area
    if crop_area <= target:
        return 1.0
    return math.sqrt(target / crop_area)


def _trim_bbox(image: Image.Image, tol: int = 12) -> Tuple[int, int, int, int] | None:
    """Bounding box of non-background content, treating a near-uniform border as
    background (median of the four corners). Returns None for a uniform image.
    Query-agnostic and content-preserving: it only removes blank margins."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    corners = [rgb.getpixel((0, 0)), rgb.getpixel((w - 1, 0)),
               rgb.getpixel((0, h - 1)), rgb.getpixel((w - 1, h - 1))]
    bg = tuple(int(np.median([c[k] for c in corners])) for k in range(3))
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, bg)).convert("L")
    mask = diff.point(lambda p: 255 if p > tol else 0)
    return mask.getbbox()


def _detail_density(image: Image.Image, thumb: int = 64, ref: float = 40.0) -> float:
    """Cheap detail-density proxy in [0,1]: mean abs gradient on a small grayscale
    thumbnail, normalized by `ref`. Text/table pages score high, flat images low."""
    g = np.asarray(image.convert("L").resize((thumb, thumb)), dtype=np.float32)
    grad = (np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=0)).mean()) / 2.0
    return float(min(1.0, grad / ref))


def _density_to_keep_ratio(
    density: float, base_keep: float, lo_mult: float = 0.5, hi_mult: float = 2.0,
    max_keep: float = 1.0,
) -> float:
    """Map detail density in [0,1] to an effective keep ratio: flat images get
    base_keep*lo_mult, dense images base_keep*hi_mult (clamped to max_keep)."""
    d = min(1.0, max(0.0, density))
    mult = lo_mult + (hi_mult - lo_mult) * d
    return float(min(max_keep, max(1e-3, base_keep * mult)))


def _relevance_keep_ratios(
    relevances: List[float], base_keep: float, max_keep: float = 1.0, temp: float = 0.1,
) -> List[float]:
    """Distribute a total token budget (base_keep per image on average) across the
    retrieved images by query relevance: more-relevant images get higher resolution.
    Softmax weights keep the mean at base_keep; clamped to max_keep."""
    n = len(relevances)
    if n == 0:
        return []
    z = np.asarray(relevances, dtype=float) / max(temp, 1e-6)
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()
    krs = np.clip(base_keep * n * w, 1e-3, max_keep)
    return krs.tolist()

# def draw_plot(layer_metrics, image_cache_id, tag_hash):
#     df = pd.DataFrame(layer_metrics)

#     plt.figure()
#     plt.plot(df["layer_idx"], df["query_to_image_attention_mean"], marker="o")
#     plt.xlabel("Layer")
#     plt.ylabel("Mean Query-to-Image Attention")
#     plt.title("Query-to-Image Attention vs Layer")
#     plt.grid(True)
#     plt.savefig("query_to_image_attention.png", bbox_inches="tight")
#     plt.close()

#     plt.figure()
#     plt.plot(df["layer_idx"], df["topk_mass_ratio"], marker="o")
#     plt.xlabel("Layer")
#     plt.ylabel("Top-k Attention Mass Ratio")
#     plt.title("Top-k Attention Concentration vs Layer")
#     plt.grid(True)
#     plt.savefig("topk_attention_concentration.png", bbox_inches="tight")
#     plt.close()

#     plt.figure()
#     plt.plot(df["layer_idx"], df["attention_entropy"], marker="o")
#     plt.xlabel("Layer")
#     plt.ylabel("Attention Entropy")
#     plt.title("Attention Entropy vs Layer")
#     plt.grid(True)
#     plt.savefig("attention_entropy.png", bbox_inches="tight")
#     plt.close()

#     plt.figure()
#     plt.plot(df["layer_idx"], df["topk_overlap_with_prev_layer"], marker="o")
#     plt.xlabel("Layer")
#     plt.ylabel("Top-k Overlap with Previous Layer")
#     plt.title("Token Ranking Stability vs Layer")
#     plt.grid(True)
#     plt.savefig("token_ranking_stability.png", bbox_inches="tight")
#     plt.close()

def draw_plot(layer_metrics, icd, th):
    df = pd.DataFrame(layer_metrics)

    # Create a single figure with a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Top-Left: Query-to-Image Attention
    axs[0, 0].plot(df["layer_idx"], df["query_to_image_attention_mean"], marker="o")
    axs[0, 0].set_xlabel("Layer")
    axs[0, 0].set_ylabel("Mean Query-to-Image Attention")
    axs[0, 0].set_title("Query-to-Image Attention vs Layer")
    axs[0, 0].grid(True)

    # Top-Right: Top-k Attention Concentration
    axs[0, 1].plot(df["layer_idx"], df["topk_mass_ratio"], marker="o")
    axs[0, 1].set_xlabel("Layer")
    axs[0, 1].set_ylabel("Top-k Attention Mass Ratio")
    axs[0, 1].set_title("Top-k Attention Concentration vs Layer")
    axs[0, 1].grid(True)

    # Bottom-Left: Attention Entropy
    axs[1, 0].plot(df["layer_idx"], df["attention_entropy"], marker="o")
    axs[1, 0].set_xlabel("Layer")
    axs[1, 0].set_ylabel("Attention Entropy")
    axs[1, 0].set_title("Attention Entropy vs Layer")
    axs[1, 0].grid(True)

    # Bottom-Right: Token Ranking Stability
    axs[1, 1].plot(df["layer_idx"], df["topk_overlap_with_prev_layer"], marker="o")
    axs[1, 1].set_xlabel("Layer")
    axs[1, 1].set_ylabel("Top-k Overlap with Previous Layer")
    axs[1, 1].set_title("Token Ranking Stability vs Layer")
    axs[1, 1].grid(True)

    # Automatically adjust spacing to prevent labels from overlapping
    plt.tight_layout()

    # Save the single combined image
    plt.savefig(f"data/mmdocrag/analysis/{icd}_{th}_layer_metrics.png", bbox_inches="tight")
    plt.close()

@dataclass(frozen=True)
class PruningStats:
    mode: str
    images_before: int
    images_after: int
    visual_tokens_before: int
    visual_tokens_after: int

    def to_dict(self) -> Dict[str, int | str]:
        return {
            "mode": self.mode,
            "images_before": self.images_before,
            "images_after": self.images_after,
            "visual_tokens_before": self.visual_tokens_before,
            "visual_tokens_after": self.visual_tokens_after,
        }


class RetrievalPruner:
    """
    Retrieval-side and image-patch-side pruning.

    Modes:
      - no_pruning
      - visual_patch_pruning
      - safecrop_pruning
      - cluster_pruning
    """

    SUPPORTED_MODES = {
        "no_pruning",
        "visual_patch_pruning",
        "clip_safecrop",
        "clip_safecrop_downscale",
        "downscale_baseline",
        "trim_downscale",
        "density_adaptive_downscale",
        "relevance_adaptive_downscale",
        "safecrop_pruning",
        "cluster_pruning"
    }

    def __init__(
        self,
        mode: str = "no_pruning",
        keep_ratio: float = 0.5,
        percentile_ratio: float = 0.5,
        image_model_name: str | None = None,
        device: str = "cuda",
        patch_grid_rows: int = 4,
        patch_grid_cols: int = 4,
        min_visual_tokens: int = 4,
        montage_tile_size: int = 224,
        output_dir: str | Path = "data/mmdocrag/outputs/pruned_images",
        analysis_file: str | Path = "data/mmdocrag/analysis/layer_metrics_data.jsonl",
    ):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported pruning mode: {mode}. "
                f"Expected one of {sorted(self.SUPPORTED_MODES)}"
            )
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError("keep_ratio must be in the range (0, 1].")
        if patch_grid_rows <= 0 or patch_grid_cols <= 0:
            raise ValueError("patch_grid_rows and patch_grid_cols must be positive.")

        self.mode = mode
        self.keep_ratio = keep_ratio
        self.percentile_ratio = percentile_ratio
        self.patch_grid_rows = patch_grid_rows
        self.patch_grid_cols = patch_grid_cols
        self.min_visual_tokens = min_visual_tokens
        self.montage_tile_size = montage_tile_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_file = analysis_file

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_model_name = image_model_name
        self.clip_processor = None
        self.clip_model = None
        self.catp_cropper = None
        if mode in ("visual_patch_pruning", "clip_safecrop", "clip_safecrop_downscale"):
            if not image_model_name:
                raise ValueError(
                    "image_model_name is required for visual patch pruning modes."
                )
            self.clip_processor = CLIPProcessor.from_pretrained(image_model_name)
            self.clip_model = CLIPModel.from_pretrained(image_model_name).to(self.device)
            self.clip_model.eval()
        elif mode == "safecrop_pruning" or mode == "cluster_pruning":
            from rag.qwen2vl_catp_pruner_v2 import Qwen2VLCATPBoundingBoxCropper

            self.catp_cropper = Qwen2VLCATPBoundingBoxCropper(device=str(self.device))

    def apply(self, query: str, retrieval: Dict) -> Dict:
        text_quotes = list(retrieval.get("selected_text_quotes", []))
        img_quotes = list(retrieval.get("selected_img_quotes", []))

        pruned_texts = text_quotes
        pruned_images = img_quotes
        visual_before = sum(self._estimate_visual_tokens(q) for q in img_quotes)
        visual_after = visual_before
        if self.mode == "uniform_pruning":
            pruned_texts = self._prune_list(text_quotes)
            pruned_images = self._prune_list(img_quotes)
            visual_after = sum(self._estimate_visual_tokens(q) for q in pruned_images)
        elif self.mode == "visual_only_pruning":
            pruned_images = self._prune_list(img_quotes)
            visual_after = sum(self._estimate_visual_tokens(q) for q in pruned_images)
        elif self.mode == "visual_patch_pruning":
            processed = []
            visual_before = 0
            visual_after = 0
            for q in img_quotes:
                new_q, before_i, after_i = self._patch_prune_image(query, q)
                processed.append(new_q)
                visual_before += before_i
                visual_after += after_i
            pruned_images = processed
        elif self.mode == "clip_safecrop":
            processed = []
            visual_before = 0
            visual_after = 0
            for q in img_quotes:
                new_q, before_i, after_i = self._clip_safecrop_image(query, q)
                processed.append(new_q)
                visual_before += before_i
                visual_after += after_i
            pruned_images = processed
        elif self.mode == "clip_safecrop_downscale":
            processed = []
            visual_before = 0
            visual_after = 0
            for q in img_quotes:
                new_q, before_i, after_i = self._hybrid_crop_downscale_image(query, q)
                processed.append(new_q)
                visual_before += before_i
                visual_after += after_i
            pruned_images = processed
        elif self.mode == "downscale_baseline":
            processed = []
            visual_before = 0
            visual_after = 0
            for q in img_quotes:
                new_q, before_i, after_i = self._downscale_image(q)
                processed.append(new_q)
                visual_before += before_i
                visual_after += after_i
            pruned_images = processed
        elif self.mode == "trim_downscale":
            processed, visual_before, visual_after = [], 0, 0
            for q in img_quotes:
                new_q, before_i, after_i = self._trim_downscale_image(q)
                processed.append(new_q)
                visual_before += before_i
                visual_after += after_i
            pruned_images = processed
        elif self.mode == "density_adaptive_downscale":
            processed, visual_before, visual_after = [], 0, 0
            for q in img_quotes:
                new_q, before_i, after_i = self._density_downscale_image(q)
                processed.append(new_q)
                visual_before += before_i
                visual_after += after_i
            pruned_images = processed
        elif self.mode == "relevance_adaptive_downscale":
            relevances = [float(q.get("clip_relevance", 0.0)) for q in img_quotes]
            krs = _relevance_keep_ratios(relevances, self.keep_ratio)
            processed, visual_before, visual_after = [], 0, 0
            for q, kr in zip(img_quotes, krs):
                new_q, before_i, after_i = self._downscale_to_kr_image(q, kr)
                processed.append(new_q)
                visual_before += before_i
                visual_after += after_i
            pruned_images = processed
        elif self.mode == "safecrop_pruning" or self.mode == "cluster_pruning":
            processed = []
            visual_before = 0
            visual_after = 0
            for q in img_quotes:
                new_q, before_i, after_i, layer_metrics = self._catp_prune_image(query, q)
                processed.append(new_q)
                if layer_metrics is not None:
                    with open(self.analysis_file, 'a') as jsonl_file:
                        json_string = json.dumps(layer_metrics)
                        jsonl_file.write(json_string + '\n')
                visual_before += before_i
                visual_after += after_i
            pruned_images = processed

        stats = PruningStats(
            mode=self.mode,
            images_before=len(img_quotes),
            images_after=len(pruned_images),
            visual_tokens_before=visual_before,
            visual_tokens_after=visual_after,
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

    def _estimate_visual_tokens(self, q: Dict) -> int:
        meta = q.get("visual_pruning")
        if isinstance(meta, dict) and "tokens_after" in meta:
            return int(meta["tokens_after"])
        return self.patch_grid_rows * self.patch_grid_cols

    def _patch_prune_image(self, query: str, q: Dict) -> Tuple[Dict, int, int]:
        img_path = q.get("local_img_path")
        before = self.patch_grid_rows * self.patch_grid_cols
        after = before
        if not img_path or not Path(img_path).exists():
            q["visual_pruning"] = {
                "mode": self.mode,
                "skipped": True,
                "reason": "missing_image",
                "tokens_before": before,
                "tokens_after": after,
            }
            return q, before, after

        image = Image.open(img_path).convert("RGB")
        tiles, boxes = self._extract_grid_tiles(image)
        scores = self._score_tiles(query, tiles)

        keep_n = max(self.min_visual_tokens, int(len(tiles) * self.keep_ratio))
        keep_n = min(max(1, keep_n), len(tiles))
        keep_idx = np.argsort(-scores)[:keep_n].tolist()
        keep_idx.sort()
        after = len(keep_idx)

        q["visual_pruning"] = {
            "mode": self.mode,
            "tokens_before": before,
            "tokens_after": after,
            "tag_hash": q.get("tag_hash"),
        }

        kept_tiles = [tiles[i] for i in keep_idx]
        pruned_path = self._save_montage(image_path=Path(img_path), kept_tiles=kept_tiles, quote=q)
        q["local_img_path"] = str(pruned_path)
        q["visual_pruning"]["rendered_image_path"] = str(pruned_path)

        return q, before, after

    def _clip_safecrop_image(self, query: str, q: Dict) -> Tuple[Dict, int, int]:
        img_path = q.get("local_img_path")
        before = self.patch_grid_rows * self.patch_grid_cols
        after = before
        if not img_path or not Path(img_path).exists():
            q["visual_pruning"] = {
                "mode": self.mode,
                "skipped": True,
                "reason": "missing_image",
                "tokens_before": before,
                "tokens_after": after,
            }
            return q, before, after

        image = Image.open(img_path).convert("RGB")
        tiles, boxes = self._extract_grid_tiles(image)
        scores = self._score_tiles(query, tiles)

        keep_n = max(self.min_visual_tokens, int(len(tiles) * self.keep_ratio))
        keep_n = min(max(1, keep_n), len(tiles))
        keep_idx = np.argsort(-scores)[:keep_n].tolist()

        crop_box = _bbox_union([boxes[i] for i in keep_idx])
        cropped = image.crop(tuple(crop_box))
        pruned_path = self._save_pruned_image(
            image_path=Path(img_path), image=cropped, mode=self.mode, quote=q
        )

        # Rough area-ratio estimate; Phase 2 uses the target model processor for
        # the authoritative token count.
        area_ratio = (cropped.width * cropped.height) / max(
            1, image.width * image.height
        )
        after = max(1, round(before * area_ratio))

        q["local_img_path"] = str(pruned_path)
        q["visual_pruning"] = {
            "mode": self.mode,
            "tokens_before": before,
            "tokens_after": after,
            "crop_box": crop_box,
            "tag_hash": q.get("tag_hash"),
        }
        return q, before, after

    def _downscale_image(self, q: Dict) -> Tuple[Dict, int, int]:
        """Mediocre baseline: query-agnostic uniform downscale to the same token
        budget as the crop. Visual tokens scale ~ pixel area, so a linear edge
        factor of sqrt(keep_ratio) hits ~keep_ratio of the tokens."""
        img_path = q.get("local_img_path")
        before = self.patch_grid_rows * self.patch_grid_cols
        if not img_path or not Path(img_path).exists():
            q["visual_pruning"] = {
                "mode": self.mode,
                "skipped": True,
                "reason": "missing_image",
                "tokens_before": before,
                "tokens_after": before,
            }
            return q, before, before

        image = Image.open(img_path).convert("RGB")
        factor = math.sqrt(self.keep_ratio)
        new_size = (max(1, int(image.width * factor)), max(1, int(image.height * factor)))
        resized = image.resize(new_size)
        pruned_path = self._save_pruned_image(
            image_path=Path(img_path), image=resized, mode=self.mode, quote=q
        )
        after = max(1, round(before * self.keep_ratio))
        q["local_img_path"] = str(pruned_path)
        q["visual_pruning"] = {
            "mode": self.mode,
            "tokens_before": before,
            "tokens_after": after,
            "tag_hash": q.get("tag_hash"),
        }
        return q, before, after

    def _missing_image_meta(self, q: Dict, before: int) -> Tuple[Dict, int, int]:
        q["visual_pruning"] = {
            "mode": self.mode, "skipped": True, "reason": "missing_image",
            "tokens_before": before, "tokens_after": before,
        }
        return q, before, before

    def _finish_resized(
        self, q: Dict, img_path: str, image: Image.Image, full_area: int,
        before: int, extra: Dict | None = None,
    ) -> Tuple[Dict, int, int]:
        """Save a resized image as a single jpg and record token bookkeeping."""
        pruned_path = self._save_pruned_image(
            image_path=Path(img_path), image=image, mode=self.mode, quote=q
        )
        area_ratio = (image.width * image.height) / max(1, full_area)
        after = max(1, round(before * area_ratio))
        q["local_img_path"] = str(pruned_path)
        meta = {
            "mode": self.mode, "tokens_before": before, "tokens_after": after,
            "tag_hash": q.get("tag_hash"),
        }
        if extra:
            meta.update(extra)
        q["visual_pruning"] = meta
        return q, before, after

    def _trim_downscale_image(self, q: Dict) -> Tuple[Dict, int, int]:
        """Trim near-uniform borders (content-preserving), THEN downscale the
        trimmed image to the keep_ratio token budget. Removing blank margins first
        spends the budget on actual content at higher effective resolution."""
        img_path = q.get("local_img_path")
        before = self.patch_grid_rows * self.patch_grid_cols
        if not img_path or not Path(img_path).exists():
            return self._missing_image_meta(q, before)

        image = Image.open(img_path).convert("RGB")
        full_area = image.width * image.height
        box = _trim_bbox(image)
        trimmed = image
        if box is not None:
            cand = image.crop(box)
            # Guard: only accept a trim that removes margin but not (almost) everything.
            if 0.1 * full_area < cand.width * cand.height < 0.995 * full_area:
                trimmed = cand

        factor = _area_budget_factor(trimmed.width * trimmed.height, full_area, self.keep_ratio)
        if factor < 1.0:
            trimmed = trimmed.resize(
                (max(1, int(trimmed.width * factor)), max(1, int(trimmed.height * factor)))
            )
        return self._finish_resized(
            q, img_path, trimmed, full_area, before,
            extra={"trim_box": list(box) if box is not None else None},
        )

    def _density_downscale_image(self, q: Dict) -> Tuple[Dict, int, int]:
        """Query-agnostic: set this image's keep ratio from its detail density
        (text/table-dense pages keep resolution; flat images give tokens back),
        then downscale to that budget."""
        img_path = q.get("local_img_path")
        before = self.patch_grid_rows * self.patch_grid_cols
        if not img_path or not Path(img_path).exists():
            return self._missing_image_meta(q, before)

        image = Image.open(img_path).convert("RGB")
        full_area = image.width * image.height
        density = _detail_density(image)
        kr = _density_to_keep_ratio(density, self.keep_ratio)
        factor = math.sqrt(kr)
        resized = image.resize(
            (max(1, int(image.width * factor)), max(1, int(image.height * factor)))
        )
        return self._finish_resized(
            q, img_path, resized, full_area, before,
            extra={"density": density, "effective_keep_ratio": kr},
        )

    def _downscale_to_kr_image(self, q: Dict, keep_ratio: float) -> Tuple[Dict, int, int]:
        """Uniform downscale to an explicit per-image keep_ratio budget (used by
        relevance-adaptive allocation)."""
        img_path = q.get("local_img_path")
        before = self.patch_grid_rows * self.patch_grid_cols
        if not img_path or not Path(img_path).exists():
            return self._missing_image_meta(q, before)

        image = Image.open(img_path).convert("RGB")
        full_area = image.width * image.height
        factor = min(1.0, math.sqrt(max(1e-6, keep_ratio)))
        resized = image.resize(
            (max(1, int(image.width * factor)), max(1, int(image.height * factor)))
        )
        return self._finish_resized(
            q, img_path, resized, full_area, before,
            extra={"effective_keep_ratio": keep_ratio,
                   "clip_relevance": q.get("clip_relevance")},
        )

    def _hybrid_crop_downscale_image(self, query: str, q: Dict) -> Tuple[Dict, int, int]:
        """Query-aware crop to the relevant bounding box, THEN uniformly downscale
        that crop to the same ~keep_ratio token budget as downscale_baseline. This
        lifts clip_safecrop's token floor (its single bbox stays large on scattered
        content) while keeping the crop focused on the query-relevant region."""
        img_path = q.get("local_img_path")
        before = self.patch_grid_rows * self.patch_grid_cols
        after = before
        if not img_path or not Path(img_path).exists():
            q["visual_pruning"] = {
                "mode": self.mode,
                "skipped": True,
                "reason": "missing_image",
                "tokens_before": before,
                "tokens_after": after,
            }
            return q, before, after

        image = Image.open(img_path).convert("RGB")
        tiles, boxes = self._extract_grid_tiles(image)
        scores = self._score_tiles(query, tiles)

        keep_n = max(self.min_visual_tokens, int(len(tiles) * self.keep_ratio))
        keep_n = min(max(1, keep_n), len(tiles))
        keep_idx = np.argsort(-scores)[:keep_n].tolist()

        crop_box = _bbox_union([boxes[i] for i in keep_idx])
        cropped = image.crop(tuple(crop_box))

        full_area = image.width * image.height
        factor = _area_budget_factor(cropped.width * cropped.height, full_area, self.keep_ratio)
        if factor < 1.0:
            new_size = (max(1, int(cropped.width * factor)), max(1, int(cropped.height * factor)))
            cropped = cropped.resize(new_size)

        pruned_path = self._save_pruned_image(
            image_path=Path(img_path), image=cropped, mode=self.mode, quote=q
        )

        area_ratio = (cropped.width * cropped.height) / max(1, full_area)
        after = max(1, round(before * area_ratio))

        q["local_img_path"] = str(pruned_path)
        q["visual_pruning"] = {
            "mode": self.mode,
            "tokens_before": before,
            "tokens_after": after,
            "crop_box": crop_box,
            "downscale_factor": factor,
            "tag_hash": q.get("tag_hash"),
        }
        return q, before, after

    def _catp_prune_image(self, query: str, q: Dict) -> Tuple[Dict, int, int]:
        img_path = q.get("local_img_path")
        before = self.patch_grid_rows * self.patch_grid_cols
        after = before
        if not img_path or not Path(img_path).exists():
            q["visual_pruning"] = {
                "reason": "missing_image",
                "tokens_before": before,
                "tokens_after": after,
            }
            return q, before, after

        assert self.catp_cropper is not None
        image_path = Path(img_path)
        image = Image.open(image_path).convert("RGB")
        clusters, meta = self.catp_cropper.get_pruned_image(
            image=image,
            query=query,
            keep_ratio=self.keep_ratio,
            percentile_ratio = self.percentile_ratio,
            image_cache_id = q.get("image_cache_id"),
            tag_hash = q.get("tag_hash"),
            isCluster = self.mode == "cluster_pruning",
        )

        before = int(meta.get("tokens_before", before))
        after = int(meta.get("tokens_after", after))
        pruned_path = self._save_clusters(clusters=clusters, mode=self.mode, quote=q)

        q["local_img_path"] = str(pruned_path)
        q["visual_pruning"] = {
            "tokens_before": meta["tokens_before"],
            "tokens_after": meta["tokens_after"],
            "tokens_before_diversity": meta["tokens_before_diversity"],
            "tokens_after_diversity": meta["tokens_after_diversity"],
        }
        # (dev)
        # draw_plot(meta["layer_metrics"], q["image_cache_id"], q["tag_hash"])

        return q, before, after, meta["layer_metrics"]

    def _extract_grid_tiles(self, image: Image.Image) -> Tuple[List[Image.Image], List[List[int]]]:
        width, height = image.size
        xs = np.linspace(0, width, self.patch_grid_cols + 1, dtype=int)
        ys = np.linspace(0, height, self.patch_grid_rows + 1, dtype=int)

        tiles: List[Image.Image] = []
        boxes: List[List[int]] = []
        for r in range(self.patch_grid_rows):
            for c in range(self.patch_grid_cols):
                left, right = int(xs[c]), int(xs[c + 1])
                top, bottom = int(ys[r]), int(ys[r + 1])
                box = [left, top, right, bottom]
                tile = image.crop(box)
                tiles.append(tile)
                boxes.append(box)
        return tiles, boxes

    def _score_tiles(self, query: str, tiles: List[Image.Image]) -> np.ndarray:
        assert self.clip_model is not None and self.clip_processor is not None
        with torch.no_grad():
            text_inputs = self.clip_processor(
                text=[query], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            text_feats = _clip_features_to_tensor(
                self.clip_model.get_text_features(**text_inputs)
            )
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            image_inputs = self.clip_processor(images=tiles, return_tensors="pt", padding=True).to(self.device)
            img_feats = _clip_features_to_tensor(
                self.clip_model.get_image_features(**image_inputs)
            )
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            sims = (img_feats @ text_feats.T).squeeze(-1)
        return sims.detach().float().cpu().numpy()
    
    def _safe_filename_part(self, value: object, default: str) -> str:
        text = str(value or default)
        text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
        return text or default

    def _pruned_output_path(self, image_path: Path, mode: str, quote: Dict | None = None, is_dir: bool = False) -> Path:
        suffix = image_path.suffix or ".jpg"
        suffix = "" if is_dir else suffix
        if quote and quote.get("tag_hash"):
            image_id = self._safe_filename_part(
                quote.get("image_cache_id") or quote.get("quote_id"),
                image_path.stem,
            )
            tag_hash = self._safe_filename_part(quote.get("tag_hash"), "unknown")
            out_name = (
                f"{image_id}_tag-{tag_hash}_pruned_{mode}_"
                f"{int(self.keep_ratio * 100)}{suffix}"
            )
        else:
            out_name = f"{image_path.stem}_pruned_{mode}_{int(self.keep_ratio * 100)}{suffix}"
        return self.output_dir / out_name

    def _save_montage(self, image_path: Path, kept_tiles: List[Image.Image], quote: Dict | None = None,) -> Path:
        if not kept_tiles:
            raise ValueError("kept_tiles must not be empty.")
        n = len(kept_tiles)
        cols = min(4, n)
        rows = math.ceil(n / cols)
        tile_size = self.montage_tile_size
        canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), color=(255, 255, 255))

        for idx, tile in enumerate(kept_tiles):
            thumb = ImageOps.contain(tile, (tile_size, tile_size))
            x = (idx % cols) * tile_size + (tile_size - thumb.width) // 2
            y = (idx // cols) * tile_size + (tile_size - thumb.height) // 2
            canvas.paste(thumb, (x, y))

        out_path = self._pruned_output_path(image_path=image_path, mode=self.mode, quote=quote)
        canvas.save(out_path)
        return out_path

    def _save_pruned_image(self, image_path: Path, image: Image.Image, mode: str, quote: Dict | None = None,) -> Path:
        out_path = self._pruned_output_path(image_path=image_path, mode=mode, quote=quote)
        image.save(out_path)
        return out_path

    def _save_clusters(
        self,
        clusters: List[Dict[str, Any]],
        mode: str,
        quote: Dict | None = None,
    ) -> Path:
        cluster_path = Path(quote["local_img_path"])
        out_path_dir = self._pruned_output_path(image_path=cluster_path, mode=mode, quote=quote, is_dir=True)
        out_path_dir = Path(out_path_dir)
        out_path_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(cluster_path).convert("RGB")
        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            x_min, x_max, y_min, y_max = cluster["pixel_bbox"].values()
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            stem = str(cluster_id)
            jpg_path = out_path_dir / f"{stem}.jpg"
            json_path = out_path_dir / f"{stem}.json"
            cropped_image.save(jpg_path, format="JPEG", quality=95)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(dict(cluster), f, indent=2, ensure_ascii=False)
        return out_path_dir