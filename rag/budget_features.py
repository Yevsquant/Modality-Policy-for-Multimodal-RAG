"""CLIP features for the learned budget policy (Phase 2).

Cheap, query-conditioned features for the "small model at the very end": the same
`openai/clip-vit-base-patch32` the retriever already loads (`rag/retriever.py`).
Per (image, question) we concatenate the L2-normalized CLIP image embedding, the
CLIP text embedding of the question, and their cosine similarity:

    feat = [ img_emb (512) | txt_emb (512) | cos(img, txt) (1) ]  -> 1025 dims

No 7B, no full-attention capture — this is the entire point of the repositioning
("do less work; a small model at the end is enough").
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

_CLIP_NAME = "openai/clip-vit-base-patch32"


class ClipFeaturizer:
    def __init__(self, model_name: str = _CLIP_NAME, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    @staticmethod
    def _to_tensor(raw):
        import torch

        if isinstance(raw, torch.Tensor):
            return raw
        po = getattr(raw, "pooler_output", None)
        if po is not None:
            return po
        raise TypeError(f"Unexpected CLIP feature output type: {type(raw)}")

    def _img_emb(self, paths: Sequence[str]) -> np.ndarray:
        from PIL import Image

        imgs = [Image.open(Path(p)).convert("RGB") for p in paths]
        with self._torch.no_grad():
            inp = self.processor(images=imgs, return_tensors="pt", padding=True).to(
                self.device
            )
            feats = self._to_tensor(self.model.get_image_features(**inp))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    def _txt_emb(self, texts: Sequence[str]) -> np.ndarray:
        with self._torch.no_grad():
            inp = self.processor(
                text=list(texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            feats = self._to_tensor(self.model.get_text_features(**inp))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    def features(
        self, image_paths: Sequence[str], questions: Sequence[str], batch_size: int = 32
    ) -> np.ndarray:
        """[n, 1025] feature matrix: [img_emb | txt_emb | cos_sim]."""
        assert len(image_paths) == len(questions)
        img_chunks: List[np.ndarray] = []
        txt_chunks: List[np.ndarray] = []
        for s in range(0, len(image_paths), batch_size):
            img_chunks.append(self._img_emb(image_paths[s : s + batch_size]))
            txt_chunks.append(self._txt_emb(questions[s : s + batch_size]))
        img = np.concatenate(img_chunks, axis=0)
        txt = np.concatenate(txt_chunks, axis=0)
        return assemble_features(img, txt)


def assemble_features(img_emb: np.ndarray, txt_emb: np.ndarray) -> np.ndarray:
    """Concatenate normalized image+text embeddings with their cosine similarity.

    Pure (no torch) so it is unit-testable. cos == dot product since inputs are L2-norm.
    """
    cos = np.sum(img_emb * txt_emb, axis=1, keepdims=True).astype("float32")
    return np.concatenate([img_emb, txt_emb, cos], axis=1).astype("float32")


# ---------------------------------------------------------------------------
# Phase 4: SPATIAL features for the budget policy.
#
# Phase 2's pooled CLIP carried ~0 signal about downscale sensitivity (OOF AUC
# ~0.50). Hypothesis: sensitivity is driven by whether a SMALL query-relevant
# region exists that downscaling destroys — a spatial property the pooled
# embedding discards. So we use the per-patch CLIP ViT tokens (NOT the pooled
# vector): cosine of each patch to the question's CLIP text embedding gives a
# spatial query-relevance map, from which we derive scalar peakiness / spread /
# off-center features. We add cheap detail-density (reusing pruner._detail_density)
# and native-resolution features. Still cheap — one CLIP forward, no 7B.
# ---------------------------------------------------------------------------

# Names of the spatial scalar features, in the order assemble_spatial_features emits.
SPATIAL_FEATURE_NAMES = [
    "rel_max",          # peak patch relevance (concentration)
    "rel_mean",         # mean patch relevance
    "rel_std",          # spread of relevance values
    "rel_top1_mass",    # fraction of total (softmax) relevance in the single peak patch
    "rel_top5_mass",    # fraction in the top-5 patches (is the signal concentrated?)
    "rel_entropy",      # normalized entropy of the softmax map (1=diffuse, 0=peaky)
    "rel_active_frac",  # fraction of patches above (mean+std): size of high-rel region
    "peak_dist_center", # L2 distance of the peak patch from the grid center, normalized
    "peak_spread",      # weighted spatial std of the relevance map (region size)
    "detail_global",    # _detail_density of the whole image
    "detail_peak",      # _detail_density of the crop around the peak-relevance region
    "log_area",         # log10(native pixel area) — bigger images stress downscaling more
    "aspect",           # max(w,h)/min(w,h) aspect ratio
]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def spatial_features_from_map(
    rel: np.ndarray, grid_hw: Tuple[int, int]
) -> np.ndarray:
    """Scalar peakiness/spread/off-center features from a patch-relevance map.

    `rel` is a flat 1-D array of per-patch cosine similarities to the query text
    embedding, laid out row-major on a `grid_hw = (rows, cols)` grid. Pure (numpy
    only) so it is unit-testable without a CLIP forward. Returns the first 9 entries
    of SPATIAL_FEATURE_NAMES (the map-derived ones); detail/resolution features are
    appended by the featurizer which has the PIL image.
    """
    rows, cols = grid_hw
    rel = np.asarray(rel, dtype=np.float64).reshape(rows, cols)
    flat = rel.reshape(-1)

    rel_max = float(flat.max())
    rel_mean = float(flat.mean())
    rel_std = float(flat.std())

    # Softmax turns raw cosines into a probability mass we can concentrate-measure.
    p = _softmax(flat * 10.0)  # temperature sharpens; cosines are small in magnitude
    order = np.argsort(p)[::-1]
    rel_top1_mass = float(p[order[0]])
    rel_top5_mass = float(p[order[: min(5, p.size)]].sum())
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    rel_entropy = ent / math.log(p.size) if p.size > 1 else 0.0

    thresh = rel_mean + rel_std
    rel_active_frac = float((flat > thresh).mean())

    # Peak location & spread on the grid.
    peak_idx = int(np.argmax(flat))
    pr, pc = divmod(peak_idx, cols)
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    diag = math.sqrt(cr * cr + cc * cc) or 1.0
    peak_dist_center = math.sqrt((pr - cr) ** 2 + (pc - cc) ** 2) / diag

    # Mass-weighted spatial std of the relevance map (size of the relevant region).
    ys, xs = np.divmod(np.arange(flat.size), cols)
    my = float(np.sum(p * ys))
    mx = float(np.sum(p * xs))
    var = float(np.sum(p * ((ys - my) ** 2 + (xs - mx) ** 2)))
    peak_spread = math.sqrt(var) / diag

    return np.array(
        [
            rel_max, rel_mean, rel_std, rel_top1_mass, rel_top5_mass,
            rel_entropy, rel_active_frac, peak_dist_center, peak_spread,
        ],
        dtype="float32",
    )


class SpatialClipFeaturizer:
    """Per-patch CLIP relevance map + scalar spatial/detail/resolution features.

    Uses the same `openai/clip-vit-base-patch32`, but reads the per-patch ViT tokens
    (`vision_model.last_hidden_state[:, 1:]`, the 7x7=49 patches at 224px), projects
    them to the shared space, and computes cosine to the question text embedding.
    """

    GRID = (7, 7)  # 224 / 32

    def __init__(self, model_name: str = _CLIP_NAME, device: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()

    def _patch_relevance(self, img, question: str) -> np.ndarray:
        """Flat 49-vector of per-patch cosine to the question text embedding."""
        torch = self._torch
        with torch.no_grad():
            pin = self.processor(images=[img], return_tensors="pt").to(self.device)
            vout = self.model.vision_model(pixel_values=pin["pixel_values"])
            patches = vout.last_hidden_state[:, 1:, :]  # drop CLS -> [1,49,768]
            patch_emb = self.model.visual_projection(patches)[0]  # [49,512]
            patch_emb = patch_emb / patch_emb.norm(dim=-1, keepdim=True)

            tin = self.processor(
                text=[question], return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            temb = ClipFeaturizer._to_tensor(self.model.get_text_features(**tin))
            temb = temb / temb.norm(dim=-1, keepdim=True)

            rel = (patch_emb @ temb[0]).cpu().numpy().astype("float32")  # [49]
        return rel

    def _detail_features(self, img) -> np.ndarray:
        from rag.pruner import _detail_density

        rel = self._last_rel  # set by features(); peak region for detail_peak
        rows, cols = self.GRID
        peak = int(np.argmax(rel))
        pr, pc = divmod(peak, cols)
        w, h = img.size
        # 3x3 patch window around the peak -> crop box in pixel coords
        r0, r1 = max(0, pr - 1), min(rows, pr + 2)
        c0, c1 = max(0, pc - 1), min(cols, pc + 2)
        box = (
            int(c0 / cols * w), int(r0 / rows * h),
            int(c1 / cols * w), int(r1 / rows * h),
        )
        crop = img.crop(box) if box[2] > box[0] and box[3] > box[1] else img
        detail_global = _detail_density(img)
        detail_peak = _detail_density(crop)
        area = float(w * h)
        log_area = math.log10(max(1.0, area))
        aspect = max(w, h) / max(1, min(w, h))
        return np.array(
            [detail_global, detail_peak, log_area, aspect], dtype="float32"
        )

    def features(
        self, image_paths: Sequence[str], questions: Sequence[str]
    ) -> np.ndarray:
        """[n, len(SPATIAL_FEATURE_NAMES)] scalar feature matrix."""
        from PIL import Image

        assert len(image_paths) == len(questions)
        rows = []
        for p, q in zip(image_paths, questions):
            img = Image.open(Path(p)).convert("RGB")
            rel = self._patch_relevance(img, q)
            self._last_rel = rel
            map_feats = spatial_features_from_map(rel, self.GRID)
            det_feats = self._detail_features(img)
            rows.append(np.concatenate([map_feats, det_feats]).astype("float32"))
        return np.stack(rows).astype("float32")
