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
