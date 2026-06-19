"""Faithful FastV-style in-model visual-token pruning for Qwen2-VL-7B (HF).

FastV (Chen et al., 2024): run the first K decoder layers on the full sequence,
then at layer K drop the visual tokens that receive the least attention and run the
remaining layers on the smaller set. This puts the pruning *inside* the answer model
(unlike the project's input-side crops), and on the SAME 7B that produces the answer
(unlike the old 30B-vs-7B mismatch).

Implementation (single prefill, multiple-choice scoring — no generation/KV cache):
  1. One full forward with output_attentions + output_hidden_states. `hidden_states[K]`
     is the input to layer K; `attentions[K-1]` is the last full layer's attention,
     used to rank image tokens (attention received from the final query token).
  2. Keep all text tokens + the top-`keep_ratio` image tokens; gather `hidden_states[K]`
     and the rotary `position_embeddings` (captured via a hook) at those indices, build
     a fresh causal mask, and run `layers[K:]` + norm + lm_head on the pruned set.
  3. Score by the argmax over the {A,B,C,D} option-letter logits at the last position.

`fastv_layer=None` is the no-in-model-pruning baseline (use the full forward's logits).
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from rag.image_ops import downscale_to_keep

_DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
# "Full" reference resolution (~1024 visual tokens; tokens ~= pixels / 28^2) so the
# input_keep axis is a real downscale below it (not re-clamped by the processor) and
# full-forward attention capture stays memory-bounded.
_DEFAULT_BASE_PIXELS = int(os.environ.get("MRAG_FASTV_BASE_PIXELS", 1024 * 28 * 28))


def _fit_area(image: Image.Image, target_pixels: int) -> Image.Image:
    """Downscale so the image has at most `target_pixels` (never upscale)."""
    area = image.width * image.height
    if area <= target_pixels:
        return image
    f = (target_pixels / area) ** 0.5
    return image.resize((max(1, int(image.width * f)), max(1, int(image.height * f))))


def select_keep_indices(
    attn_to_image: torch.Tensor,
    image_positions: torch.Tensor,
    seq_len: int,
    keep_ratio: float,
) -> torch.Tensor:
    """Indices to keep after FastV pruning: all non-image tokens + the top
    `keep_ratio` image tokens by `attn_to_image` (one score per image token, aligned
    with `image_positions`). Returns a sorted LongTensor (causal order preserved)."""
    n_img = image_positions.numel()
    n_keep = max(1, int(round(keep_ratio * n_img)))
    if n_keep >= n_img:
        keep_img = image_positions
    else:
        top = torch.topk(attn_to_image, n_keep).indices
        keep_img = image_positions[top]
    is_image = torch.zeros(seq_len, dtype=torch.bool, device=image_positions.device)
    is_image[image_positions] = True
    non_image = torch.nonzero(~is_image, as_tuple=True)[0]
    keep = torch.cat([non_image, keep_img])
    return torch.sort(keep).values


class FastVQwen2VL:
    def __init__(self, model_id: str = _DEFAULT_MODEL, base_pixels: int = _DEFAULT_BASE_PIXELS):
        self.base_pixels = base_pixels
        self.processor = AutoProcessor.from_pretrained(model_id)
        # GPTQ kernels are fp16-locked. fp16 *eager* attention overflows to NaN over
        # ~1k tokens, so SDPA is used for the stable forward / deep re-run, and eager
        # is switched on only for the shallow ranking forward (whose layer-K attention
        # and hidden_states[K] are finite). See _set_attn / answer_mc.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="cuda",
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        # text backbone pieces used for the manual deep-layer re-run
        self.text_model = self.model.model.language_model
        self.lm_head = self.model.lm_head
        self.image_token_id = getattr(self.model.config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # capture rotary position_embeddings (cos, sin) from the full forward
        self._pos_emb = None
        self.text_model.rotary_emb.register_forward_hook(
            lambda mod, inp, out: setattr(self, "_pos_emb", out)
        )
        # option-letter token ids
        self.letter_ids = {
            L: self.processor.tokenizer.encode(L, add_special_tokens=False)[0]
            for L in "ABCDEFGH"
        }

    def _set_attn(self, impl: str):
        """Switch the attention implementation at runtime (eager exposes per-token
        attention for ranking; sdpa is numerically stable in fp16)."""
        self.model.config._attn_implementation = impl
        for m in self.model.modules():
            cfg = getattr(m, "config", None)
            if cfg is not None:
                cfg._attn_implementation = impl

    def _build_inputs(self, image: Image.Image, question: str, input_keep: float = 1.0):
        # Fit to the base reference resolution, then apply the input-level downscale,
        # so input_keep genuinely reduces the visual-token count.
        image = downscale_to_keep(_fit_area(image, self.base_pixels), input_keep)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        return inputs.to(self.device)

    @torch.no_grad()
    def _deep_rerun(self, hidden_K, keep_idx, layer_K):
        """Run layers[layer_K:] + norm + lm_head on the gathered (pruned) sequence;
        return last-position logits."""
        h = hidden_K[:, keep_idx, :]
        cos, sin = self._pos_emb
        # cos/sin: [..., seq, head_dim] (M-RoPE prepends a 3-dim leading axis)
        cos_g = cos.index_select(-2, keep_idx)
        sin_g = sin.index_select(-2, keep_idx)
        Lp = h.shape[1]
        mask = torch.full((1, 1, Lp, Lp), torch.finfo(h.dtype).min, device=h.device, dtype=h.dtype)
        mask = torch.triu(mask, diagonal=1)
        for layer in self.text_model.layers[layer_K:]:
            h = layer(h, attention_mask=mask, position_embeddings=(cos_g, sin_g),
                      position_ids=None, past_key_values=None, use_cache=False)
        h = self.text_model.norm(h)
        return self.lm_head(h[:, -1, :]).float().squeeze(0)

    @torch.no_grad()
    def answer_mc(self, image: Image.Image, question: str, num_choices: int = 4,
                  input_keep: float = 1.0, fastv_layer: Optional[int] = None,
                  keep_ratio: float = 1.0):
        inputs = self._build_inputs(image, question, input_keep)
        input_ids = inputs["input_ids"][0]
        seq_len = input_ids.numel()
        image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
        n_img = int(image_positions.numel())

        need_attn = fastv_layer is not None and keep_ratio < 1.0

        if not need_attn:
            self._set_attn("sdpa")
            out = self.model(**inputs, use_cache=False)
            logits = out.logits[0, -1].float()
            img_after = n_img
        else:
            # eager exposes layer-K attention for ranking (shallow layers stay finite);
            # the deep re-run then uses sdpa for numerically stable logits.
            self._set_attn("eager")
            out = self.model(**inputs, output_attentions=True,
                             output_hidden_states=True, use_cache=False)
            attn = out.attentions[fastv_layer - 1][0]          # [heads, L, L]
            recv = attn.mean(0)[-1]                             # [L] attn from last token
            attn_to_image = recv[image_positions]              # [n_img]
            keep_idx = select_keep_indices(attn_to_image, image_positions, seq_len, keep_ratio)
            hidden_K = out.hidden_states[fastv_layer]           # input to layer K
            self._set_attn("sdpa")
            logits = self._deep_rerun(hidden_K, keep_idx, fastv_layer)
            img_after = int((keep_idx.unsqueeze(1) == image_positions).any(1).sum())

        letters = "ABCDEFGH"[:num_choices]
        scores = torch.tensor([logits[self.letter_ids[L]] for L in letters])
        pred = letters[int(scores.argmax())]
        return {"pred": pred, "image_tokens_before": n_img, "image_tokens_after": img_after,
                "seq_len": seq_len}
