import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import base64
import io
from typing import Any, Dict, Tuple
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class Qwen2VLCATPBoundingBoxCropper:
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4", device="cuda"):
        self.device = device
        if "gptq" in model_id.lower():
            from optimum.utils import is_gptqmodel_available

            if not is_gptqmodel_available():
                raise RuntimeError(
                    "Loading this GPTQ model requires gptqmodel (used by optimum with transformers). "
                    "Install with: pip install 'gptqmodel>=1.6.0'"
                )
        # Load the quantized Qwen2-VL model and its specific processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        self.model.eval()

    def _diversity_pre_filter(self, hidden_states, similarity_threshold=0.98):
        """
        CATP Stage 1: Unconditional diversity-based pruning.
        Removes visually identical patches before attention scaling.
        """
        norm_states = F.normalize(hidden_states.float(), p=2, dim=1)
        sim_matrix = torch.matmul(norm_states, norm_states.T)

        unique_mask = torch.ones(
            hidden_states.shape[0],
            dtype=torch.bool,
            device=hidden_states.device,
        )

        for i in range(hidden_states.shape[0]):
            if unique_mask[i]:
                duplicates = sim_matrix[i] > similarity_threshold
                duplicates[i] = False
                unique_mask[duplicates] = False

        return unique_mask

    def _get_prune_stages(self, num_layers: int, num_cuts: int):
        return np.linspace(0, num_layers - 1, num_cuts + 1, dtype=int)[1:].tolist()

    def _get_keep_ratios(self, final_keep_ratio: float, num_cuts: int):
        return [
            final_keep_ratio ** (i / num_cuts)
            for i in range(1, num_cuts + 1)
        ]

    def get_pruned_image(
        self,
        image: Image.Image,
        query: str,
        keep_ratio: float = 0.3,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Executes CATP Attention-Based Pruning using Qwen2-VL's dynamic spatial grid.
        Returns the cropped image and pruning metadata.
        """
        width, height = image.size

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # Qwen style prompt
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
            )

        # Map Qwen2-VL's Dynamic Grid
        input_ids = inputs.input_ids.squeeze(0)
        
        # Qwen2-VL stores the dynamic shape of the image in image_grid_thw (Temporal, Height, Width)
        # For a static image, Temporal is 1. Height and Width are the patch grids.
        grid_t, grid_h, grid_w = inputs.image_grid_thw[0].tolist()
        merge_size = self.processor.image_processor.merge_size
        num_image_tokens = int(grid_t * grid_h * grid_w // (merge_size ** 2))
        
        # Qwen2-VL uses a specific token for image patches (usually <|image_pad|>)
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
        if image_token_indices.numel() == 0:
            raise ValueError("Could not find image tokens in input_ids.")

        early_image_features = outputs.hidden_states[0].squeeze(0)[image_token_indices]
        active_mask = self._diversity_pre_filter(early_image_features)
        active_image_token_indices = image_token_indices[active_mask]
        current_active_indices = torch.arange(
            image_token_indices.numel(),
            device=input_ids.device,
        )[active_mask]
        if active_image_token_indices.numel() == 0:
            raise ValueError("Diversity pre-filter removed all image tokens.")
        diversity_keep_indices = current_active_indices.detach().cpu().tolist()
        
        # Find query token idx
        query_ids = self.processor.tokenizer(
            query,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.squeeze(0).to(input_ids.device)
        def find_subsequence(seq, subseq):
            n, m = seq.numel(), subseq.numel()
            for i in range(n - m + 1):
                if torch.equal(seq[i:i + m], subseq):
                    return i
            return -1
        query_start_idx = find_subsequence(input_ids, query_ids)
        if query_start_idx == -1:
            raise ValueError("Could not find query tokens in input_ids.")

        # 4. Execute progressive contextual pruning across transformer depth.
        num_cuts = 3
        num_layers = len(outputs.attentions)
        prune_stages = self._get_prune_stages(num_layers, num_cuts)
        keep_ratios = self._get_keep_ratios(keep_ratio, num_cuts)
        progressive_stages = []
        final_scores = None

        for layer_idx, target_ratio in zip(prune_stages, keep_ratios):
            layer_attn = outputs.attentions[layer_idx].mean(dim=1).squeeze(0)
            active_global_indices = image_token_indices[current_active_indices]
            query_to_image_attn = layer_attn[query_start_idx:, active_global_indices]
            patch_importance = query_to_image_attn.mean(dim=0)

            tokens_before_stage = int(current_active_indices.numel())
            k = max(1, int(num_image_tokens * target_ratio))
            k = min(k, tokens_before_stage)

            top_k = torch.topk(patch_importance, k)
            current_active_indices = current_active_indices[top_k.indices]
            final_scores = top_k.values.detach().float().cpu().tolist()

            progressive_stages.append({
                "layer_idx": int(layer_idx),
                "target_ratio": float(target_ratio),
                "tokens_before": tokens_before_stage,
                "tokens_after": int(current_active_indices.numel()),
                "keep_indices": [
                    int(i) for i in current_active_indices.detach().cpu().tolist()
                ],
            })

        top_k_indices = current_active_indices.detach().cpu().numpy()
        top_k_scores = final_scores or []
        k = int(current_active_indices.numel())

        # Translate 1D Tokens back to 2D Bounding Box
        # Qwen2-VL flattens the grid row-by-row
        merged_grid_w = grid_w // merge_size
        merged_grid_h = grid_h // merge_size

        patch_y = top_k_indices // merged_grid_w
        patch_x = top_k_indices % merged_grid_w
        
        # Get bounding box in grid coordinates for metadata and fallback context.
        min_grid_x, max_grid_x = np.min(patch_x), np.max(patch_x)
        min_grid_y, max_grid_y = np.min(patch_y), np.max(patch_y)

        px_per_grid_x = width / merged_grid_w
        px_per_grid_y = height / merged_grid_h

        x_min = int(min_grid_x * px_per_grid_x)
        y_min = int(min_grid_y * px_per_grid_y)
        x_max = int((max_grid_x + 1) * px_per_grid_x)
        y_max = int((max_grid_y + 1) * px_per_grid_y)

        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(width, x_max), min(height, y_max)

        cropped_image, slicing_meta = self._whitespace_guillotine_slice(
            image=image,
            patch_indices=top_k_indices,
            patch_scores=top_k_scores,
            grid_h=int(merged_grid_h),
            grid_w=int(merged_grid_w),
        )

        metadata = {
            "tokens_before": num_image_tokens,
            "tokens_after": k,
            "tokens_before_diversity": int(image_token_indices.numel()),
            "tokens_after_diversity": int(active_image_token_indices.numel()),
            "progressive_pruning_cuts": num_cuts,
            "progressive_pruning_stages": progressive_stages,
            "qwen_grid_thw": [int(grid_t), int(grid_h), int(grid_w)],
            "merged_grid_h": int(merged_grid_h),
            "merged_grid_w": int(merged_grid_w),
            "merge_size": int(merge_size),
            "diversity_keep_indices": [
                int(i) for i in diversity_keep_indices
            ],
            "keep_indices": [int(i) for i in top_k_indices.tolist()],
            "keep_grid_xy": [
                [int(x), int(y)] for x, y in zip(patch_x.tolist(), patch_y.tolist())
            ],
            "scores": [float(score) for score in top_k_scores],
            "crop_box": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "document_slicing": slicing_meta,
        }

        return cropped_image, metadata

    def _whitespace_guillotine_slice(
        self,
        image: Image.Image,
        patch_indices,
        patch_scores,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Delete low-attention grid rows and columns, then stitch the survivors.

        This is intended for document-like images where removing blank gutters is
        less destructive than taking one large bounding box.
        """
        width, height = image.size
        patch_indices = np.asarray(patch_indices, dtype=np.int64)
        patch_scores = np.asarray(patch_scores, dtype=np.float32)
        if patch_scores.size != patch_indices.size:
            patch_scores = np.ones(patch_indices.size, dtype=np.float32)

        patch_y = patch_indices // grid_w
        patch_x = patch_indices % grid_w
        row_scores = np.zeros(grid_h, dtype=np.float32)
        col_scores = np.zeros(grid_w, dtype=np.float32)

        for x, y, score in zip(patch_x, patch_y, patch_scores):
            row_scores[int(y)] += float(score)
            col_scores[int(x)] += float(score)

        active_rows = np.unique(patch_y).astype(np.int64)
        active_cols = np.unique(patch_x).astype(np.int64)
        kept_rows, row_threshold, row_fallback = self._select_guillotine_axis(
            scores=row_scores,
            active_indices=active_rows,
        )
        kept_cols, col_threshold, col_fallback = self._select_guillotine_axis(
            scores=col_scores,
            active_indices=active_cols,
        )

        row_edges = np.linspace(0, height, grid_h + 1, dtype=int)
        col_edges = np.linspace(0, width, grid_w + 1, dtype=int)
        stitched = self._stitch_grid_intersections(
            image=image,
            kept_rows=kept_rows,
            kept_cols=kept_cols,
            row_edges=row_edges,
            col_edges=col_edges,
        )

        deleted_rows = sorted(set(range(grid_h)) - set(kept_rows.tolist()))
        deleted_cols = sorted(set(range(grid_w)) - set(kept_cols.tolist()))
        metadata = {
            "method": "whitespace_guillotine",
            "grid_h": int(grid_h),
            "grid_w": int(grid_w),
            "row_threshold": float(row_threshold),
            "col_threshold": float(col_threshold),
            "row_threshold_fallback": bool(row_fallback),
            "col_threshold_fallback": bool(col_fallback),
            "kept_rows": [int(i) for i in kept_rows.tolist()],
            "kept_cols": [int(i) for i in kept_cols.tolist()],
            "deleted_rows": [int(i) for i in deleted_rows],
            "deleted_cols": [int(i) for i in deleted_cols],
            "row_scores": [float(score) for score in row_scores.tolist()],
            "col_scores": [float(score) for score in col_scores.tolist()],
            "original_size": [int(width), int(height)],
            "stitched_size": [int(stitched.width), int(stitched.height)],
        }
        return stitched, metadata

    def _select_guillotine_axis(self, scores: np.ndarray, active_indices: np.ndarray):
        if active_indices.size == 0:
            return np.arange(scores.size, dtype=np.int64), 0.0, True

        positive_scores = scores[scores > 0]
        if positive_scores.size == 0:
            return np.sort(active_indices), 0.0, True

        threshold = float(positive_scores.mean() * 0.5)
        kept = np.flatnonzero(scores >= threshold).astype(np.int64)

        active_set = set(int(i) for i in active_indices.tolist())
        if kept.size == 0 or not active_set.issubset(set(int(i) for i in kept.tolist())):
            return np.sort(active_indices), threshold, True

        return kept, threshold, False

    def _stitch_grid_intersections(
        self,
        image: Image.Image,
        kept_rows: np.ndarray,
        kept_cols: np.ndarray,
        row_edges: np.ndarray,
        col_edges: np.ndarray,
    ) -> Image.Image:
        row_heights = [
            int(row_edges[row + 1] - row_edges[row])
            for row in kept_rows.tolist()
        ]
        col_widths = [
            int(col_edges[col + 1] - col_edges[col])
            for col in kept_cols.tolist()
        ]
        out_width = max(1, sum(col_widths))
        out_height = max(1, sum(row_heights))
        stitched = Image.new("RGB", (out_width, out_height), color=(255, 255, 255))

        y_out = 0
        for row, row_height in zip(kept_rows.tolist(), row_heights):
            x_out = 0
            for col, col_width in zip(kept_cols.tolist(), col_widths):
                box = (
                    int(col_edges[col]),
                    int(row_edges[row]),
                    int(col_edges[col + 1]),
                    int(row_edges[row + 1]),
                )
                tile = image.crop(box)
                stitched.paste(tile, (x_out, y_out))
                x_out += col_width
            y_out += row_height

        return stitched

    def get_pruned_image_base64(self, image: Image.Image, query: str, keep_ratio: float = 0.3) -> str:
        """
        Executes CATP Attention-Based Pruning and returns a base64 string.
        Kept for compatibility; pruner.py should use get_pruned_image().
        """
        cropped_image, _ = self.get_pruned_image(image, query, keep_ratio)

        buffered = io.BytesIO()
        cropped_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return f"data:image/jpeg;base64,{img_str}"
