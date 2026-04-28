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

        cropped_image, seam_meta = self._mask_guided_seam_carve(
            image=image,
            patch_indices=top_k_indices,
            grid_h=int(merged_grid_h),
            grid_w=int(merged_grid_w),
            keep_ratio=keep_ratio,
        )

        metadata = {
            "tokens_before": num_image_tokens,
            "tokens_after": k,
            "tokens_before_diversity": int(image_token_indices.numel()),
            "tokens_after_diversity": int(active_image_token_indices.numel()),
            "seam_carving": seam_meta,
        }

        return cropped_image, metadata

    def _mask_guided_seam_carve(
        self,
        image: Image.Image,
        patch_indices,
        grid_h: int,
        grid_w: int,
        keep_ratio: float,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width, _ = arr.shape
        protect_mask, protected_boxes = self._build_protective_mask(
            width=width,
            height=height,
            patch_indices=patch_indices,
            grid_h=grid_h,
            grid_w=grid_w,
        )

        target_ratio = min(max(float(keep_ratio), 0.05), 1.0)
        scale = target_ratio ** 0.5
        min_width = max(1, int(protect_mask.any(axis=0).sum()))
        min_height = max(1, int(protect_mask.any(axis=1).sum()))
        target_width = max(min_width, int(width * scale))
        target_height = max(min_height, int(height * scale))

        vertical_removed = 0
        horizontal_removed = 0
        vertical_protected_block = False
        horizontal_protected_block = False

        while arr.shape[1] > target_width:
            seam = self._find_vertical_seam(arr, protect_mask)
            if protect_mask[np.arange(arr.shape[0]), seam].any():
                vertical_protected_block = True
                break
            arr, protect_mask = self._remove_vertical_seam(arr, protect_mask, seam)
            vertical_removed += 1

        while arr.shape[0] > target_height:
            seam = self._find_horizontal_seam(arr, protect_mask)
            if protect_mask[seam, np.arange(arr.shape[1])].any():
                horizontal_protected_block = True
                break
            arr, protect_mask = self._remove_horizontal_seam(arr, protect_mask, seam)
            horizontal_removed += 1

        carved = Image.fromarray(arr, mode="RGB")
        metadata = {
            "method": "mask_guided_seam_carving",
            "target_area_ratio": target_ratio,
            "original_size": [int(width), int(height)],
            "target_size": [int(target_width), int(target_height)],
            "carved_size": [int(carved.width), int(carved.height)],
            "protected_boxes": protected_boxes,
        }
        return carved, metadata

    def _build_protective_mask(
        self,
        width: int,
        height: int,
        patch_indices,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[np.ndarray, list]:
        patch_indices = np.asarray(patch_indices, dtype=np.int64)
        row_edges = np.linspace(0, height, grid_h + 1, dtype=int)
        col_edges = np.linspace(0, width, grid_w + 1, dtype=int)
        mask = np.zeros((height, width), dtype=bool)
        boxes = []

        for patch_idx in patch_indices.tolist():
            row = int(patch_idx // grid_w)
            col = int(patch_idx % grid_w)
            left = int(col_edges[col])
            right = int(col_edges[col + 1])
            top = int(row_edges[row])
            bottom = int(row_edges[row + 1])
            mask[top:bottom, left:right] = True
            boxes.append([left, top, right, bottom])

        return mask, boxes

    def _energy_map(self, arr: np.ndarray, protect_mask: np.ndarray) -> np.ndarray:
        gray = (
            0.299 * arr[:, :, 0].astype(np.float32)
            + 0.587 * arr[:, :, 1].astype(np.float32)
            + 0.114 * arr[:, :, 2].astype(np.float32)
        )
        dy, dx = np.gradient(gray)
        energy = np.abs(dx) + np.abs(dy)
        energy = energy - 100.0
        energy[protect_mask] = 1_000_000.0
        return energy

    def _find_vertical_seam(self, arr: np.ndarray, protect_mask: np.ndarray) -> np.ndarray:
        energy = self._energy_map(arr, protect_mask)
        height, width = energy.shape
        cost = energy.copy()
        backtrack = np.zeros((height, width), dtype=np.int32)

        for row in range(1, height):
            prev = cost[row - 1]
            for col in range(width):
                left = max(0, col - 1)
                right = min(width, col + 2)
                offset = int(np.argmin(prev[left:right]))
                best_col = left + offset
                cost[row, col] += prev[best_col]
                backtrack[row, col] = best_col

        seam = np.zeros(height, dtype=np.int64)
        seam[-1] = int(np.argmin(cost[-1]))
        for row in range(height - 2, -1, -1):
            seam[row] = backtrack[row + 1, seam[row + 1]]
        return seam

    def _find_horizontal_seam(self, arr: np.ndarray, protect_mask: np.ndarray) -> np.ndarray:
        seam = self._find_vertical_seam(
            np.transpose(arr, (1, 0, 2)),
            protect_mask.T,
        )
        return seam

    def _remove_vertical_seam(
        self,
        arr: np.ndarray,
        protect_mask: np.ndarray,
        seam: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        height, width, channels = arr.shape
        keep = np.ones((height, width), dtype=bool)
        keep[np.arange(height), seam] = False
        carved_arr = arr[keep].reshape(height, width - 1, channels)
        carved_mask = protect_mask[keep].reshape(height, width - 1)
        return carved_arr, carved_mask

    def _remove_horizontal_seam(
        self,
        arr: np.ndarray,
        protect_mask: np.ndarray,
        seam: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        transposed_arr = np.transpose(arr, (1, 0, 2))
        transposed_mask = protect_mask.T
        carved_arr, carved_mask = self._remove_vertical_seam(
            transposed_arr,
            transposed_mask,
            seam,
        )
        return np.transpose(carved_arr, (1, 0, 2)), carved_mask.T

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
