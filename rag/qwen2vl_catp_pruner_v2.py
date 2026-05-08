import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import base64
import io
from typing import Any, Dict, Tuple
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def _normalize_map(attn_map: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    attn_map = attn_map.astype(np.float32)
    return (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + eps)

def _patch_importance_to_spatial_map(
    patch_importance: torch.Tensor,
    active_token_indices: torch.Tensor,
    merged_grid_h: int,
    merged_grid_w: int,
) -> np.ndarray:
    """
    Convert 1D visual-token importance into a 2D spatial attention map.

    patch_importance:
        shape = (num_active_tokens,)

    active_token_indices:
        original visual-token indices among all image tokens.
        shape = (num_active_tokens,)
    """
    spatial_map = torch.zeros(
        merged_grid_h * merged_grid_w,
        device=patch_importance.device,
        dtype=patch_importance.dtype,
    )

    spatial_map[active_token_indices] = patch_importance
    spatial_map = spatial_map.reshape(merged_grid_h, merged_grid_w)

    spatial_map = spatial_map.detach().float().cpu().numpy()
    spatial_map = _normalize_map(spatial_map)

    return spatial_map

def collect_spatial_attention_maps(
    outputs,
    input_ids,
    image_token_indices,
    current_active_indices,
    query_start_idx,
    grid_h: int,
    grid_w: int,
    merge_size: int,
    layer_start: int = 6,
    layer_end: int = 14,
):
    """
    Collect query-to-image spatial attention maps from layer_start to layer_end.

    Returns:
        spatial_maps: dict[layer_idx] -> 2D numpy array
    """
    merged_grid_h = grid_h // merge_size
    merged_grid_w = grid_w // merge_size

    spatial_maps = {}
    active_global_indices = image_token_indices[current_active_indices]

    for layer_idx in range(layer_start, layer_end + 1):
        layer_attn = outputs.attentions[layer_idx].squeeze(0)

        query_to_image_attn = layer_attn[:,query_start_idx:,:,][:, :, active_global_indices]
        patch_importance = query_to_image_attn.mean(dim=1).max(dim=0).values

        spatial_map = _patch_importance_to_spatial_map(
            patch_importance=patch_importance,
            active_token_indices=current_active_indices,
            merged_grid_h=merged_grid_h,
            merged_grid_w=merged_grid_w,
        )

        spatial_maps[layer_idx] = spatial_map

    return spatial_maps

def save_layer_spatial_attention_grid(
    image,
    spatial_maps,
    save_path: str,
    alpha: float = 0.45,
):
    """
    Save one figure containing spatial attention maps from multiple layers.

    image:
        PIL.Image

    spatial_maps:
        dict[layer_idx] -> 2D numpy array
    """
    layer_items = sorted(spatial_maps.items(), key=lambda x: x[0])

    num_maps = len(layer_items)
    cols = 3
    rows = math.ceil(num_maps / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).reshape(-1)

    image_np = np.array(image.convert("RGB"))

    for ax, (layer_idx, spatial_map) in zip(axes, layer_items):
        h, w = image_np.shape[:2]

        spatial_tensor = torch.tensor(spatial_map)[None, None, :, :]
        spatial_resized = F.interpolate(
            spatial_tensor,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        spatial_resized = _normalize_map(spatial_resized)

        ax.imshow(image_np)
        ax.imshow(spatial_resized, alpha=alpha)
        ax.set_title(f"Layer {layer_idx}")
        ax.axis("off")

    for ax in axes[num_maps:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def get_fused_patch_importance(
    outputs,
    image_token_indices,
    current_active_indices,
    query_start_idx,
    selected_layers=(10, 11, 12, 13),
    head_reduce="max",   # "max" or "mean"
    layer_reduce="sum", # "mean", "max", or "sum"
):
    active_global_indices = image_token_indices[current_active_indices]
    layer_importances = []

    for layer_idx in selected_layers:
        # (batch, heads, seq, seq) -> (heads, seq, seq)
        layer_attn = outputs.attentions[layer_idx].squeeze(0)
        # (heads, query_tokens, active_image_tokens)
        query_to_image_attn = layer_attn[:,query_start_idx:,:,][:, :, active_global_indices]
        # (heads, active_image_tokens)
        head_patch_importance = query_to_image_attn.mean(dim=1)

        if head_reduce == "max":
            # keep strongest grounding head
            patch_importance = head_patch_importance.max(dim=0).values
        elif head_reduce == "mean":
            # smoother, more conservative
            patch_importance = head_patch_importance.mean(dim=0)
        else:
            raise ValueError(f"Unknown head_reduce: {head_reduce}")

        layer_importances.append(patch_importance.float())

    stacked = torch.stack(layer_importances, dim=0)
    # shape: (num_selected_layers, active_image_tokens)

    if layer_reduce == "mean":
        fused_importance = stacked.mean(dim=0)
    elif layer_reduce == "max":
        fused_importance = stacked.max(dim=0).values
    elif layer_reduce == "sum":
        fused_importance = stacked.sum(dim=0)
    else:
        raise ValueError(f"Unknown layer_reduce: {layer_reduce}")

    return fused_importance

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

    def _attention_entropy(self, scores: torch.Tensor, eps: float = 1e-12) -> float:
        """
        scores: shape (num_image_tokens,)
        """
        probs = scores.float()
        probs = probs / (probs.sum() + eps)
        entropy = -(probs * torch.log(probs + eps)).sum()
        return float(entropy.detach().cpu())


    def _cluster_active_patches(self, active_patch_mask: torch.Tensor, max_gap: int = 1,):
        """
        active_patch_mask: bool tensor, shape = (merged_grid_h, merged_grid_w)

        max_gap:
            Allows small gaps between active patches.
            max_gap=1 means patches within a 3x3 neighborhood are connected.
            max_gap=2 means patches within a 5x5 neighborhood are connected.

        Returns:
            list of clusters, each cluster is list of (y, x)
        """
        h, w = active_patch_mask.shape
        visited = torch.zeros_like(active_patch_mask, dtype=torch.bool)

        clusters = []

        active_coords = active_patch_mask.nonzero(as_tuple=False)

        for coord in active_coords:
            start_y, start_x = int(coord[0]), int(coord[1])

            if visited[start_y, start_x]:
                continue

            queue = [(start_y, start_x)]
            visited[start_y, start_x] = True
            cluster = []

            while queue:
                y, x = queue.pop(0)
                cluster.append((y, x))

                y0 = max(0, y - max_gap)
                y1 = min(h - 1, y + max_gap)
                x0 = max(0, x - max_gap)
                x1 = min(w - 1, x + max_gap)

                for ny in range(y0, y1 + 1):
                    for nx in range(x0, x1 + 1):
                        if active_patch_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            clusters.append(cluster)

        return clusters

    def _bbox_prune_by_attention_percentile(
        self,
        patch_importance: torch.Tensor,
        current_active_indices: torch.Tensor,
        merged_grid_h: int,
        merged_grid_w: int,
        percentile_ratio: float = 0.7,
    ):
        """
        BBox-based visual token pruning.

        percentile_ratio:
            0.7 means keep tokens whose attention is above the 70th percentile.
            Higher value => more aggressive pruning.
        """

        assert 0.0 <= percentile_ratio <= 1.0

        device = patch_importance.device

        # 1. Convert percentile ratio into threshold
        threshold = torch.quantile(patch_importance.float(),percentile_ratio,)

        # 2. Select high-attention active tokens
        high_mask = patch_importance > threshold

        # Safety fallback: avoid pruning everything
        if high_mask.sum() == 0:
            high_mask[torch.argmax(patch_importance)] = True

        selected_active_indices = current_active_indices[high_mask]

        # 3. Convert selected 1D token indices to 2D grid coordinates
        selected_y = selected_active_indices // merged_grid_w
        selected_x = selected_active_indices % merged_grid_w

        # 4. Build bounding box around selected high-attention tokens
        min_x = selected_x.min()
        max_x = selected_x.max()
        min_y = selected_y.min()
        max_y = selected_y.max()

        # 5. Keep every token inside the bbox, not only top attention tokens
        all_active_y = current_active_indices // merged_grid_w
        all_active_x = current_active_indices % merged_grid_w

        bbox_mask = (
            (all_active_x >= min_x)
            & (all_active_x <= max_x)
            & (all_active_y >= min_y)
            & (all_active_y <= max_y)
        )

        pruned_active_indices = current_active_indices[bbox_mask]

        # Safety fallback
        if pruned_active_indices.numel() == 0:
            pruned_active_indices = selected_active_indices

        bbox_info = {
            "threshold": float(threshold.detach().cpu()),
            "percentile_ratio": float(percentile_ratio),
            "bbox_grid": {
                "min_x": int(min_x.detach().cpu()),
                "max_x": int(max_x.detach().cpu()),
                "min_y": int(min_y.detach().cpu()),
                "max_y": int(max_y.detach().cpu()),
            },
            "tokens_before": int(current_active_indices.numel()),
            "high_attention_tokens": int(selected_active_indices.numel()),
            "tokens_after_bbox": int(pruned_active_indices.numel()),
        }

        return pruned_active_indices, bbox_info

    def get_pruned_image(
        self,
        image: Image.Image,
        query: str,
        keep_ratio: float = 0.3,
        percentile_ratio: float = 0.5,
        image_cache_id: str = "example",
        tag_hash: str = "123",
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
        # active_mask = self._diversity_pre_filter(early_image_features)
        active_mask = torch.ones(early_image_features.shape[0], dtype=torch.bool, device=early_image_features.device,)
        active_image_token_indices = image_token_indices[active_mask]
        current_active_indices = torch.arange(
            image_token_indices.numel(),
            device=input_ids.device,
        )[active_mask]
        if active_image_token_indices.numel() == 0:
            raise ValueError("Diversity pre-filter removed all image tokens.")
        diversity_keep_indices = current_active_indices.detach().cpu().tolist()
        
        # Find query token idx
        query_ids = self.processor.tokenizer(query, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(input_ids.device)
        def find_subsequence(seq, subseq):
            n, m = seq.numel(), subseq.numel()
            for i in range(n - m + 1):
                if torch.equal(seq[i:i + m], subseq):
                    return i
            return -1
        query_start_idx = find_subsequence(input_ids, query_ids)
        if query_start_idx == -1:
            raise ValueError("Could not find query tokens in input_ids.")

        # # Layer-wise measurement: analyze query -> image attention for every layer (dev)
        layer_metrics = []
        # prev_topk_set = None
        # analysis_topk_ratio = keep_ratio

        # for layer_idx in range(len(outputs.attentions)):
        #     layer_attn = outputs.attentions[layer_idx].mean(dim=1).squeeze(0)

        #     active_global_indices = image_token_indices[current_active_indices]
        #     query_to_image_attn = layer_attn[query_start_idx:, active_global_indices]

        #     patch_importance = query_to_image_attn.mean(dim=0)

        #     num_active = int(current_active_indices.numel())
        #     analysis_k = max(1, int(num_active * analysis_topk_ratio))
        #     analysis_k = min(analysis_k, num_active)

        #     topk = torch.topk(patch_importance, analysis_k)
        #     topk_indices_local = topk.indices.detach().cpu().tolist()
        #     topk_set = set(int(i) for i in topk_indices_local)

        #     total_mass = patch_importance.sum().item()
        #     topk_mass = topk.values.sum().item()
        #     topk_mass_ratio = topk_mass / total_mass if total_mass > 0 else 0.0

        #     entropy = self._attention_entropy(patch_importance)

        #     if prev_topk_set is None:
        #         topk_overlap = None
        #     else:
        #         topk_overlap = len(topk_set & prev_topk_set) / max(1, len(topk_set))

        #     layer_metrics.append({
        #         "layer_idx": int(layer_idx),
        #         "num_active_tokens": num_active,
        #         "query_to_image_attention_mean": float(patch_importance.mean().detach().cpu()),
        #         "query_to_image_attention_max": float(patch_importance.max().detach().cpu()),
        #         "topk_mass_ratio": float(topk_mass_ratio),
        #         "attention_entropy": float(entropy),
        #         "topk_overlap_with_prev_layer": topk_overlap,
        #     })
        #     prev_topk_set = topk_set

        # # Spatial Heat Map (dev)
        # spatial_maps = collect_spatial_attention_maps(
        #     outputs=outputs,
        #     input_ids=input_ids,
        #     image_token_indices=image_token_indices,
        #     current_active_indices=current_active_indices,
        #     query_start_idx=query_start_idx,
        #     grid_h=grid_h,
        #     grid_w=grid_w,
        #     merge_size=merge_size,
        #     layer_start=6,
        #     layer_end=14,
        # )
        # save_layer_spatial_attention_grid(
        #     image=image,
        #     spatial_maps=spatial_maps,
        #     save_path=f"data/mmdocrag/analysis/{image_cache_id}_{tag_hash}_spatial_attention_layers_6_to_14.png",
        # )

        percentile_ratio = 0.5
        merged_grid_h = grid_h // merge_size
        merged_grid_w = grid_w // merge_size
        patch_importance = get_fused_patch_importance(
            outputs=outputs,
            image_token_indices=image_token_indices,
            current_active_indices=current_active_indices,
            query_start_idx=query_start_idx,
            selected_layers=(10, 11, 12, 13),
        )
        current_active_indices, bbox_info = self._bbox_prune_by_attention_percentile(
            patch_importance=patch_importance,
            current_active_indices=current_active_indices,
            merged_grid_h=merged_grid_h,
            merged_grid_w=merged_grid_w,
            percentile_ratio=percentile_ratio,
        )
        true_tokens_after = bbox_info["tokens_after_bbox"]
        x_min, x_max, y_min, y_max = bbox_info["bbox_grid"].values()
        cropped_image = image.crop((x_min, y_min, x_max, y_max))


        # Execute progressive contextual pruning across transformer depth. (Old)
        # num_layers = len(outputs.attentions)
        # num_cuts = min(1, num_layers-1)
        # prune_stages = self._get_prune_stages(num_layers, num_cuts)
        # keep_ratios = self._get_keep_ratios(keep_ratio, num_cuts)
        # progressive_stages = []
        # final_scores = None
        # for layer_idx, target_ratio in zip(prune_stages, keep_ratios):
        #     layer_attn = outputs.attentions[layer_idx].mean(dim=1).squeeze(0)
        #     active_global_indices = image_token_indices[current_active_indices]
        #     query_to_image_attn = layer_attn[query_start_idx:, active_global_indices]
        #     patch_importance = query_to_image_attn.mean(dim=0)

        #     tokens_before_stage = int(current_active_indices.numel())
        #     k = max(1, int(num_image_tokens * target_ratio))
        #     k = min(k, tokens_before_stage)

        #     top_k = torch.topk(patch_importance, k)
        #     current_active_indices = current_active_indices[top_k.indices]
        #     final_scores = top_k.values.detach().float().cpu().tolist()

        #     progressive_stages.append({
        #         "layer_idx": int(layer_idx),
        #         "target_ratio": float(target_ratio),
        #         "tokens_before": tokens_before_stage,
        #         "tokens_after": int(current_active_indices.numel()),
        #         "keep_indices": [
        #             int(i) for i in current_active_indices.detach().cpu().tolist()
        #         ],
        #     })

        # top_k_indices = current_active_indices.detach().cpu().numpy()
        # top_k_scores = final_scores or []
        # k = int(current_active_indices.numel())

        # # Translate 1D Tokens back to 2D Bounding Box
        # # Qwen2-VL flattens the grid row-by-row
        # merged_grid_w = grid_w // merge_size
        # merged_grid_h = grid_h // merge_size

        # patch_y = top_k_indices // merged_grid_w
        # patch_x = top_k_indices % merged_grid_w
        
        # # Get bounding box in grid coordinates
        # min_grid_x, max_grid_x = np.min(patch_x), np.max(patch_x)
        # min_grid_y, max_grid_y = np.min(patch_y), np.max(patch_y)
        
        # # Convert grid coordinates to physical pixel coordinates based on original image size
        # px_per_grid_x = width / merged_grid_w
        # px_per_grid_y = height / merged_grid_h
        
        # x_min = int(min_grid_x * px_per_grid_x)
        # y_min = int(min_grid_y * px_per_grid_y)
        # x_max = int((max_grid_x + 1) * px_per_grid_x)
        # y_max = int((max_grid_y + 1) * px_per_grid_y)
        
        # # Clamp to image boundaries
        # x_min, y_min = max(0, x_min), max(0, y_min)
        # x_max, y_max = min(width, x_max), min(height, y_max)
        
        # # Physical Safe Crop
        # cropped_image = image.crop((x_min, y_min, x_max, y_max))
        # bbox_grid_w = max_grid_x - min_grid_x + 1
        # bbox_grid_h = max_grid_y - min_grid_y + 1
        # true_tokens_after = int(bbox_grid_w * bbox_grid_h)

        metadata = {
            "tokens_before": num_image_tokens,
            "tokens_after": true_tokens_after,
            "tokens_before_diversity": int(image_token_indices.numel()),
            "tokens_after_diversity": int(active_image_token_indices.numel()),
            "layer_metrics": layer_metrics if len(layer_metrics) != 0 else None,
        }

        return cropped_image, metadata

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
