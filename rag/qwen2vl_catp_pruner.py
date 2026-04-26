import torch
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
            outputs = self.model(**inputs, output_attentions=True)
            
            # Get the attention from the final transformer layer
            # Shape: (batch, num_heads, seq_len, seq_len)
            last_layer_attn = outputs.attentions[-1]
            
            # Average the attention across all heads
            avg_attn = last_layer_attn.mean(dim=1).squeeze(0) # Shape: (seq_len, seq_len)

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
        
        # 4. Execute Contextual Importance Scoring (CATP)
        # We want the attention from the text query tokens to the image patch tokens
        # Shape: (num_query_tokens, num_image_tokens)
        query_to_image_attn = avg_attn[query_start_idx:, image_token_indices]
        
        # Average the attention to find the most globally important patches for this query
        patch_importance = query_to_image_attn.mean(dim=0) # Shape: (num_image_tokens,)

        # Identify Top K Patches
        k = max(1, int(num_image_tokens * keep_ratio))
        top_k = torch.topk(patch_importance, k)
        top_k_indices = top_k.indices.cpu().numpy()
        top_k_scores = top_k.values.detach().float().cpu().tolist()

        # Translate 1D Tokens back to 2D Bounding Box
        # Qwen2-VL flattens the grid row-by-row
        merged_grid_w = grid_w // merge_size
        merged_grid_h = grid_h // merge_size

        patch_y = top_k_indices // merged_grid_w
        patch_x = top_k_indices % merged_grid_w
        
        # Get bounding box in grid coordinates
        min_grid_x, max_grid_x = np.min(patch_x), np.max(patch_x)
        min_grid_y, max_grid_y = np.min(patch_y), np.max(patch_y)
        
        # Convert grid coordinates to physical pixel coordinates based on original image size
        px_per_grid_x = width / merged_grid_w
        px_per_grid_y = height / merged_grid_h
        
        x_min = int(min_grid_x * px_per_grid_x)
        y_min = int(min_grid_y * px_per_grid_y)
        x_max = int((max_grid_x + 1) * px_per_grid_x)
        y_max = int((max_grid_y + 1) * px_per_grid_y)
        
        # Clamp to image boundaries
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(width, x_max), min(height, y_max)
        
        # 7. Physical Safe Crop
        cropped_image = image.crop((x_min, y_min, x_max, y_max))

        metadata = {
            "tokens_before": num_image_tokens,
            "tokens_after": k,
            "qwen_grid_thw": [int(grid_t), int(grid_h), int(grid_w)],
            "merged_grid_h": int(merged_grid_h),
            "merged_grid_w": int(merged_grid_w),
            "merge_size": int(merge_size),
            "keep_indices": [int(i) for i in top_k_indices.tolist()],
            "keep_grid_xy": [
                [int(x), int(y)] for x, y in zip(patch_x.tolist(), patch_y.tolist())
            ],
            "scores": [float(score) for score in top_k_scores],
            "crop_box": [int(x_min), int(y_min), int(x_max), int(y_max)],
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
