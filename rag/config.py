from dataclasses import dataclass
from pathlib import Path

@dataclass
class RAGConfig:
    data_root: Path = Path("data/mmdocrag")
    ann_file: Path = Path("data/mmdocrag/dev_15.jsonl")
    images_root: Path = Path("data/mmdocrag/images")
    output_dir: Path = Path("data/mmdocrag/outputs")

    # retrieval
    text_top_k: int = 4
    image_top_k: int = 2
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_embedding_model: str = "openai/clip-vit-base-patch32"
    retrieval_device: str = "cuda"

    # pruning baselines
    pruning_mode: str = "model_internal_visual_pruning"  # {"no_pruning", "uniform_pruning", "visual_only_pruning", "visual_patch_pruning", "model_internal_visual_pruning"}
    pruning_keep_ratio: float = 0.5
    patch_grid_rows: int = 4
    patch_grid_cols: int = 4
    min_visual_tokens: int = 4
    montage_tile_size: int = 224
    pruned_image_dir: Path = Path("data/mmdocrag/outputs/pruned_images")

    # local probe / local hooked generation
    enable_visual_hook_probe: bool = True
    enable_local_hooked_generation: bool = True
    local_vlm_model_name: str | None = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    local_generation_device: str = "cuda"
    local_generation_dtype: str = "auto"   # {"auto", "float16", "bfloat16", "float32"}
    local_generation_max_new_tokens: int = 128

    # generation
    vlm_api_base: str = "http://127.0.0.1:8000/v1"
    vlm_model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    temperature: float = 0.0

    # offline judge
    judge_api_base: str = "http://127.0.0.1:8000/v1"
    judge_model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    # eval
    max_examples: int = 50   # start small for baseline