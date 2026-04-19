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

    # generation
    vlm_api_base: str = "http://127.0.0.1:8000/v1"
    vlm_model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    temperature: float = 0.0

    # eval
    max_examples: int = 50   # start small for baseline