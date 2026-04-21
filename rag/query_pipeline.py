import base64
import time
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from rag.config import RAGConfig
from rag.prompt_builder import build_prompt
from rag.pruner import RetrievalPruner
from rag.retriever import QuoteRetriever
from rag.visual_hook_probe import VisualTokenHookProbe
from rag.local_vlm_runner import LocalHookedVLMRunner

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class MMDocRAGPipeline:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.retriever = QuoteRetriever(
            text_model_name=cfg.text_embedding_model,
            image_model_name=cfg.image_embedding_model,
            device=cfg.retrieval_device,
        )
        self.pruner = RetrievalPruner(
            mode=cfg.pruning_mode,
            keep_ratio=cfg.pruning_keep_ratio,
            image_model_name=cfg.image_embedding_model,
            device=cfg.retrieval_device,
            patch_grid_rows=cfg.patch_grid_rows,
            patch_grid_cols=cfg.patch_grid_cols,
            min_visual_tokens=cfg.min_visual_tokens,
            montage_tile_size=cfg.montage_tile_size,
            output_dir=cfg.pruned_image_dir,
        )
        self.client = OpenAI(base_url=cfg.vlm_api_base, api_key="EMPTY")
        self.visual_hook_probe = (
            VisualTokenHookProbe(
                image_model_name=cfg.image_embedding_model,
                device=cfg.retrieval_device,
            )
            if cfg.enable_visual_hook_probe
            else None
        )
        self.local_vlm_runner = (
            LocalHookedVLMRunner(
                model_name=cfg.local_vlm_model_name,
                device=cfg.local_generation_device,
                dtype=cfg.local_generation_dtype,
            )
            if cfg.enable_local_hooked_generation and cfg.local_vlm_model_name
            else None
        )

    def run_one(self, example: Dict) -> Dict:
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(
            example,
            text_top_k=self.cfg.text_top_k,
            image_top_k=self.cfg.image_top_k,
        )
        t1 = time.perf_counter()

        pruned_retrieval = self.pruner.apply(example, retrieval)

        prompt = build_prompt(example, pruned_retrieval)

        local_generation_meta = None
        if self.local_vlm_runner is not None and self.cfg.pruning_mode == "model_internal_visual_pruning":
            local_generation_meta = self._maybe_run_local_hooked_generation(
                prompt=prompt,
                selected_img_quotes=pruned_retrieval["selected_img_quotes"],
            )

        if local_generation_meta and local_generation_meta.get("applied"):
            t2 = time.perf_counter()
            pred = local_generation_meta.get("generated_text", "")
            first_token_time = None
            t3 = time.perf_counter()
        else:
            content = [{"type": "text", "text": prompt}]
            for q in pruned_retrieval["selected_img_quotes"]:
                path = q.get("local_img_path")
                if path and Path(path).exists():
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}
                    })

            t2 = time.perf_counter()
            stream = self.client.chat.completions.create(
                model=self.cfg.vlm_model_name,
                messages=[{"role": "user", "content": content}],
                temperature=self.cfg.temperature,
                stream=True,
            )
            first_token_time = None
            pieces = []

            for chunk in stream:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    pieces.append(text)

            t3 = time.perf_counter()
            pred = "".join(pieces)

        retrieved_ids = [
            q["quote_id"] for q in pruned_retrieval["selected_text_quotes"]
        ] + [
            q["quote_id"] for q in pruned_retrieval["selected_img_quotes"]
        ]

        ttft = None if first_token_time is None else (first_token_time - t2)

        visual_pruning_meta = [
            {
                "quote_id": q.get("quote_id"),
                "visual_pruning": q.get("visual_pruning"),
            }
            for q in pruned_retrieval["selected_img_quotes"]
            if q.get("visual_pruning") is not None
        ]

        hook_probe_meta = self._run_visual_hook_probe(pruned_retrieval["selected_img_quotes"])

        return {
            "q_id": example["q_id"],
            "question": example["question"],
            "gold_answer": example["answer_short"],
            "pred_answer": pred,
            "gold_quotes": example["gold_quotes"],
            "retrieved_quote_ids": retrieved_ids,
            "selected_text_quotes": pruned_retrieval["selected_text_quotes"],
            "selected_img_quotes": pruned_retrieval["selected_img_quotes"],
            "pruning": pruned_retrieval["pruning"],
            "visual_pruning": visual_pruning_meta,
            "visual_hook_probe": hook_probe_meta,
            "local_hooked_generation": local_generation_meta,
            "timing": {
                "retrieval_sec": t1 - t0,
                "request_build_sec": t2 - t1,
                "ttft_sec": ttft,
                "generation_sec": t3 - t2,
                "total_sec": t3 - t0,
            }
        }

    def _run_visual_hook_probe(self, selected_img_quotes: List[Dict]) -> List[Dict]:
        if self.visual_hook_probe is None:
            return []

        results = []
        for q in selected_img_quotes:
            visual_pruning = q.get("visual_pruning")
            image_path = q.get("local_img_path")
            if not visual_pruning or not image_path:
                continue
            if visual_pruning.get("mode") != "model_internal_visual_pruning":
                continue

            result = self.visual_hook_probe.run_on_image(
                image_path=image_path,
                visual_pruning=visual_pruning,
            )
            results.append({
                "quote_id": q.get("quote_id"),
                "probe": result,
            })
        return results

    def _maybe_run_local_hooked_generation(
        self,
        prompt: str,
        selected_img_quotes: List[Dict],
    ) -> Dict:
        if not selected_img_quotes:
            return {"applied": False, "reason": "no_selected_images"}

        q = selected_img_quotes[0]
        visual_pruning = q.get("visual_pruning")
        image_path = q.get("local_img_path")
        if not visual_pruning or not image_path:
            return {"applied": False, "reason": "missing_pruning_metadata_or_image"}

        result = self.local_vlm_runner.generate(
            prompt=prompt,
            image_path=image_path,
            visual_pruning=visual_pruning,
            max_new_tokens=self.cfg.local_generation_max_new_tokens,
        )
        result["quote_id"] = q.get("quote_id")
        if len(selected_img_quotes) > 1:
            result["note"] = "local hooked generation currently uses the first selected image only"
        return result