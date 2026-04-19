import json
from pathlib import Path
from statistics import mean

from rag.config import RAGConfig
from rag.mmdocrag_dataset import load_examples
from rag.metrics import exact_match, token_f1, retrieval_recall
from rag.query_pipeline import MMDocRAGPipeline

def run_baseline(cfg: RAGConfig):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(cfg.ann_file, cfg.images_root, limit=cfg.max_examples)
    pipe = MMDocRAGPipeline(cfg)

    rows = []
    for i, ex in enumerate(examples, start=1):
        out = pipe.run_one(ex)
        out["metrics"] = {
            "em": exact_match(out["pred_answer"], out["gold_answer"]),
            "f1": token_f1(out["pred_answer"], out["gold_answer"]),
            "retrieval_recall": retrieval_recall(out["retrieved_quote_ids"], out["gold_quotes"]),
        }
        rows.append(out)
        print(
            f"[{i}/{len(examples)}] "
            f"EM={out['metrics']['em']:.2f} "
            f"F1={out['metrics']['f1']:.2f} "
            f"R-Recall={out['metrics']['retrieval_recall']:.2f} "
            f"Total={out['timing']['total_sec']:.2f}s"
        )
    valid_ttft = [r["timing"]["ttft_sec"] for r in rows if r["timing"]["ttft_sec"] is not None]

    summary = {
        "num_examples": len(rows),
        "avg_em": mean(r["metrics"]["em"] for r in rows),
        "avg_f1": mean(r["metrics"]["f1"] for r in rows),
        "avg_retrieval_recall": mean(r["metrics"]["retrieval_recall"] for r in rows),
        "avg_retrieval_sec": mean(r["timing"]["retrieval_sec"] for r in rows),
        "avg_ttft_sec": mean(valid_ttft) if valid_ttft else None,
        "avg_generation_sec": mean(r["timing"]["generation_sec"] for r in rows),
        "avg_total_sec": mean(r["timing"]["total_sec"] for r in rows),
    }

    results = {
        "summary": summary,
        "rows": rows,
    }

    with (cfg.output_dir / "baseline_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results