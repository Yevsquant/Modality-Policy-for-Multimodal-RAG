from rag.config import RAGConfig
from rag.eval_baseline import run_baseline, run_offline_judge
import requests
from typing import Dict
import os
import psutil
from pathlib import Path
import json


def _merge_lines_into_metrics(
    lines: list[str],
    alias_to_canonical: dict[str, str],
    lmcache_only: bool,
) -> tuple[Dict[str, float], int]:
    out: Dict[str, float] = {}
    parse_errors = 0
    lmcache_keys = {k for k in alias_to_canonical.values() if k.startswith("lmcache:")}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        metric_name = parts[0].split("{")[0]
        canonical_name = alias_to_canonical.get(metric_name)
        if canonical_name is None:
            continue
        if lmcache_only and canonical_name not in lmcache_keys:
            continue
        try:
            out[canonical_name] = float(parts[-1])
        except ValueError:
            parse_errors += 1
    return out, parse_errors


def scrape_prometheus_metrics(metrics_url: str = "http://127.0.0.1:8000/metrics") -> Dict[str, float]:
    """
    Scrape selected counters and gauges from a Prometheus text endpoint.

    **Model tokens processed (vLLM):** ``vllm:prompt_tokens_total`` plus
    ``vllm:generation_tokens_total`` are the cumulative prompt and generated
    token counters. ``vllm:prompt_tokens_cached`` / ``vllm:prompt_tokens_recomputed``
    and ``vllm:kv_cache_usage_perc`` add KV-cache detail.

    **LMCache:** usage bytes (``lmcache:local_cache_usage``,
    ``lmcache:remote_cache_usage``, ``lmcache:local_storage_usage``), hit rates,
    and request/token counters are merged when present on ``metrics_url``. If
    ``curl …/metrics`` has no ``lmcache:`` lines (common when multiprocess
    exposition is split), set env ``LMCACHE_METRICS_URL`` to the full URL of the
    endpoint that exposes LMCache series; those keys are merged into the same
    dict.
    """
    wanted = {
        "vllm:kv_cache_usage_perc": ["vllm:kv_cache_usage_perc"],
        "vllm:prompt_tokens_cached": [
            "vllm:prompt_tokens_cached",
            "vllm:prompt_tokens_cached_total",
        ],
        "vllm:prompt_tokens_recomputed": [
            "vllm:prompt_tokens_recomputed",
            "vllm:prompt_tokens_recomputed_total",
        ],
        "vllm:prompt_tokens_total": [
            "vllm:prompt_tokens_total",
            "vllm:prompt_tokens",
        ],
        "vllm:generation_tokens_total": [
            "vllm:generation_tokens_total",
            "vllm:generation_tokens",
        ],
        "lmcache:local_cache_usage": ["lmcache:local_cache_usage"],
        "lmcache:remote_cache_usage": ["lmcache:remote_cache_usage"],
        "lmcache:local_storage_usage": ["lmcache:local_storage_usage"],
        "lmcache:retrieve_hit_rate": ["lmcache:retrieve_hit_rate"],
        "lmcache:lookup_hit_rate": ["lmcache:lookup_hit_rate"],
        "lmcache:num_requested_tokens": [
            "lmcache:num_requested_tokens",
            "lmcache:num_requested_tokens_total",
        ],
        "lmcache:num_hit_tokens": [
            "lmcache:num_hit_tokens",
            "lmcache:num_hit_tokens_total",
        ],
        "lmcache:num_stored_tokens": [
            "lmcache:num_stored_tokens",
            "lmcache:num_stored_tokens_total",
        ],
        "lmcache:num_retrieve_requests": [
            "lmcache:num_retrieve_requests",
            "lmcache:num_retrieve_requests_total",
        ],
        "lmcache:num_store_requests": [
            "lmcache:num_store_requests",
            "lmcache:num_store_requests_total",
        ],
        "lmcache:num_lookup_requests": [
            "lmcache:num_lookup_requests",
            "lmcache:num_lookup_requests_total",
        ],
        "lmcache:num_lookup_tokens": [
            "lmcache:num_lookup_tokens",
            "lmcache:num_lookup_tokens_total",
        ],
        "lmcache:num_lookup_hits": [
            "lmcache:num_lookup_hits",
            "lmcache:num_lookup_hits_total",
        ],
        "lmcache:num_vllm_hit_tokens": [
            "lmcache:num_vllm_hit_tokens",
            "lmcache:num_vllm_hit_tokens_total",
        ],
    }
    alias_to_canonical = {
        alias: canonical
        for canonical, aliases in wanted.items()
        for alias in aliases
    }

    response = requests.get(metrics_url, timeout=5)
    lines = response.text.splitlines()
    out, _ = _merge_lines_into_metrics(lines, alias_to_canonical, lmcache_only=False)

    extra = os.getenv("LMCACHE_METRICS_URL", "").strip()
    if extra:
        try:
            r2 = requests.get(extra, timeout=5)
            lmcache_out, _ = _merge_lines_into_metrics(
                r2.text.splitlines(), alias_to_canonical, lmcache_only=True
            )
            out.update(lmcache_out)
        except Exception:
            pass

    return out

def get_host_memory_stats():
    vm = psutil.virtual_memory()
    return {
        "ram_total_gb": vm.total / (1024**3),
        "ram_used_gb": vm.used / (1024**3),
        "ram_available_gb": vm.available / (1024**3),
        "ram_percent": vm.percent,
    }

def get_dir_size_gb(path: str):
    p = Path(path)
    if not p.exists():
        return 0.0
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024**3)


def _lmcache_prometheus_subset(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in metrics.items() if k.startswith("lmcache:")}


def build_lmcache_payload(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    dir_size_before_gb: float,
    dir_size_after_gb: float,
) -> dict:
    prom_before = _lmcache_prometheus_subset(metrics_before)
    prom_after = _lmcache_prometheus_subset(metrics_after)
    keys = set(prom_before) | set(prom_after)
    delta = {
        k: float(prom_after.get(k, 0.0)) - float(prom_before.get(k, 0.0)) for k in keys
    }
    return {
        "prometheus_metrics_before": prom_before,
        "prometheus_metrics_after": prom_after,
        "prometheus_metrics_delta": delta,
        "local_disk_size_gb_before": dir_size_before_gb,
        "local_disk_size_gb_after": dir_size_after_gb,
        "local_disk_size_gb_delta": dir_size_after_gb - dir_size_before_gb,
    }


if __name__ == "__main__":
    cfg = RAGConfig()
    path = "/home/runying2/lmcache_storage"
    metrics_before = scrape_prometheus_metrics()
    host_memory_stats_before = get_host_memory_stats()
    dir_size_before = get_dir_size_gb(path)
    baseline_results = run_baseline(cfg)
    metrics_after = scrape_prometheus_metrics()
    host_memory_stats_after = get_host_memory_stats()
    dir_size_after = get_dir_size_gb(path)
    offline_results = run_offline_judge(cfg)

    lmcache_payload = build_lmcache_payload(
        metrics_before, metrics_after, dir_size_before, dir_size_after
    )

    final_results = {
        "summary": offline_results["summary"],
        "rows": offline_results["rows"],
        "lmcache": lmcache_payload,
        "system_utilization": {
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "host_memory_stats_before": host_memory_stats_before,
            "host_memory_stats_after": host_memory_stats_after,
            "lmcache_dir_size_gb_before": dir_size_before,
            "lmcache_dir_size_gb_after": dir_size_after,
            "lmcache_dir_size_gb_delta": dir_size_after - dir_size_before,
        },
    }

    output_path = cfg.output_dir / "final_results_with_utilization.json"
    judged_path = cfg.output_dir / "baseline_results_judged.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    with judged_path.open("r", encoding="utf-8") as f:
        judged_data = json.load(f)
    judged_data["lmcache"] = lmcache_payload
    with judged_path.open("w", encoding="utf-8") as f:
        json.dump(judged_data, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {output_path}")
    print(f"Updated judged results with lmcache block: {judged_path}")
    pb = lmcache_payload["prometheus_metrics_before"]
    pa = lmcache_payload["prometheus_metrics_after"]
    print(
        "LMCache Prometheus metrics (before): "
        + (json.dumps(pb, indent=2) if pb else "(none — check vLLM /metrics or LMCACHE_METRICS_URL)")
    )
    print(
        "LMCache Prometheus metrics (after): "
        + (json.dumps(pa, indent=2) if pa else "(none — check vLLM /metrics or LMCACHE_METRICS_URL)")
    )
    print(
        "LMCache block in JSON (jq):\n"
        f"  jq '.lmcache' {output_path}\n"
        f"  jq '.lmcache' {judged_path}"
    )
