"""Angle 1 (measured) — confirm the FLOPs reframing with real wall-clock latency.

The analytical FLOPs model (rag/flops_model.py) said FastV's advantage over input
downscale collapses on a fair *total compute* axis because FastV pays the full vision
encode + full early layers. This measures that on the actual Qwen2-VL-7B (GPTQ-Int4) by
timing two primitives and composing each condition's latency exactly as the FLOPs model
composes FLOPs:
  - t_vision(P): vision tower forward for an image yielding P merged tokens.
  - per_layer(L): one decoder-layer forward at sequence length L (= 28-layer stack / 28).
  condition latency = t_vision(P) + n_full * per_layer(T+P) + n_pruned * per_layer(T+A)
    (non-FastV: 28 full layers; FastV K=3: 3 full + 25 pruned). lm_head/embedding excluded
    (small, ~constant, same for all conditions — this isolates the method-differentiating cost).

Composing measured primitives (rather than timing answer_mc) avoids the 2-pass artifact of
the FastV implementation (eager rank-pass + sdpa deep re-run), which would double-count
early layers. Usage:
    PYTHONPATH=. python scripts/angle1_measure_latency.py --out data/vqa_stress/angle1_latency.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from rag.fastv import FastVQwen2VL, _fit_area
from rag.flops_model import TEXT_TOKENS
from rag.image_ops import downscale_to_keep
from rag.vqa_datasets import load_vstar

# (cond, img_before P, img_after A, prune_layer K)  — matches the FLOPs/data conditions
CONDS = [
    ("full", 1018, 1018, None),
    ("full+fastv0.5", 1018, 509, 3),
    ("full+fastv0.25", 1018, 255, 3),
    ("ds0.3", 297, 297, None),
    ("ds0.3+fastv0.5", 297, 148, 3),
    ("ds0.3+fastv0.25", 297, 75, 3),
    ("ds0.5", 507, 507, None),
    ("ds0.25", 258, 258, None),
]
ACC = {  # from data/vqa_stress/fastv_vstar.jsonl (n=191)
    "full": 0.607, "full+fastv0.5": 0.602, "full+fastv0.25": 0.576,
    "ds0.3": 0.518, "ds0.3+fastv0.5": 0.393, "ds0.3+fastv0.25": 0.382,
    "ds0.5": 0.550, "ds0.25": 0.476,
}
N_LAYERS = 28


def _sync():
    torch.cuda.synchronize()


@torch.no_grad()
def time_layers(fv, L, trials=20, warmup=5):
    """Median time (s) to run all 28 decoder layers on a length-L sequence."""
    tm = fv.text_model
    d = fv.model.config.get_text_config().hidden_size
    h0 = (torch.randn(1, L, d, device=fv.device, dtype=torch.float16) * 0.1)
    pos = torch.arange(L, device=fv.device).view(1, 1, L).expand(3, 1, L)
    cos, sin = tm.rotary_emb(h0, pos)
    mask = torch.triu(torch.full((1, 1, L, L), torch.finfo(torch.float16).min,
                                 device=fv.device, dtype=torch.float16), diagonal=1)
    def run():
        h = h0
        for layer in tm.layers:
            h = layer(h, attention_mask=mask, position_embeddings=(cos, sin),
                      position_ids=None, past_key_values=None, use_cache=False)
        return h
    for _ in range(warmup):
        run()
    _sync()
    ts = []
    for _ in range(trials):
        _sync(); t0 = time.perf_counter()
        run()
        _sync(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


@torch.no_grad()
def time_vision(fv, inputs, trials=20, warmup=5):
    """Median time (s) for the vision tower on these inputs."""
    model = fv.model
    pv = inputs["pixel_values"]; grid = inputs["image_grid_thw"]
    def run():
        return model.model.get_image_features(pv, grid)
    for _ in range(warmup):
        run()
    _sync()
    ts = []
    for _ in range(trials):
        _sync(); t0 = time.perf_counter()
        run()
        _sync(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/vqa_stress/angle1_latency.json")
    args = ap.parse_args()

    fv = FastVQwen2VL()
    fv._set_attn("sdpa")
    rec = load_vstar(limit=1)[0]
    base = Image.open(rec["image_path"]).convert("RGB")

    # token counts we need timings at
    p_set = sorted({c[1] for c in CONDS})                       # vision input sizes
    a_set = sorted({c[2] for c in CONDS} | {c[1] for c in CONDS})  # deep seq sizes

    # vision: build an input at each P by targeting ~P merged tokens via base_pixels fit
    t_vis = {}
    for P in p_set:
        img = downscale_to_keep(_fit_area(base, fv.base_pixels), 1.0) if P >= 1000 \
            else downscale_to_keep(_fit_area(base, fv.base_pixels), P / 1018.0)
        inputs = fv._build_inputs(base, rec["question"],
                                  input_keep=1.0 if P >= 1000 else P / 1018.0)
        n_img = int((inputs["input_ids"][0] == fv.image_token_id).sum())
        t_vis[P] = (time_vision(fv, inputs), n_img)
        print(f"[vision] target P={P} actual={n_img} -> {t_vis[P][0]*1000:.1f} ms")

    per_layer = {}
    for A in a_set:
        L = TEXT_TOKENS + A
        t28 = time_layers(fv, L)
        per_layer[A] = t28 / N_LAYERS
        print(f"[layers] L={L} (A={A}): 28-layer {t28*1000:.1f} ms -> per-layer {per_layer[A]*1000:.2f} ms")

    rows = []
    for name, P, A, K in CONDS:
        tv = t_vis[P][0]
        if K is None:
            llm = N_LAYERS * per_layer[A]            # A==P
        else:
            llm = K * per_layer[P] + (N_LAYERS - K) * per_layer[A]
        total = tv + llm
        rows.append({"cond": name, "acc": ACC[name], "deep_tok": A, "input_tok": P,
                     "vision_ms": tv * 1e3, "llm_ms": llm * 1e3, "total_ms": total * 1e3,
                     "vision_frac": tv / total})
    rows.sort(key=lambda r: r["total_ms"])

    # matched-latency: interpolate downscale-only frontier at each FastV point's latency
    ds_only = sorted([r for r in rows if "fastv" not in r["cond"]], key=lambda r: r["total_ms"])
    def ds_acc_at(ms):
        if ms <= ds_only[0]["total_ms"]:
            return ds_only[0]["acc"]
        if ms >= ds_only[-1]["total_ms"]:
            return ds_only[-1]["acc"]
        for a, b in zip(ds_only, ds_only[1:]):
            if a["total_ms"] <= ms <= b["total_ms"]:
                t = (ms - a["total_ms"]) / (b["total_ms"] - a["total_ms"])
                return a["acc"] + t * (b["acc"] - a["acc"])
        return ds_only[-1]["acc"]
    matched = [{"cond": r["cond"], "total_ms": r["total_ms"], "fastv_acc": r["acc"],
                "downscale_acc_same_latency": ds_acc_at(r["total_ms"]),
                "fastv_advantage": r["acc"] - ds_acc_at(r["total_ms"])}
               for r in rows if "fastv" in r["cond"] and "ds0.3" not in r["cond"]]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_condition": rows, "matched_latency": matched}, f, indent=2)

    print("\n=== Angle 1 MEASURED latency (Qwen2-VL-7B GPTQ, V*Bench) ===")
    print(f"{'cond':<18}{'acc':>7}{'deepTok':>9}{'inTok':>7}{'vis_ms':>9}{'llm_ms':>9}{'total_ms':>10}{'vis%':>6}")
    for r in rows:
        print(f"{r['cond']:<18}{r['acc']:>7.3f}{r['deep_tok']:>9}{r['input_tok']:>7}"
              f"{r['vision_ms']:>9.1f}{r['llm_ms']:>9.1f}{r['total_ms']:>10.1f}{100*r['vision_frac']:>6.0f}")
    print("  matched-latency: FastV advantage over downscale at the SAME latency:")
    for m in matched:
        print(f"    {m['cond']:<16} {m['total_ms']:.1f} ms: FastV {m['fastv_acc']:.3f} "
              f"vs downscale {m['downscale_acc_same_latency']:.3f} -> +{m['fastv_advantage']:.3f}")
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
