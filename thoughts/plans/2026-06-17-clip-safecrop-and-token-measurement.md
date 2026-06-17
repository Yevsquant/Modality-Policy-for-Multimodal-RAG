# CLIP-Safecrop 裁剪 + 真实视觉Token度量 实现计划

## Overview

把 "决定怎么裁剪" 的信号从**昂贵的 7B Qwen2-VL 全注意力前向**换成**已有的廉价 CLIP 区域打分**，并保持 safecrop 的**单框裁剪、保留空间位置关系**的输出（不做 montage 重排，不做 cluster 多子图）。同时补上项目主目标真正缺失的东西：**用目标生成模型（Qwen3-Omni）自己的 processor 来统计视觉 token**，把 "平均 token 降幅" 汇总进结果并画图。最后加入一条平庸基线（整图等比降采样到同样 token 预算）和一个 keep_ratio 扫描，画出 "token vs 准确率" 权衡曲线，以证明方法是否真的优于平庸做法。

本计划对应 [REFLECTION_REVIEW_zh.md](../../REFLECTION_REVIEW_zh.md) 的建议 A（度量对）、B（去掉 7B）、D（权衡曲线）、F（平庸基线），并按用户要求**否决建议 C 的 cluster 拼接**（拼接会破坏位置关系，下游 visual encoder 需要它；safecrop 单框正是为保护位置关系）。

## Current State Analysis

- 主回答模型是 Qwen3-Omni-30B（[config.py:38](../../rag/config.py)）。
- 现有三种图像裁剪模式（[pruner.py:128](../../rag/pruner.py)）：
  - `visual_patch_pruning`：**用 CLIP** 给 4×4 tile 打分（[`_score_tiles`](../../rag/pruner.py:351) + [`_extract_grid_tiles`](../../rag/pruner.py:334)），保留 top-k，但用 [`_save_montage`](../../rag/pruner.py:393) **把保留的 tile 重排到新画布上 → 破坏空间位置关系**。
  - `safecrop_pruning`：用 **7B Qwen2-VL** 注意力（[`Qwen2VLCATPBoundingBoxCropper`](../../rag/qwen2vl_catp_pruner_v2.py:204)，`eager` 注意力 + `output_attentions` + `output_hidden_states`，最烧资源）→ 输出**单个 bbox 裁剪**（[`_safe_crop`](../../rag/qwen2vl_catp_pruner_v2.py:274)），保留位置关系。
  - `cluster_pruning`：同样用 7B → 输出多子图（用户已否决）。
- **关键缺口**：`tokens_before/after` 是在被裁掉的 7B 代理网格上算的（[pruner.py:318](../../rag/pruner.py)），且 [`aggregate_summary`](../../rag/metrics.py:134) **从不汇总视觉 token**，只汇总延迟和质量。两张结果图也没画 token。
- 管线在 [`run_one`](../../rag/query_pipeline.py:272) 里把每张图编码进 `content`：`".jpg" in path` 走单图分支，目录走多子图分支（[query_pipeline.py:289](../../rag/query_pipeline.py)）。单框裁剪输出单 jpg，天然走单图分支，**无需改 prompt_builder**。
- 管线会**覆盖** `q["local_img_path"]` 为裁剪后路径（[pruner.py:288](../../rag/pruner.py) / [pruner.py:322](../../rag/pruner.py)），因此 "裁剪前 token" 必须在 `pruner.apply()` 之前抓取原图路径。

### Key Discoveries:
- CLIP 区域打分链路已存在且可直接复用：[`_extract_grid_tiles`](../../rag/pruner.py:334) 返回 `(tiles, boxes)`，`boxes` 已是像素坐标 `[left,top,right,bottom]`；[`_score_tiles`](../../rag/pruner.py:351) 已用 CLIP 算 query↔tile 相似度。新模式 ≈ "用这两个已有函数选 tile → 取被选 tile 的外接框 → 裁原图一次"。
- 单图保存已有：[`_save_pruned_image`](../../rag/pruner.py:412) + [`_pruned_output_path`](../../rag/pruner.py:376)（保留 `.jpg` 后缀）。
- CLIP 仅在 `mode == "visual_patch_pruning"` 时加载（[pruner.py:175](../../rag/pruner.py)）；新模式只需把这个条件扩一项。
- Qwen token 数 = `prod(image_grid_thw) // merge_size**2`（见 [pruner_v2.py:544](../../rag/qwen2vl_catp_pruner_v2.py)），可用目标模型的 `image_processor` 直接复现，**只需加载 processor 配置，不加载 30B 权重**。

## Desired End State

- 新增 `clip_safecrop` 模式：CLIP 选区 + 单框裁剪，保留位置关系，**不加载 7B**。设为默认 `pruning_mode`，使主链路彻底摆脱 7B。
- 旧模式 `safecrop_pruning` / `cluster_pruning` / `visual_patch_pruning` **保留**，供论文 A/B 对比。
- benchmark 结果 summary 中出现 `avg_visual_tokens_before / after / reduction_pct`，且这些数字由**目标模型 processor** 算出；并产出一张 token 前后对比图。
- 新增 `downscale_baseline` 平庸基线模式 + `scripts/sweep_keep_ratio.py`，产出一张 "目标模型真实 token 数 vs judge_correct" 的权衡曲线，对比 clip_safecrop 与降采样基线。

验证方式见各 Phase 的 Success Criteria。

## What We're NOT Doing

- **不做 cluster 拼接 / 多子图重排**（用户明确否决：会破坏 visual encoder 依赖的空间位置关系）。
- **不删除** 现有 7B 模式与 `qwen2vl_catp_pruner_v2.py`（用户要求保留做对比）。
- **不改** `prompt_builder.py`（单 jpg 走已有单图分支即可）。
- 不重构磁盘缓存逻辑、检索逻辑、judge 逻辑。
- 不引入新的检索/嵌入模型；CLIP 复用现有 `image_embedding_model`。
- 不追求在线把 CLIP 也省掉（离线预裁剪等更激进优化留作后续）。

## Implementation Approach

三个相互独立、可分别验证的 Phase：

1. **Phase 1** 只负责 "产出正确的单框裁剪图"（功能正确、保留布局、无 7B）。
2. **Phase 2** 只负责 "用目标模型 processor 正确度量 token 并汇总/画图"（度量正确，单一可信来源）。
3. **Phase 3** 只负责 "科学对照"（平庸基线 + keep_ratio 扫描 + 权衡曲线）。

每个 Phase 都尽量复用已有函数、改动面最小。

---

## Phase 1: 新增 `clip_safecrop` 模式（CLIP 选区 + 单框裁剪，保留布局，去掉 7B）

### Overview
让 safecrop 风格的单框裁剪由 CLIP 区域打分驱动，而不是 7B 注意力。复用 `_extract_grid_tiles` / `_score_tiles` / `_save_pruned_image`，只新增 "选 tile → 取外接框 → 裁原图" 这一小段。

### Changes Required:

#### 1. 模式注册与 CLIP 初始化
**File**: `rag/pruner.py`
**Changes**:
- `SUPPORTED_MODES` 增加 `"clip_safecrop"`。
- CLIP 加载条件由 `if mode == "visual_patch_pruning":` 改为 `if mode in ("visual_patch_pruning", "clip_safecrop"):`（[pruner.py:175](../../rag/pruner.py)）。

#### 2. 纯几何辅助函数（可单测，不依赖模型）
**File**: `rag/pruner.py`（模块级或类级静态方法）
```python
def _bbox_union(boxes: List[List[int]]) -> List[int]:
    """被选 tile 像素框 [l,t,r,b] 的外接框。"""
    lefts   = [b[0] for b in boxes]
    tops    = [b[1] for b in boxes]
    rights  = [b[2] for b in boxes]
    bottoms = [b[3] for b in boxes]
    return [min(lefts), min(tops), max(rights), max(bottoms)]
```

#### 3. CLIP-safecrop 主逻辑
**File**: `rag/pruner.py`（新增方法，仿照 [`_patch_prune_image`](../../rag/pruner.py:255)）
```python
def _clip_safecrop_image(self, query: str, q: Dict) -> Tuple[Dict, int, int]:
    img_path = q.get("local_img_path")
    before = self.patch_grid_rows * self.patch_grid_cols
    after = before
    if not img_path or not Path(img_path).exists():
        q["visual_pruning"] = {"mode": self.mode, "skipped": True,
                               "reason": "missing_image",
                               "tokens_before": before, "tokens_after": after}
        return q, before, after

    image = Image.open(img_path).convert("RGB")
    tiles, boxes = self._extract_grid_tiles(image)          # 复用
    scores = self._score_tiles(query, tiles)                # 复用 CLIP

    keep_n = max(self.min_visual_tokens, int(len(tiles) * self.keep_ratio))
    keep_n = min(max(1, keep_n), len(tiles))
    keep_idx = np.argsort(-scores)[:keep_n].tolist()

    crop_box = _bbox_union([boxes[i] for i in keep_idx])    # 单个外接框 → 保留布局
    cropped = image.crop(tuple(crop_box))
    pruned_path = self._save_pruned_image(                  # 复用，存单张 jpg
        image_path=Path(img_path), image=cropped, mode=self.mode, quote=q)

    # 面积比的粗略估计；Phase 2 用目标模型 processor 给出权威数字
    area_ratio = (cropped.width * cropped.height) / max(1, image.width * image.height)
    after = max(1, round(before * area_ratio))

    q["local_img_path"] = str(pruned_path)
    q["visual_pruning"] = {"mode": self.mode, "tokens_before": before,
                           "tokens_after": after, "crop_box": crop_box,
                           "tag_hash": q.get("tag_hash")}
    return q, before, after
```

#### 4. 在 `apply()` 中接线
**File**: `rag/pruner.py`（[apply](../../rag/pruner.py:188) 内新增分支，仿照 visual_patch_pruning 的循环）
```python
elif self.mode == "clip_safecrop":
    processed, visual_before, visual_after = [], 0, 0
    for q in img_quotes:
        new_q, before_i, after_i = self._clip_safecrop_image(query, q)
        processed.append(new_q)
        visual_before += before_i
        visual_after += after_i
    pruned_images = processed
```

#### 5. 默认模式切到 clip_safecrop（让主链路不再加载 7B）
**File**: `rag/config.py`
**Changes**: `pruning_mode: str = "clip_safecrop"`（[config.py:26](../../rag/config.py)）；注释里的可选集合追加 `"clip_safecrop"`。

#### 6. 最小单测
**File**: `tests/test_pruner_geometry.py`（新建）
- 测 `_bbox_union`：给定若干像素框，返回正确外接框；单框时等于自身。

### Success Criteria:

#### Automated Verification:
- [ ] 模块可导入：`PYTHONPATH=. python -c "import rag.pruner"`
- [ ] 新模式已注册：`PYTHONPATH=. python -c "from rag.pruner import RetrievalPruner; assert 'clip_safecrop' in RetrievalPruner.SUPPORTED_MODES"`
- [ ] 几何单测通过：`PYTHONPATH=. python -m pytest tests/test_pruner_geometry.py -q`
- [ ] 默认模式生效：`PYTHONPATH=. python -c "from rag.config import RAGConfig; assert RAGConfig().pruning_mode=='clip_safecrop'"`

#### Manual Verification (需 GPU + vLLM server)：
- [ ] 跑 `clip_safecrop` benchmark：`PYTHONPATH=. python scripts/run_mmodcrag_benchmark.py --eval-slice-start 0 --eval-slice-stop 50 --max-examples 0`
- [ ] `pruned_images/` 下产出的是**单张 .jpg**（非 cluster 目录），且裁剪图保留原图相对位置（肉眼看是原图的一个连续矩形区域，而非拼贴）。
- [ ] 运行期间**不加载 Qwen2-VL-7B**（无对应显存占用 / 加载日志）。
- [ ] `clip_safecrop` 的 `avg_total_sec` **低于 baseline**（去掉 7B 后实时裁剪应快于旧的 on-the-fly，亦不应慢于 no_pruning）。

**Implementation Note**: 本 Phase 自动验证通过后，暂停等待人工确认手动测试通过，再进入 Phase 2。

---

## Phase 2: 用目标模型 processor 正确度量视觉 token + 汇总 + 画图

### Overview
项目主目标的唯一可信度量：用 **Qwen3-Omni 自己的 image processor** 数 token。在管线里对 "原图（before）" 与 "真正发出去的图（after）" 各数一次，写进每行结果，汇总进 summary，并画一张前后对比图。

### Changes Required:

#### 1. Token 计数器（只加载 processor，不加载 30B 权重）
**File**: `rag/visual_token_counter.py`（新建）
```python
from PIL import Image
from transformers import AutoProcessor

class VisualTokenCounter:
    def __init__(self, model_name: str):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.image_processor = self.processor.image_processor
        self.merge_size = int(getattr(self.image_processor, "merge_size", 2))

    def count(self, image: Image.Image) -> int:
        out = self.image_processor(images=[image.convert("RGB")], return_tensors="pt")
        t, h, w = out["image_grid_thw"][0].tolist()
        return int(t * h * w // (self.merge_size ** 2))

    def count_path(self, path: str) -> int:
        return self.count(Image.open(path))
```
> 若目标模型 processor 接口字段名不同（如 omni 变体），在 `count()` 内做一次兼容回退（拿不到 `image_grid_thw` 时回退到按像素面积估算并打日志）。保持简单：先按 Qwen 标准字段实现。

#### 2. 在管线里抓 "原图路径" 并度量
**File**: `rag/query_pipeline.py`（[run_one](../../rag/query_pipeline.py:272)）
**Changes**:
- 懒加载一个 `VisualTokenCounter(self.cfg.vlm_model_name)`（构造函数里或首次使用时）。
- **在 `self.pruner.apply()` 之前**抓取原图路径（因为 apply 会覆盖 `local_img_path`）：
  ```python
  original_paths = {q.get("quote_id"): q.get("local_img_path")
                    for q in retrieval["selected_img_quotes"]}
  ```
- 在组完 `content` 之后，统计：
  - `tokens_after` = 对 `pruned_retrieval["selected_img_quotes"]` 里**实际发出的图**（单 jpg 数一次；目录则对每张子图求和）逐一 `counter.count_path` 求和；
  - `tokens_before` = 对应 `original_paths[quote_id]`（原始整图）求和。
  - 把结果挂到返回值：`out["visual_tokens"] = {"before": tokens_before, "after": tokens_after}`。
- 缓存命中的图（`cached_img_quotes`）只统计 after（before 属于上一次运行），在 `visual_tokens` 里单列 `from_cache_after`，不污染 before/after 降幅口径。

#### 3. 汇总进 summary
**File**: `rag/metrics.py`（[aggregate_summary](../../rag/metrics.py:134)）
```python
tb = [r["visual_tokens"]["before"] for r in rows if r.get("visual_tokens")]
ta = [r["visual_tokens"]["after"]  for r in rows if r.get("visual_tokens")]
if tb and ta:
    summary["avg_visual_tokens_before"] = sum(tb) / len(tb)
    summary["avg_visual_tokens_after"]  = sum(ta) / len(ta)
    summary["avg_visual_tokens_reduction_pct"] = 100.0 * (1 - sum(ta) / max(1, sum(tb)))
```

#### 4. 前后对比图
**File**: `scripts/plot_visual_tokens.py`（新建，简单）
- 读取一个或多个 `baseline_results_judged.json`（按方法），画 before/after 平均 token 柱状图，存到 `imgs/VisualTokensByMethods.png`。

### Success Criteria:

#### Automated Verification:
- [ ] 计数器可导入：`PYTHONPATH=. python -c "import rag.visual_token_counter"`
- [ ] summary 含新字段（用一个内联假 rows 调 `aggregate_summary` 断言键存在）：`PYTHONPATH=. python -c "from rag.metrics import aggregate_summary; s=aggregate_summary([{'metrics':{},'timing':{'retrieval_sec':0,'ttft_sec':0.1,'generation_sec':0,'total_sec':0},'visual_tokens':{'before':100,'after':40}}]); assert abs(s['avg_visual_tokens_reduction_pct']-60)<1e-6"`

#### Manual Verification (需 GPU + 目标模型 processor 可下载)：
- [ ] 计数器能加载目标模型 processor 并对一张样例图返回合理 token 数（与 vLLM 实际用量量级一致）。
- [ ] 跑完 benchmark 后，`baseline_results_judged.json` 的 summary 里出现 `avg_visual_tokens_reduction_pct`，且 clip_safecrop 下该值 **> 0**。
- [ ] `imgs/VisualTokensByMethods.png` 生成且 after < before。

**Implementation Note**: 本 Phase 自动验证通过后，暂停等待人工确认（尤其是 processor 能正确加载、token 量级合理），再进入 Phase 3。

---

## Phase 3: 平庸基线 `downscale_baseline` + keep_ratio 扫描 + 权衡曲线

### Overview
加入诚实对照：整图等比降采样到与裁剪同样的 token 预算（不看 query，全布局保留只是降分辨率）。再扫描 keep_ratio，画 "真实 token vs judge_correct" 曲线，看 clip_safecrop 是否真的优于降采样。

### Changes Required:

#### 1. 降采样基线模式
**File**: `rag/pruner.py`
**Changes**:
- `SUPPORTED_MODES` 增加 `"downscale_baseline"`。
- 新增方法（token 数 ~∝ 像素面积，故线性边长缩放 `sqrt(keep_ratio)` 即可命中约 `keep_ratio` 的 token 预算）：
  ```python
  def _downscale_image(self, q: Dict) -> Tuple[Dict, int, int]:
      img_path = q.get("local_img_path")
      before = self.patch_grid_rows * self.patch_grid_cols
      if not img_path or not Path(img_path).exists():
          q["visual_pruning"] = {"mode": self.mode, "skipped": True,
                                 "reason": "missing_image",
                                 "tokens_before": before, "tokens_after": before}
          return q, before, before
      image = Image.open(img_path).convert("RGB")
      factor = math.sqrt(self.keep_ratio)
      new_size = (max(1, int(image.width * factor)), max(1, int(image.height * factor)))
      resized = image.resize(new_size)
      path = self._save_pruned_image(Path(img_path), resized, self.mode, q)
      after = max(1, round(before * self.keep_ratio))
      q["local_img_path"] = str(path)
      q["visual_pruning"] = {"mode": self.mode, "tokens_before": before, "tokens_after": after}
      return q, before, after
  ```
- 在 `apply()` 增加 `elif self.mode == "downscale_baseline":` 循环分支（与 clip_safecrop 同构）。

#### 2. keep_ratio 扫描脚本
**File**: `scripts/sweep_keep_ratio.py`（新建）
- 对 `keep_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]`，逐个设 `cfg.pruning_keep_ratio`，复用 [`run_rag_benchmark`](../../rag/eval.py:17) + [`run_rag_benchmark_offline_judge`](../../rag/eval.py:71)，记录每点 `(avg_visual_tokens_after, avg_judge_correct, avg_total_sec)`。
- 对 `clip_safecrop` 与 `downscale_baseline` 各扫一遍，写 `data/mmdocrag/analysis/keep_ratio_sweep.json`。

#### 3. 权衡曲线图
**File**: `scripts/sweep_keep_ratio.py`（同脚本输出）
- 横轴 `avg_visual_tokens_after`，纵轴 `avg_judge_correct`，两条线（clip_safecrop / downscale_baseline），存 `imgs/TokenVsAccuracy.png`。

### Success Criteria:

#### Automated Verification:
- [ ] 新模式已注册：`PYTHONPATH=. python -c "from rag.pruner import RetrievalPruner; assert 'downscale_baseline' in RetrievalPruner.SUPPORTED_MODES"`
- [ ] 降采样比例数学正确（单测 `sqrt(keep_ratio)` 边长 → 面积比 ≈ keep_ratio）：加进 `tests/test_pruner_geometry.py` 并 `pytest -q` 通过。
- [ ] 扫描脚本可被导入/`--help` 不报错：`PYTHONPATH=. python scripts/sweep_keep_ratio.py --help`

#### Manual Verification (需 GPU + vLLM)：
- [ ] 跑 `downscale_baseline`，Phase 2 的真实 token 计数确认其 after ≈ keep_ratio × before（验证 "同 token 预算" 的公平性）。
- [ ] `imgs/TokenVsAccuracy.png` 生成；从曲线能读出 clip_safecrop 在相同 token 预算下的准确率是否高于降采样基线（这是判断方法是否值得的关键结论）。
- [ ] 在曲线上能指认一个 "拐点" keep_ratio（token 大幅下降而准确率几乎不掉的点）。

---

## Testing Strategy

### Unit Tests (CPU，无需模型/网络):
- `_bbox_union`：多框外接、单框、共线框。
- 降采样面积比 ≈ keep_ratio。
- `aggregate_summary` 的 token 降幅计算。

### Integration Tests (需 GPU + vLLM server):
- clip_safecrop 端到端跑通，产出单 jpg、summary 含 token 降幅、总延迟低于 baseline。
- downscale_baseline 端到端跑通，after token ≈ keep_ratio×before。

### Manual Testing Steps:
1. 启动 vLLM（README 命令），跑 clip_safecrop benchmark，肉眼检查裁剪图是连续矩形区域（保留布局）。
2. 检查 summary JSON 的 `avg_visual_tokens_reduction_pct`。
3. 运行 `scripts/plot_visual_tokens.py` 与 `scripts/sweep_keep_ratio.py`，检查两张图。
4. 对比三方法（baseline / clip_safecrop / downscale_baseline）的 token、准确率、延迟。

## Performance Considerations

- 去掉 7B `eager` + `output_attentions/hidden_states` 前向是主要提速来源；CLIP 对 16 个 tile 的一次批量前向比 7B 便宜约 1~2 个数量级。
- `VisualTokenCounter` 只跑 image_processor（CPU 即可），不跑模型，开销极小；processor 只加载一次。
- bbox 为被选 tile 的外接框，最坏情况退化为整图（无收益但不报错，与旧 safecrop 语义一致）；真实收益在相关内容空间集中时显现。

## Migration Notes

- 旧模式与 `qwen2vl_catp_pruner_v2.py` 全部保留，仅默认模式切到 `clip_safecrop`；要复跑 7B 对比，把 `pruning_mode` 改回 `safecrop_pruning` 即可。
- 已有磁盘缓存里存的是旧裁剪图路径；切换模式后建议清空 `image_prune_cache.json` 与 `pruned_images/` 以免混用不同模式的缓存（在缓存命中阈值场景下手动清理即可，代码无需改）。

## References

- 复盘与建议：[REFLECTION_REVIEW_zh.md](../../REFLECTION_REVIEW_zh.md)
- 复用的 CLIP 打分链路：[rag/pruner.py:334](../../rag/pruner.py)（`_extract_grid_tiles`）、[rag/pruner.py:351](../../rag/pruner.py)（`_score_tiles`）
- 单图保存：[rag/pruner.py:412](../../rag/pruner.py)（`_save_pruned_image`）
- 管线接入点：[rag/query_pipeline.py:272](../../rag/query_pipeline.py)（`run_one`）
- 汇总函数：[rag/metrics.py:134](../../rag/metrics.py)（`aggregate_summary`）
- 被替换的 7B 实现（保留）：[rag/qwen2vl_catp_pruner_v2.py:204](../../rag/qwen2vl_catp_pruner_v2.py)
