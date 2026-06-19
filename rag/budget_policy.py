"""Learned query-conditioned visual-token budget policy (Phase 2).

The policy chooses, per (image, question), the smallest downscale keep_ratio that
still answers correctly — so visual tokens are spent only where the query needs
resolution. `trim_downscale`/plain downscale are the static, query-agnostic special
case (one budget for everyone); this generalizes it.

The whole evaluation is *free* (no VLM calls): Phase 0 already recorded, per example,
the answer model's score AND token count at every budget in the keep-ladder
(`data/vqa_stress/vstar_downscale.jsonl`). So a budget choice is scored by *looking up*
the recorded (score, tokens) at the chosen keep.

Output-space framing (plan Q3, option 1 — global scalar budget):
  For each budget k in the ladder, predict whether the example is still correct at k.
  At inference, walk the ladder cheapest-first and pick the smallest k whose predicted
  P(correct@k) >= threshold; fall back to the most expensive budget if none qualify.
  Sweeping the threshold traces the policy's (accuracy vs avg-tokens) frontier.

This module holds only the *pure* logic (oracle budgets, the threshold-sweep budget
selector, frontier construction). Feature extraction and the sklearn fit live in the
driver script so this stays importable without torch/sklearn.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def oracle_budget(
    keeps: Sequence[float], scores_by_keep: Dict[float, float], threshold: float = 0.5
) -> Optional[float]:
    """Smallest keep that is still correct (score >= threshold).

    Returns None if the example is wrong even at the largest (most expensive) keep —
    i.e. no budget in the ladder solves it.
    """
    correct = [k for k in keeps if scores_by_keep[k] >= threshold]
    return min(correct) if correct else None


def select_budget(
    keeps_cheap_first: Sequence[float],
    pred_correct_prob: Dict[float, float],
    threshold: float,
) -> float:
    """Pick the cheapest budget whose predicted P(correct) clears `threshold`.

    `keeps_cheap_first` must be ordered cheapest (smallest keep) -> most expensive.
    If no budget clears the threshold, fall back to the most expensive budget (the
    safest, highest-resolution option).
    """
    if not keeps_cheap_first:
        raise ValueError("keeps_cheap_first must be non-empty")
    for k in keeps_cheap_first:
        if pred_correct_prob[k] >= threshold:
            return k
    return keeps_cheap_first[-1]


def evaluate_policy_at_threshold(
    examples: Sequence[dict],
    keeps_cheap_first: Sequence[float],
    threshold: float,
) -> Tuple[float, float]:
    """Mean (recorded) accuracy and mean tokens of the policy at one threshold.

    Each example dict must provide:
      - "scores": {keep: recorded score 0/1}
      - "tokens": {keep: recorded token count}
      - "probs":  {keep: predicted P(correct@keep)}  (held-out, from CV)
    """
    accs: List[float] = []
    toks: List[float] = []
    for ex in examples:
        k = select_budget(keeps_cheap_first, ex["probs"], threshold)
        accs.append(ex["scores"][k])
        toks.append(ex["tokens"][k])
    n = len(examples)
    return (sum(accs) / n, sum(toks) / n)


def policy_per_example(
    examples: Sequence[dict],
    keeps_cheap_first: Sequence[float],
    threshold: float,
) -> Tuple[List[float], List[float]]:
    """Per-example (score, tokens) the policy realizes at one threshold.

    Returns two aligned lists (scores, tokens), example order preserved — for paired
    CIs against another policy/baseline evaluated on the same examples.
    """
    scores: List[float] = []
    toks: List[float] = []
    for ex in examples:
        k = select_budget(keeps_cheap_first, ex["probs"], threshold)
        scores.append(ex["scores"][k])
        toks.append(ex["tokens"][k])
    return scores, toks


def static_frontier(
    examples: Sequence[dict], keeps: Sequence[float]
) -> List[Tuple[float, float, float]]:
    """The query-agnostic baseline: everyone gets the same keep.

    Returns [(keep, mean_acc, mean_tokens)] over the ladder.
    """
    out = []
    n = len(examples)
    for k in keeps:
        acc = sum(ex["scores"][k] for ex in examples) / n
        tok = sum(ex["tokens"][k] for ex in examples) / n
        out.append((k, acc, tok))
    return out


def oracle_frontier_point(
    examples: Sequence[dict], keeps: Sequence[float], threshold: float = 0.5
) -> Tuple[float, float]:
    """The upper bound: each example at its own minimum-sufficient budget.

    Unsolved examples (wrong at every budget) take the most expensive budget and
    count as incorrect there — the best any budget policy could do.
    """
    keeps_desc = sorted(keeps, reverse=True)
    most_expensive = keeps_desc[0]
    accs, toks = [], []
    for ex in examples:
        ob = oracle_budget(keeps_desc, ex["scores"], threshold)
        k = ob if ob is not None else most_expensive
        accs.append(ex["scores"][k])
        toks.append(ex["tokens"][k])
    n = len(examples)
    return (sum(accs) / n, sum(toks) / n)


def policy_frontier(
    examples: Sequence[dict],
    keeps_cheap_first: Sequence[float],
    thresholds: Sequence[float],
) -> List[Tuple[float, float, float]]:
    """Sweep thresholds -> [(threshold, mean_acc, mean_tokens)] for the learned policy."""
    out = []
    for t in thresholds:
        acc, tok = evaluate_policy_at_threshold(examples, keeps_cheap_first, t)
        out.append((t, acc, tok))
    return out
