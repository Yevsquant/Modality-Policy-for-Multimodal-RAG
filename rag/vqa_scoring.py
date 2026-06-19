"""Dataset-native scoring for the VQA stress test.

- Multiple-choice (V*Bench): extract the chosen letter and exact-match the gold.
- Short-answer (DocVQA): ANLS (Average Normalized Levenshtein Similarity), the
  standard DocVQA metric, taken as the best score over the gold answer list.
"""
from __future__ import annotations

import re
from typing import List, Optional


def extract_choice(text: str, valid: str = "ABCDEFGH") -> Optional[str]:
    """Pull a single multiple-choice letter from a model answer.

    Handles "(A)", "A.", "A)", "A", and "The answer is B" forms. Returns the
    uppercase letter, or None if no valid choice letter is present."""
    if not text:
        return None
    t = text.strip().upper()
    m = re.match(r"^\(?([A-Z])\)?[\.\):]?(?:\s|$)", t)
    if m and m.group(1) in valid:
        return m.group(1)
    m2 = re.search(r"\b([A-Z])\b", t)
    if m2 and m2.group(1) in valid:
        return m2.group(1)
    return None


def mc_score(pred: str, gold: str, valid: str = "ABCDEFGH") -> float:
    choice = extract_choice(pred, valid=valid)
    return float(choice is not None and choice == gold.strip().upper())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds: List[str], tau: float = 0.5) -> float:
    """Best ANLS over the gold-answer list. Similarity below tau is zeroed
    (standard DocVQA thresholding)."""
    p = (pred or "").strip().lower()
    best = 0.0
    for g in golds:
        g = (g or "").strip().lower()
        if not p and not g:
            sim = 1.0
        else:
            nl = _levenshtein(p, g) / max(len(p), len(g), 1)
            sim = 1.0 - nl
        if sim < tau:
            sim = 0.0
        best = max(best, sim)
    return best


def score_example(task: str, pred: str, record: dict) -> float:
    if task == "mc":
        return mc_score(pred, record["gold"])
    if task == "anls":
        return anls(pred, record["answers"])
    raise ValueError(f"unknown task: {task}")
