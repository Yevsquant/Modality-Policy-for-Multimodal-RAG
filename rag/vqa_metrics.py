"""
VQAv2-style answer normalization and soft accuracy (min(matches/3, 1)).
Aligned with common VQA eval practice (punctuation, articles, number words).
"""

from __future__ import annotations

import re
from typing import List

# Articles and expansions (subset of official VQA eval helpers).
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
_PUNCTUATION = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?"
]


def _process_punctuation(s: str) -> str:
    s = _COMMA_STRIP.sub(r"\1\3", s)
    s = _PERIOD_STRIP.sub("", s, count=s.count("."))
    if "'" in s:
        s = s.replace("'", "")
        s = s.replace("`", "")
    for ch in _PUNCTUATION:
        if ch in s:
            s = s.replace(ch, " ")
    return s


def _process_digit_article(in_tokens: List[str]) -> List[str]:
    out: List[str] = []
    for token in in_tokens:
        if token == "":
            continue
        if token[0].isdigit() and token not in _ARTICLES:
            out.append(token)
        else:
            if token not in _ARTICLES:
                out.append(token)
    return out


def process_answer(answer: str) -> str:
    answer = str(answer).lower()
    answer = answer.replace("\n", " ").replace("\t", " ").strip()
    answer = _process_punctuation(answer)
    answer = re.sub(r"\s+", " ", answer)
    tokens = answer.split()
    tokens = _process_digit_article(tokens)
    return " ".join(tokens)


def vqa_accuracy(pred: str, ref_answers: List[str]) -> float:
    if not ref_answers:
        return 0.0
    pred_n = process_answer(pred)
    matches = sum(1 for a in ref_answers if process_answer(a) == pred_n)
    return min(matches / 3.0, 1.0)
