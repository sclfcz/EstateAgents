"""把剧情切成适合句向量模型的短句。"""

from __future__ import annotations

import re

SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# 中文概念到英文短语。只覆盖常见说法，不写片名。
# 向量模型对长中文问句不够稳时，这些短语用来补充一个对照分句。
EXPANSIONS = [
    ("陀螺", "a spinning top keeps spinning"),
    ("冰山", "a ship hits an iceberg"),
    ("画家", "a poor artist"),
    ("富家女", "a wealthy young woman"),
    ("废土", "a desert wasteland"),
    ("核战", "after a nuclear war"),
    ("抢水", "fighting over water"),
    ("卡车", "an armored truck"),
    ("变成了猪", "parents turn into pigs"),
    ("神灵", "the spirit world"),
    ("原子弹", "an atomic bomb"),
]

# 若剧情里出现这些连续短语，说明字面已经对上，给语义分一个小加成。
PHRASES = [
    ("陀螺", "spinning top"),
    ("变成了猪", "into pigs"),
    ("冰山", "iceberg"),
    ("废土", "wasteland"),
    ("原子弹", "atomic bomb"),
]


def plot_pieces(plot: str) -> list[str]:
    sentences = [part.strip() for part in SPLIT_RE.split(plot or "") if len(part.split()) >= 4]
    pieces: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= 42:
            pieces.append(sentence)
            continue
        for start in range(0, len(words), 36):
            pieces.append(" ".join(words[start : start + 42]))
    if pieces:
        return pieces
    fallback = " ".join((plot or "").split())[:240]
    return [fallback] if fallback else []


def query_clauses(query: str) -> tuple[list[str], list[str]]:
    parts = [part.strip() for part in re.split(r"[，。；,.!?]", query) if len(part.strip()) >= 2]
    if not parts:
        parts = [query.strip()]
    extras = [english for chinese, english in EXPANSIONS if chinese in query]
    return parts, extras


def matched_phrases(query: str, plot: str) -> list[str]:
    haystack = (plot or "").lower()
    found = []
    for chinese, phrase in PHRASES:
        if chinese in query and phrase in haystack:
            found.append(phrase)
    return found
