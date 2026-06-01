from __future__ import annotations

import hashlib
from math import sqrt

from mini_eqa.evaluation.answer_metrics import normalize_text


def hashed_text_embedding(text: str, dim: int = 64) -> list[float]:
    vector = [0.0] * dim
    for token in normalize_text(text).split():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += sign

    norm = sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def mean_pool_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []

    width = len(embeddings[0])
    pooled = [0.0] * width
    for row in embeddings:
        for index, value in enumerate(row):
            pooled[index] += float(value)
    return [value / len(embeddings) for value in pooled]


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
