from __future__ import annotations

import numpy as np


def retrieve_topk(
    captions: list[dict],
    question: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """
    Uniformly sample captions without using the question.

    The score is set to 1.0 for every returned item. It is only a placeholder
    to match the shared retriever output format and does not represent similarity.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    if not captions:
        return []

    if top_k >= len(captions):
        indices = list(range(len(captions)))
    else:
        # Pick reproducible endpoints-spanning indices, e.g. len=6, top_k=3 -> [0, 2, 5].
        indices = np.linspace(0, len(captions) - 1, num=top_k, dtype=int).tolist()

    return [
        {
            "frame_id": captions[idx]["frame_id"],
            "caption": captions[idx]["caption"],
            "score": 1.0,
        }
        for idx in indices
    ]
