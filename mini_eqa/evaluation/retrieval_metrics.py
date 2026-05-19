from __future__ import annotations

from typing import Sequence


def recall_at_k(
    retrieved_frame_ids: Sequence[str],
    gold_frame_ids: Sequence[str],
) -> float:
    """
    Recall@K measures how many gold evidence frames are retrieved.

    Example:
        retrieved = ["frame_004", "frame_005", "frame_006"]
        gold = ["frame_004"]

        Recall@3 = 1 / 1 = 1.0

    If there are multiple gold frames:
        retrieved = ["frame_002"]
        gold = ["frame_002", "frame_005"]

        Recall@1 = 1 / 2 = 0.5
    """
    gold_set = set(gold_frame_ids)
    if not gold_set:
        return 0.0

    retrieved_set = set(retrieved_frame_ids)
    hits = retrieved_set & gold_set

    return len(hits) / len(gold_set)


def precision_at_k(
    retrieved_frame_ids: Sequence[str],
    gold_frame_ids: Sequence[str],
) -> float:
    """
    Precision@K measures how many retrieved frames are actually useful evidence.

    Example:
        retrieved = ["frame_004", "frame_005", "frame_006"]
        gold = ["frame_004"]

        Precision@3 = 1 / 3 = 0.3333
    """
    if not retrieved_frame_ids:
        return 0.0

    gold_set = set(gold_frame_ids)
    retrieved_set = set(retrieved_frame_ids)
    hits = retrieved_set & gold_set

    return len(hits) / len(retrieved_frame_ids)


def mrr(
    retrieved_frame_ids: Sequence[str],
    gold_frame_ids: Sequence[str],
) -> float:
    """
    MRR means Mean Reciprocal Rank.

    It checks the rank of the first correct retrieved frame.

    Example:
        retrieved = ["frame_004", "frame_005", "frame_006"]
        gold = ["frame_004"]

        First correct evidence appears at rank 1.
        MRR = 1 / 1 = 1.0

    Example:
        retrieved = ["frame_005", "frame_006", "frame_004"]
        gold = ["frame_004"]

        First correct evidence appears at rank 3.
        MRR = 1 / 3 = 0.3333
    """
    gold_set = set(gold_frame_ids)
    if not gold_set:
        return 0.0

    for rank, frame_id in enumerate(retrieved_frame_ids, start=1):
        if frame_id in gold_set:
            return 1.0 / rank

    return 0.0