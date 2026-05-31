from __future__ import annotations

import random
from typing import Iterable

from mini_eqa.evaluation.answer_metrics import normalize_text


def _tokenize(text: str) -> set[str]:
    return set(normalize_text(text).split())


def rank_frames_for_question(captions: list[dict], question: str) -> list[dict]:
    question_tokens = _tokenize(question)
    ranked = []
    for index, item in enumerate(captions):
        caption_tokens = _tokenize(item["caption"])
        overlap = len(question_tokens & caption_tokens)
        ranked.append(
            {
                "frame_id": item["frame_id"],
                "caption": item["caption"],
                "score": float(overlap),
                "rank_source": "lexical_overlap",
                "original_index": index,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["original_index"], item["frame_id"]))
    return ranked


def _sample_frames(pool: Iterable[dict], sample_size: int, rng: random.Random) -> list[dict]:
    pool_list = list(pool)
    if len(pool_list) <= sample_size:
        return pool_list[:sample_size]
    indices = sorted(rng.sample(range(len(pool_list)), sample_size))
    return [pool_list[index] for index in indices]


def _uniform_frames(ranked_frames: list[dict], sample_size: int) -> list[dict]:
    if len(ranked_frames) <= sample_size:
        return ranked_frames[:sample_size]

    last_index = len(ranked_frames) - 1
    indices = []
    for step in range(sample_size):
        index = round(step * last_index / (sample_size - 1))
        indices.append(index)
    return [ranked_frames[index] for index in indices]


def _middle_slice(ranked_frames: list[dict]) -> list[dict]:
    if len(ranked_frames) <= 3:
        return ranked_frames
    third = max(1, len(ranked_frames) // 3)
    start = third
    end = max(start + 1, len(ranked_frames) - third)
    return ranked_frames[start:end]


def _low_rank_slice(ranked_frames: list[dict]) -> list[dict]:
    if len(ranked_frames) <= 3:
        return ranked_frames
    third = max(1, len(ranked_frames) // 3)
    return ranked_frames[-third:]


def _perturb_top_frames(
    ranked_frames: list[dict],
    sample_size: int,
    rng: random.Random,
) -> list[dict]:
    base = ranked_frames[:sample_size]
    if len(ranked_frames) <= sample_size:
        return base

    replacement_pool = ranked_frames[sample_size : max(sample_size + 3, len(ranked_frames))]
    replacement = replacement_pool[rng.randrange(len(replacement_pool))]
    replaced_index = rng.randrange(len(base))

    perturbed = list(base)
    perturbed[replaced_index] = replacement
    seen = set()
    unique = []
    for item in perturbed + ranked_frames:
        frame_id = item["frame_id"]
        if frame_id in seen:
            continue
        seen.add(frame_id)
        unique.append(item)
        if len(unique) == sample_size:
            break
    return unique


def build_candidate_sets(
    captions: list[dict],
    question: str,
    sample_size: int = 3,
    seed: int = 0,
) -> list[dict]:
    ranked_frames = rank_frames_for_question(captions=captions, question=question)
    rng = random.Random(seed)

    candidates = [
        {
            "candidate_type": "reqa_top3",
            "frames": ranked_frames[:sample_size],
        },
        {
            "candidate_type": "reqa_top6_random3",
            "frames": _sample_frames(ranked_frames[:6], sample_size, rng),
        },
        {
            "candidate_type": "uniform3",
            "frames": _uniform_frames(ranked_frames, sample_size),
        },
        {
            "candidate_type": "random3",
            "frames": _sample_frames(ranked_frames, sample_size, rng),
        },
        {
            "candidate_type": "reqa_perturbed3",
            "frames": _perturb_top_frames(ranked_frames, sample_size, rng),
        },
        {
            "candidate_type": "middle_rank_random3",
            "frames": _sample_frames(_middle_slice(ranked_frames), sample_size, rng),
        },
        {
            "candidate_type": "low_rank_random3",
            "frames": _sample_frames(_low_rank_slice(ranked_frames), sample_size, rng),
        },
    ]

    deduped_candidates = []
    seen_signatures = set()
    for candidate in candidates:
        frame_ids = tuple(item["frame_id"] for item in candidate["frames"])
        if not frame_ids or frame_ids in seen_signatures:
            continue
        seen_signatures.add(frame_ids)
        deduped_candidates.append(candidate)
    return deduped_candidates
