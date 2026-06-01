from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from mini_eqa.data_loading import load_episode_data_bundle
from mini_eqa.training.feature_utils import hashed_text_embedding
from mini_eqa.utils.io_utils import load_json, load_jsonl

_DEFAULT_EMBEDDING_SUBDIR = "sentence-transformers_all-MiniLM-L6-v2"


@dataclass
class SelectorTrainingExample:
    question_id: str
    frame_id: str
    question_embedding: list[float]
    frame_embedding: list[float]
    label: float


@lru_cache(maxsize=None)
def _load_sbert(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_frame_index(
    episode_dir: Path,
    embedding_subdir: str,
) -> dict[str, list[float]]:
    bundle = load_episode_data_bundle(
        captions_path=episode_dir / "captions.json",
        embeddings_path=episode_dir / "embeddings" / embedding_subdir / "caption_embeddings.npy",
    )
    return {
        record.frame_id: [float(v) for v in bundle.caption_embeddings[record.embedding_index]]
        for record in bundle.frame_records
        if record.embedding_index is not None
    }


def _make_frame_features(
    frame_id: str,
    frame_index: dict[str, list[float]],
    q_emb: list[float],
    sbert_mode: bool,
) -> list[float] | None:
    if frame_id not in frame_index:
        return None
    f_emb = frame_index[frame_id]
    if sbert_mode:
        cosine = _cosine_sim(q_emb, f_emb)
        return f_emb + [cosine]
    return f_emb


def build_selector_training_examples(
    dataset_path: str | Path,
    episode_dir: str | Path,
    question_dim: int = 64,
    sbert_model_name: str | None = None,
    embedding_subdir: str = _DEFAULT_EMBEDDING_SUBDIR,
    min_reward_gap: float = 0.25,
) -> tuple[list[SelectorTrainingExample], dict]:
    """Build selector training examples from a candidate reward dataset.

    Args:
        min_reward_gap: Minimum gap between high and low reward to generate
            pseudo labels for a question. Questions with reward_gap < threshold
            are skipped entirely — they carry no preference signal.

    Returns:
        (examples, summary) where summary contains aggregate stats and per-question detail.
    """
    dataset_rows = load_jsonl(dataset_path)
    episode_dir = Path(episode_dir)
    questions = load_json(episode_dir / "questions.json")
    question_map = {item["question_id"]: item["question"] for item in questions}
    frame_index = _build_frame_index(episode_dir, embedding_subdir)

    question_emb_map: dict[str, list[float]] = {}
    if sbert_model_name is not None:
        model = _load_sbert(sbert_model_name)
        unique_texts = list(dict.fromkeys(question_map.values()))
        encoded = model.encode(unique_texts, convert_to_numpy=True, normalize_embeddings=True)
        text_to_emb = {t: encoded[i].tolist() for i, t in enumerate(unique_texts)}
        question_emb_map = {qid: text_to_emb[text] for qid, text in question_map.items()}

    sbert_mode = sbert_model_name is not None

    grouped_rows: dict[str, list[dict]] = {}
    for row in dataset_rows:
        grouped_rows.setdefault(row["question_id"], []).append(row)

    examples: list[SelectorTrainingExample] = []
    per_question: dict[str, dict] = {}
    skipped_reason_counts: dict[str, int] = {}
    total_positive = 0
    total_negative = 0
    num_questions_skipped = 0

    for question_id, rows in grouped_rows.items():
        rewards = [float(row["reward"]) for row in rows]
        high_reward = max(rewards)
        low_reward = min(rewards)
        reward_gap = high_reward - low_reward

        # Skip questions with insufficient reward spread — no preference signal.
        if reward_gap < min_reward_gap:
            skip_reason = "insufficient_reward_gap"
            skipped_reason_counts[skip_reason] = skipped_reason_counts.get(skip_reason, 0) + 1
            num_questions_skipped += 1
            per_question[question_id] = {
                "high_reward": high_reward,
                "low_reward": low_reward,
                "reward_gap": reward_gap,
                "skipped": True,
                "skip_reason": skip_reason,
                "positive_frames": [],
                "negative_frames": [],
            }
            continue

        # Positive: frames from high-reward candidates.
        positive_frames = {
            frame_id
            for row in rows
            if float(row["reward"]) == high_reward
            for frame_id in (row.get("frame_ids") or row.get("candidate_frames", []))
        }

        # Negative: frames from low-reward candidates OR hard negatives,
        # excluding any frame that appears in positive_frames.
        negative_frames = {
            frame_id
            for row in rows
            if (float(row["reward"]) == low_reward or row.get("is_hard_negative", False))
            for frame_id in (row.get("frame_ids") or row.get("candidate_frames", []))
            if frame_id not in positive_frames
        }

        if sbert_mode:
            q_emb = question_emb_map[question_id]
        else:
            q_emb = hashed_text_embedding(question_map[question_id], dim=question_dim)

        pos_added = 0
        for frame_id in sorted(positive_frames):
            features = _make_frame_features(frame_id, frame_index, q_emb, sbert_mode)
            if features is None:
                continue
            examples.append(
                SelectorTrainingExample(
                    question_id=question_id,
                    frame_id=frame_id,
                    question_embedding=q_emb,
                    frame_embedding=features,
                    label=1.0,
                )
            )
            pos_added += 1

        neg_added = 0
        for frame_id in sorted(negative_frames):
            features = _make_frame_features(frame_id, frame_index, q_emb, sbert_mode)
            if features is None:
                continue
            examples.append(
                SelectorTrainingExample(
                    question_id=question_id,
                    frame_id=frame_id,
                    question_embedding=q_emb,
                    frame_embedding=features,
                    label=0.0,
                )
            )
            neg_added += 1

        total_positive += pos_added
        total_negative += neg_added

        per_question[question_id] = {
            "high_reward": high_reward,
            "low_reward": low_reward,
            "reward_gap": reward_gap,
            "skipped": False,
            "skip_reason": None,
            "positive_frames": sorted(positive_frames),
            "negative_frames": sorted(negative_frames),
        }

    # ── Fail loudly if nothing to train on ───────────────────────────────────
    if not examples:
        raise RuntimeError(
            f"No selector training examples could be generated. "
            f"All {len(grouped_rows)} questions were skipped "
            f"(reward_gap < min_reward_gap={min_reward_gap}). "
            "Lower --min_reward_gap, use a better reward signal "
            "(e.g. --reward_mode deepseek_judge), or verify the dataset has reward variance."
        )

    # ── Class-balance warnings ────────────────────────────────────────────────
    if total_positive == 0:
        warnings.warn(
            "Selector training data has zero positive labels. "
            "Check that high-reward candidates exist.",
            stacklevel=2,
        )
    if total_negative == 0:
        warnings.warn(
            "Selector training data has zero negative labels. "
            "Check that low-reward or hard-negative candidates exist.",
            stacklevel=2,
        )
    if total_positive > 0 and total_negative > 0:
        ratio = total_positive / total_negative
        if ratio > 10 or ratio < 0.1:
            print(
                f"WARNING: Severely imbalanced selector labels — "
                f"{total_positive} positive vs {total_negative} negative. "
                "Training may be biased.",
                file=sys.stderr,
            )

    summary: dict = {
        "num_questions_total": len(grouped_rows),
        "num_questions_used": len(grouped_rows) - num_questions_skipped,
        "num_questions_skipped": num_questions_skipped,
        "skipped_reason_counts": skipped_reason_counts,
        "num_positive_frames": total_positive,
        "num_negative_frames": total_negative,
        "min_reward_gap": min_reward_gap,
        "questions": per_question,
    }

    return examples, summary
