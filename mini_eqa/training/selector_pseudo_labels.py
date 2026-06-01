from __future__ import annotations

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


def build_selector_training_examples(
    dataset_path: str | Path,
    episode_dir: str | Path,
    question_dim: int = 64,
    sbert_model_name: str | None = None,
    embedding_subdir: str = _DEFAULT_EMBEDDING_SUBDIR,
) -> tuple[list[SelectorTrainingExample], dict[str, dict[str, list[str] | float]]]:
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

    grouped_rows: dict[str, list[dict]] = {}
    for row in dataset_rows:
        grouped_rows.setdefault(row["question_id"], []).append(row)

    examples: list[SelectorTrainingExample] = []
    summaries: dict[str, dict[str, list[str] | float]] = {}
    for question_id, rows in grouped_rows.items():
        rewards = [float(row["reward"]) for row in rows]
        high_reward = max(rewards)
        low_reward = min(rewards)

        positive_frames = {
            frame_id
            for row in rows
            if float(row["reward"]) == high_reward
            for frame_id in row["candidate_frames"]
        }
        negative_frames = {
            frame_id
            for row in rows
            if float(row["reward"]) == low_reward
            for frame_id in row["candidate_frames"]
            if frame_id not in positive_frames
        }

        if sbert_model_name is not None:
            q_emb = question_emb_map[question_id]
        else:
            q_emb = hashed_text_embedding(question_map[question_id], dim=question_dim)

        for frame_id in sorted(positive_frames):
            if frame_id not in frame_index:
                continue
            f_emb = frame_index[frame_id]
            if sbert_model_name is not None:
                cosine = _cosine_sim(q_emb, f_emb)
                frame_features = f_emb + [cosine]
            else:
                frame_features = f_emb
            examples.append(
                SelectorTrainingExample(
                    question_id=question_id,
                    frame_id=frame_id,
                    question_embedding=q_emb,
                    frame_embedding=frame_features,
                    label=1.0,
                )
            )

        for frame_id in sorted(negative_frames):
            if frame_id not in frame_index:
                continue
            f_emb = frame_index[frame_id]
            if sbert_model_name is not None:
                cosine = _cosine_sim(q_emb, f_emb)
                frame_features = f_emb + [cosine]
            else:
                frame_features = f_emb
            examples.append(
                SelectorTrainingExample(
                    question_id=question_id,
                    frame_id=frame_id,
                    question_embedding=q_emb,
                    frame_embedding=frame_features,
                    label=0.0,
                )
            )

        summaries[question_id] = {
            "high_reward": high_reward,
            "low_reward": low_reward,
            "positive_frames": sorted(positive_frames),
            "negative_frames": sorted(negative_frames),
        }

    return examples, summaries
