from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mini_eqa.data_loading import load_episode_data_bundle
from mini_eqa.training.feature_utils import hashed_text_embedding
from mini_eqa.utils.io_utils import load_json, load_jsonl


@dataclass
class SelectorTrainingExample:
    question_id: str
    frame_id: str
    question_embedding: list[float]
    frame_embedding: list[float]
    label: float


def _build_frame_index(episode_dir: Path) -> dict[str, list[float]]:
    bundle = load_episode_data_bundle(
        captions_path=episode_dir / "captions.json",
        embeddings_path=episode_dir
        / "embeddings"
        / "sentence-transformers_all-MiniLM-L6-v2"
        / "caption_embeddings.npy",
    )
    return {
        record.frame_id: list(bundle.caption_embeddings[record.embedding_index])
        for record in bundle.frame_records
        if record.embedding_index is not None
    }


def build_selector_training_examples(
    dataset_path: str | Path,
    episode_dir: str | Path,
    question_dim: int = 64,
) -> tuple[list[SelectorTrainingExample], dict[str, dict[str, list[str] | float]]]:
    dataset_rows = load_jsonl(dataset_path)
    episode_dir = Path(episode_dir)
    questions = load_json(episode_dir / "questions.json")
    question_map = {item["question_id"]: item["question"] for item in questions}
    frame_index = _build_frame_index(episode_dir)

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

        question_embedding = hashed_text_embedding(question_map[question_id], dim=question_dim)
        for frame_id in sorted(positive_frames):
            if frame_id not in frame_index:
                continue
            examples.append(
                SelectorTrainingExample(
                    question_id=question_id,
                    frame_id=frame_id,
                    question_embedding=question_embedding,
                    frame_embedding=frame_index[frame_id],
                    label=1.0,
                )
            )

        for frame_id in sorted(negative_frames):
            if frame_id not in frame_index:
                continue
            examples.append(
                SelectorTrainingExample(
                    question_id=question_id,
                    frame_id=frame_id,
                    question_embedding=question_embedding,
                    frame_embedding=frame_index[frame_id],
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
