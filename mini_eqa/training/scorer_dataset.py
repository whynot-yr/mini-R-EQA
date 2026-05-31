from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mini_eqa.data_loading import load_episode_data_bundle
from mini_eqa.training.feature_utils import hashed_text_embedding, mean_pool_embeddings
from mini_eqa.utils.io_utils import load_jsonl


@dataclass
class ScorerTrainingExample:
    question_id: str
    candidate_type: str
    question_embedding: list[float]
    candidate_embedding: list[float]
    reward: float


def build_scorer_training_examples(
    dataset_path: str | Path,
    episode_dir: str | Path,
    question_dim: int = 64,
) -> list[ScorerTrainingExample]:
    dataset_rows = load_jsonl(dataset_path)
    episode_dir = Path(episode_dir)
    bundle = load_episode_data_bundle(
        captions_path=episode_dir / "captions.json",
        embeddings_path=episode_dir
        / "embeddings"
        / "sentence-transformers_all-MiniLM-L6-v2"
        / "caption_embeddings.npy",
    )

    frame_id_to_embedding = {
        record.frame_id: list(bundle.caption_embeddings[record.embedding_index])
        for record in bundle.frame_records
        if record.embedding_index is not None
    }

    examples: list[ScorerTrainingExample] = []
    for row in dataset_rows:
        candidate_embeddings = [
            frame_id_to_embedding[frame_id]
            for frame_id in row["candidate_frames"]
            if frame_id in frame_id_to_embedding
        ]
        if not candidate_embeddings:
            continue

        examples.append(
            ScorerTrainingExample(
                question_id=row["question_id"],
                candidate_type=row["candidate_type"],
                question_embedding=hashed_text_embedding(row["question"], dim=question_dim),
                candidate_embedding=mean_pool_embeddings(candidate_embeddings),
                reward=float(row["reward"]),
            )
        )
    return examples
