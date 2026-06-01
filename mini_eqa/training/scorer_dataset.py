from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from mini_eqa.data_loading import load_episode_data_bundle
from mini_eqa.training.feature_utils import hashed_text_embedding, mean_pool_embeddings
from mini_eqa.utils.io_utils import load_jsonl

_DEFAULT_EMBEDDING_SUBDIR = "sentence-transformers_all-MiniLM-L6-v2"


@dataclass
class ScorerTrainingExample:
    question_id: str
    candidate_type: str
    question_embedding: list[float]
    candidate_embedding: list[float]
    reward: float


@lru_cache(maxsize=None)
def _load_sbert(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def build_scorer_training_examples(
    dataset_path: str | Path,
    episode_dir: str | Path,
    question_dim: int = 64,
    sbert_model_name: str | None = None,
    embedding_subdir: str = _DEFAULT_EMBEDDING_SUBDIR,
) -> list[ScorerTrainingExample]:
    dataset_rows = load_jsonl(dataset_path)
    episode_dir = Path(episode_dir)
    bundle = load_episode_data_bundle(
        captions_path=episode_dir / "captions.json",
        embeddings_path=episode_dir / "embeddings" / embedding_subdir / "caption_embeddings.npy",
    )

    frame_id_to_embedding = {
        record.frame_id: [float(v) for v in bundle.caption_embeddings[record.embedding_index]]
        for record in bundle.frame_records
        if record.embedding_index is not None
    }

    question_emb_map: dict[str, list[float]] = {}
    if sbert_model_name is not None:
        model = _load_sbert(sbert_model_name)
        unique_questions = list(dict.fromkeys(row["question"] for row in dataset_rows))
        encoded = model.encode(unique_questions, convert_to_numpy=True, normalize_embeddings=True)
        question_emb_map = {q: encoded[i].tolist() for i, q in enumerate(unique_questions)}

    examples: list[ScorerTrainingExample] = []
    for row in dataset_rows:
        frame_ids = row.get("frame_ids") or row.get("candidate_frames", [])
        candidate_embeddings = [
            frame_id_to_embedding[frame_id]
            for frame_id in frame_ids
            if frame_id in frame_id_to_embedding
        ]
        if not candidate_embeddings:
            continue

        if sbert_model_name is not None:
            q_emb = question_emb_map[row["question"]]
        else:
            q_emb = hashed_text_embedding(row["question"], dim=question_dim)

        examples.append(
            ScorerTrainingExample(
                question_id=row["question_id"],
                candidate_type=row["candidate_type"],
                question_embedding=q_emb,
                candidate_embedding=mean_pool_embeddings(candidate_embeddings),
                reward=float(row["reward"]),
            )
        )
    return examples
