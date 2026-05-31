from __future__ import annotations

from pathlib import Path

from mini_eqa.data_loading import load_episode_data_bundle
from mini_eqa.training.feature_utils import dot_product, hashed_text_embedding, mean_pool_embeddings
from mini_eqa.utils.io_utils import load_json, load_jsonl


def load_scorer_checkpoint(checkpoint_path: str | Path) -> dict:
    return load_json(checkpoint_path)


def predict_reward_from_features(
    checkpoint: dict,
    question_embedding: list[float],
    candidate_embedding: list[float],
) -> float:
    features = question_embedding + candidate_embedding
    return dot_product(checkpoint["weights"], features) + checkpoint["bias"]


def predict_candidate_reward(
    checkpoint_path: str | Path,
    episode_dir: str | Path,
    question_id: str,
    candidate_frames: list[str],
) -> float:
    checkpoint = load_scorer_checkpoint(checkpoint_path)
    episode_dir = Path(episode_dir)
    questions = load_json(episode_dir / "questions.json")
    question_item = next(item for item in questions if item["question_id"] == question_id)
    question_embedding = hashed_text_embedding(
        question_item["question"],
        dim=checkpoint["question_dim"],
    )

    bundle = load_episode_data_bundle(
        captions_path=episode_dir / "captions.json",
        embeddings_path=episode_dir
        / "embeddings"
        / "sentence-transformers_all-MiniLM-L6-v2"
        / "caption_embeddings.npy",
    )
    frame_index = {
        record.frame_id: list(bundle.caption_embeddings[record.embedding_index])
        for record in bundle.frame_records
        if record.embedding_index is not None
    }
    candidate_embedding = mean_pool_embeddings(
        [frame_index[frame_id] for frame_id in candidate_frames if frame_id in frame_index]
    )
    return predict_reward_from_features(
        checkpoint=checkpoint,
        question_embedding=question_embedding,
        candidate_embedding=candidate_embedding,
    )


def build_frame_auxiliary_targets(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    episode_dir: str | Path,
) -> dict[tuple[str, str], float]:
    checkpoint = load_scorer_checkpoint(checkpoint_path)
    dataset_rows = load_jsonl(dataset_path)
    episode_dir = Path(episode_dir)
    questions = load_json(episode_dir / "questions.json")
    question_map = {item["question_id"]: item["question"] for item in questions}

    bundle = load_episode_data_bundle(
        captions_path=episode_dir / "captions.json",
        embeddings_path=episode_dir
        / "embeddings"
        / "sentence-transformers_all-MiniLM-L6-v2"
        / "caption_embeddings.npy",
    )
    frame_index = {
        record.frame_id: list(bundle.caption_embeddings[record.embedding_index])
        for record in bundle.frame_records
        if record.embedding_index is not None
    }

    grouped_scores: dict[tuple[str, str], list[float]] = {}
    for row in dataset_rows:
        question_id = row["question_id"]
        question_embedding = hashed_text_embedding(
            question_map[question_id],
            dim=checkpoint["question_dim"],
        )
        candidate_embedding = mean_pool_embeddings(
            [frame_index[frame_id] for frame_id in row["candidate_frames"] if frame_id in frame_index]
        )
        predicted_reward = predict_reward_from_features(
            checkpoint=checkpoint,
            question_embedding=question_embedding,
            candidate_embedding=candidate_embedding,
        )
        for frame_id in row["candidate_frames"]:
            grouped_scores.setdefault((question_id, frame_id), []).append(predicted_reward)

    return {
        key: sum(values) / len(values)
        for key, values in grouped_scores.items()
    }
