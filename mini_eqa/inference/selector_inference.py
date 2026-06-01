from __future__ import annotations

import math
import json
from pathlib import Path

from mini_eqa.data_loading import load_episode_data_bundle
from mini_eqa.training.feature_utils import dot_product, hashed_text_embedding
from mini_eqa.utils.io_utils import load_json


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _load_selector_checkpoint(checkpoint_path: str | Path) -> dict:
    try:
        return load_json(checkpoint_path)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Selector inference expects a fallback-format JSON checkpoint. "
            "Torch `.pt` selector checkpoints are not supported by this smoke test."
        ) from exc


def select_top_k_frames(
    checkpoint_path: str | Path,
    episode_dir: str | Path,
    question_id: str,
    top_k: int = 3,
) -> list[dict]:
    checkpoint = _load_selector_checkpoint(checkpoint_path)
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

    ranked = []
    for record in bundle.frame_records:
        frame_embedding = list(bundle.caption_embeddings[record.embedding_index])
        features = question_embedding + frame_embedding
        logit = dot_product(checkpoint["weights"], features) + checkpoint["bias"]
        ranked.append(
            {
                "frame_id": record.frame_id,
                "caption": record.caption,
                "logit": logit,
                "score": _sigmoid(logit),
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["frame_id"]))
    return ranked[:top_k]
