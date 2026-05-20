from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from mini_eqa.utils.io_utils import load_json


@lru_cache(maxsize=None)
def _load_model(model_name: str, device: str | None) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        return SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,
        )


def retrieve_topk(
    question: str,
    cache_dir: str | Path,
    top_k: int = 3,
    model_name: str | None = None,
    device: str | None = None,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    embeddings_path = cache_dir / "caption_embeddings.npy"
    metadata_path = cache_dir / "caption_embedding_meta.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    caption_embeddings = np.load(embeddings_path)
    metadata = load_json(metadata_path)
    items = metadata["items"]

    if caption_embeddings.shape[0] != len(items):
        raise ValueError(
            "Embedding row count does not match metadata item count: "
            f"{caption_embeddings.shape[0]} != {len(items)}"
        )

    if not items:
        return []

    top_k = min(top_k, len(items))
    resolved_model_name = model_name or metadata["model_name"]
    model = _load_model(model_name=resolved_model_name, device=device)

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = caption_embeddings @ question_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item = items[idx]
        results.append(
            {
                "frame_id": item["frame_id"],
                "caption": item["caption"],
                "score": float(scores[idx]),
            }
        )

    return results
