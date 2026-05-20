from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=None)
def _load_model(model_name: str, device: str | None) -> SentenceTransformer:
    """
    Load and cache the sentence-transformer model.

    We first try the standard load path. If the environment blocks network
    access but the model is already cached locally, retry in offline mode.
    """
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        return SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,
        )


def retrieve_topk(
    captions: list[dict],
    question: str,
    top_k: int = 3,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant captions with SBERT dense embeddings.

    The return format matches the TF-IDF retriever exactly:
        [{"frame_id": ..., "caption": ..., "score": ...}, ...]
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    if not captions:
        return []

    top_k = min(top_k, len(captions))
    caption_texts = [item["caption"] for item in captions]

    model = _load_model(model_name=model_name, device=device)

    # Encode all captions in one batch so retrieval stays vectorized.
    caption_embeddings = model.encode(
        caption_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # With normalized embeddings, cosine similarity is just a dot product.
    scores = caption_embeddings @ question_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(
            {
                "frame_id": captions[idx]["frame_id"],
                "caption": captions[idx]["caption"],
                "score": float(scores[idx]),
            }
        )

    return results
