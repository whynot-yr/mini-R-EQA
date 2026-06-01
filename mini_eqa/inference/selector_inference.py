from __future__ import annotations

import json
import math
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


def is_torch_checkpoint(checkpoint_path: str | Path) -> bool:
    """Return True if the file looks like a torch-saved checkpoint (not plain JSON)."""
    try:
        with open(checkpoint_path, "rb") as f:
            magic = f.read(2)
        # torch pickled files start with \x80 (pickle protocol) or PK (zip/torch 2.x format)
        return magic[:1] == b"\x80" or magic == b"PK"
    except OSError:
        return False


def select_top_k_frames_torch(
    checkpoint_path: str | Path,
    episode_dir: str | Path,
    question_id: str,
    top_k: int = 3,
    embedding_model: str | None = None,
    embedding_subdir: str = "sentence-transformers_all-MiniLM-L6-v2",
    device: str = "cpu",
) -> list[dict]:
    """Inference using a torch-native MLPSelector checkpoint."""
    try:
        import torch
    except ImportError:
        raise RuntimeError("select_top_k_frames_torch requires torch. Install PyTorch.")

    from sentence_transformers import SentenceTransformer

    from mini_eqa.selector.mlp_selector import MLPSelector

    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

    if ckpt.get("framework") != "torch":
        raise ValueError(
            f"Expected torch checkpoint at {checkpoint_path}, "
            f"got framework='{ckpt.get('framework')}'. "
            "Use select_top_k_frames() for fallback JSON checkpoints."
        )

    q_dim = ckpt["question_dim"]
    frame_dim = ckpt["frame_dim"]
    hidden_dim = ckpt["hidden_dim"]

    model = MLPSelector(q_dim, frame_dim, hidden_dim)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    episode_dir = Path(episode_dir)
    questions = load_json(episode_dir / "questions.json")
    question_item = next(item for item in questions if item["question_id"] == question_id)

    # Resolve SBERT model: try cache metadata, then fall back to arg or default
    meta_path = episode_dir / "embeddings" / embedding_subdir / "caption_embedding_meta.json"
    resolved_model = embedding_model
    if resolved_model is None and meta_path.exists():
        resolved_model = load_json(meta_path).get("model_name", "all-MiniLM-L6-v2")
    if resolved_model is None:
        resolved_model = "all-MiniLM-L6-v2"

    sbert = SentenceTransformer(resolved_model, device=device)
    q_emb_np = sbert.encode(
        [question_item["question"]],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    bundle = load_episode_data_bundle(
        captions_path=episode_dir / "captions.json",
        embeddings_path=episode_dir / "embeddings" / embedding_subdir / "caption_embeddings.npy",
    )

    # Detect if cosine_sim was appended during training (frame_dim = q_dim + 1)
    has_cosine_feature = frame_dim == q_dim + 1

    import numpy as np

    q_tensor = torch.tensor(q_emb_np, dtype=torch.float32, device=device)
    ranked = []
    with torch.no_grad():
        for record in bundle.frame_records:
            f_arr = np.array(
                [float(v) for v in bundle.caption_embeddings[record.embedding_index]]
            )
            if has_cosine_feature:
                cosine = float(np.dot(q_emb_np, f_arr))  # both normalized → cosine = dot
                frame_features = f_arr.tolist() + [cosine]
            else:
                frame_features = f_arr.tolist()

            f_tensor = torch.tensor(frame_features, dtype=torch.float32, device=device)
            logit = model(q_tensor, f_tensor)
            if logit.dim() > 0:
                logit = logit.squeeze(0)
            logit_val = float(logit.item())
            ranked.append(
                {
                    "frame_id": record.frame_id,
                    "caption": record.caption,
                    "logit": logit_val,
                    "score": _sigmoid(logit_val),
                }
            )

    ranked.sort(key=lambda item: (-item["score"], item["frame_id"]))
    return ranked[:top_k]
