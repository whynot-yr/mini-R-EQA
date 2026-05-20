from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from mini_eqa.utils.io_utils import load_json, save_json


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=None)
def load_model(model_name: str, device: str | None) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        return SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,
        )


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def resolve_output_dir(episode_dir: Path, model_name: str, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    return episode_dir / "embeddings" / sanitize_model_name(model_name)


def build_metadata(
    captions: list[dict],
    model_name: str,
    captions_path: Path,
    caption_embeddings: np.ndarray,
) -> dict:
    return {
        "model_name": model_name,
        "normalize_embeddings": True,
        "num_captions": len(captions),
        "embedding_dim": int(caption_embeddings.shape[1]),
        "captions_path": str(captions_path),
        "items": [
            {
                "index": idx,
                "frame_id": item["frame_id"],
                "caption": item["caption"],
            }
            for idx, item in enumerate(captions)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and cache SBERT caption embeddings for one episode."
    )
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    episode_dir = Path(args.episode_dir)
    captions_path = episode_dir / "captions.json"
    captions = load_json(captions_path)

    if not captions:
        raise ValueError(f"No captions found in {captions_path}")

    caption_texts = [item["caption"] for item in captions]
    output_dir = resolve_output_dir(
        episode_dir=episode_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
    )
    embeddings_path = output_dir / "caption_embeddings.npy"
    metadata_path = output_dir / "caption_embedding_meta.json"

    if not args.overwrite and (embeddings_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Embedding cache already exists in {output_dir}. "
            "Use --overwrite to replace it."
        )

    model = load_model(args.model_name, args.device)
    caption_embeddings = model.encode(
        caption_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, caption_embeddings)

    metadata = build_metadata(
        captions=captions,
        model_name=args.model_name,
        captions_path=captions_path,
        caption_embeddings=caption_embeddings,
    )
    save_json(metadata, metadata_path)

    print("=" * 80)
    print("Caption Embedding Cache Built")
    print("=" * 80)
    print(f"Episode dir: {episode_dir}")
    print(f"Model: {args.model_name}")
    print(f"Output dir: {output_dir}")
    print(f"Num captions: {metadata['num_captions']}")
    print(f"Embedding dim: {metadata['embedding_dim']}")
    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
