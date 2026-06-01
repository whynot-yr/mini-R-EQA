from __future__ import annotations

import ast
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_eqa.schema import FrameRecord
from mini_eqa.utils.io_utils import load_json

_DEFAULT_EMBEDDING_SUBDIR = "sentence-transformers_all-MiniLM-L6-v2"


def resolve_embedding_paths(
    episode_dir: str | Path,
    embedding_subdir: str = _DEFAULT_EMBEDDING_SUBDIR,
) -> tuple[Path, Path]:
    """Return (caption_embeddings.npy path, caption_embedding_meta.json path)."""
    base = Path(episode_dir) / "embeddings" / embedding_subdir
    return base / "caption_embeddings.npy", base / "caption_embedding_meta.json"


def resolve_embedding_cache_dir(
    episode_dir: str | Path,
    embedding_subdir: str = _DEFAULT_EMBEDDING_SUBDIR,
) -> Path:
    """Return the embeddings subdirectory used by cached_sbert and candidate_generation."""
    return Path(episode_dir) / "embeddings" / embedding_subdir

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in dependency-light environments
    np = None


@dataclass
class EpisodeDataBundle:
    captions_path: Path
    embeddings_path: Path
    frame_records: list[FrameRecord]
    caption_embeddings: Any


def _load_npy_without_numpy(embeddings_path: Path) -> list[list[float]]:
    with embeddings_path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"Unsupported NPY file: {embeddings_path}")

        major, minor = f.read(2)
        if (major, minor) != (1, 0):
            raise ValueError(
                f"Unsupported NPY version {(major, minor)} in {embeddings_path}"
            )

        header_len = struct.unpack("<H", f.read(2))[0]
        header = f.read(header_len).decode("latin1")
        header_dict = ast.literal_eval(header.strip())

        dtype = header_dict["descr"]
        shape = header_dict["shape"]
        fortran_order = header_dict["fortran_order"]

        if fortran_order:
            raise ValueError("Fortran-ordered arrays are not supported without numpy.")
        if len(shape) != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {shape}")
        if dtype not in {"<f4", "<f8"}:
            raise ValueError(f"Unsupported dtype {dtype} in {embeddings_path}")

        rows, cols = shape
        format_char = "f" if dtype == "<f4" else "d"
        total = rows * cols
        values = struct.unpack(f"<{total}{format_char}", f.read())
        return [
            [float(values[row * cols + col]) for col in range(cols)]
            for row in range(rows)
        ]


def load_frame_records(captions_path: str | Path) -> list[FrameRecord]:
    captions_data = load_json(captions_path)
    if not isinstance(captions_data, list):
        raise ValueError(f"Expected captions list in {captions_path}")

    frame_records: list[FrameRecord] = []
    for index, item in enumerate(captions_data):
        frame_records.append(
            FrameRecord(
                frame_id=str(item["frame_id"]),
                caption=str(item["caption"]),
                image_path=item.get("image_path"),
                embedding_index=index,
            )
        )
    return frame_records


def load_caption_embeddings(embeddings_path: str | Path) -> Any:
    embeddings_path = Path(embeddings_path)
    if np is not None:
        embeddings = np.load(embeddings_path)
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D caption embeddings array, got shape {embeddings.shape}"
            )
        return embeddings

    return _load_npy_without_numpy(embeddings_path)


def load_episode_data_bundle(
    captions_path: str | Path,
    embeddings_path: str | Path,
) -> EpisodeDataBundle:
    captions_path = Path(captions_path)
    embeddings_path = Path(embeddings_path)
    frame_records = load_frame_records(captions_path)
    caption_embeddings = load_caption_embeddings(embeddings_path)
    row_count = (
        int(caption_embeddings.shape[0])
        if hasattr(caption_embeddings, "shape")
        else len(caption_embeddings)
    )
    if len(frame_records) != row_count:
        raise ValueError(
            "Number of frame records does not match first embedding dimension: "
            f"{len(frame_records)} vs {row_count}"
        )

    return EpisodeDataBundle(
        captions_path=captions_path,
        embeddings_path=embeddings_path,
        frame_records=frame_records,
        caption_embeddings=caption_embeddings,
    )
