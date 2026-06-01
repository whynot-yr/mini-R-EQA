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


def _load_frame_index(episode_dir: Path, embedding_subdir: str) -> dict[str, list[float]]:
    embeddings_path = episode_dir / "embeddings" / embedding_subdir / "caption_embeddings.npy"
    captions_path = episode_dir / "captions.json"

    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Caption embeddings not found: {embeddings_path}")

    bundle = load_episode_data_bundle(
        captions_path=captions_path,
        embeddings_path=embeddings_path,
    )
    return {
        record.frame_id: [float(v) for v in bundle.caption_embeddings[record.embedding_index]]
        for record in bundle.frame_records
        if record.embedding_index is not None
    }


class _FrameCache:
    """Lazy per-episode frame-embedding index with in-process caching."""

    def __init__(self, embedding_subdir: str) -> None:
        self._subdir = embedding_subdir
        self._cache: dict[str, dict[str, list[float]]] = {}

    def get(self, episode_dir: Path) -> dict[str, list[float]]:
        key = str(episode_dir.resolve())
        if key not in self._cache:
            self._cache[key] = _load_frame_index(episode_dir, self._subdir)
        return self._cache[key]


def build_scorer_training_examples(
    dataset_path: str | Path,
    episode_dir: str | Path | None = None,
    question_dim: int = 64,
    sbert_model_name: str | None = None,
    embedding_subdir: str = _DEFAULT_EMBEDDING_SUBDIR,
    prepared_root: str | Path | None = None,
) -> list[ScorerTrainingExample]:
    """Build scorer training examples, supporting single or multi-episode datasets.

    Provide either:
        episode_dir   – single episode (legacy / single-episode mode)
        prepared_root – root containing episode subdirectories; each JSONL row
                        must have an "episode_id" field that resolves to
                        prepared_root / episode_id.
    """
    if prepared_root is None and episode_dir is None:
        raise ValueError(
            "Provide either episode_dir (single episode) or "
            "prepared_root (multi-episode training)."
        )

    dataset_rows = load_jsonl(dataset_path)

    # Batch-encode all unique question texts upfront (SBERT mode only).
    question_emb_map: dict[str, list[float]] = {}
    if sbert_model_name is not None:
        model = _load_sbert(sbert_model_name)
        unique_questions = list(dict.fromkeys(row["question"] for row in dataset_rows))
        encoded = model.encode(unique_questions, convert_to_numpy=True, normalize_embeddings=True)
        question_emb_map = {q: encoded[i].tolist() for i, q in enumerate(unique_questions)}

    cache = _FrameCache(embedding_subdir)
    # Pre-load single episode once when using episode_dir mode.
    if prepared_root is None:
        ep_dir = Path(episode_dir)  # type: ignore[arg-type]
        cache.get(ep_dir)  # validate + warm cache

    examples: list[ScorerTrainingExample] = []
    for row in dataset_rows:
        if prepared_root is not None:
            ep_id = row.get("episode_id")
            if not ep_id:
                raise ValueError(
                    f"Row with question_id={row.get('question_id')!r} is missing episode_id. "
                    "episode_id is required for multi-episode training with --prepared_root."
                )
            ep_dir = Path(prepared_root) / ep_id
        else:
            ep_dir = Path(episode_dir)  # type: ignore[arg-type]

        frame_index = cache.get(ep_dir)

        frame_ids = row.get("frame_ids") or row.get("candidate_frames", [])
        candidate_embeddings = [
            frame_index[frame_id]
            for frame_id in frame_ids
            if frame_id in frame_index
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
