from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.data_loading import load_episode_data_bundle
from mini_eqa.scorer import MLPScorer, MLPSupervisor
from mini_eqa.selector import MLPSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for selector-scorer skeleton imports and loaders."
    )
    parser.add_argument("--captions", type=str, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_episode_data_bundle(
        captions_path=args.captions,
        embeddings_path=args.embeddings,
    )
    if hasattr(bundle.caption_embeddings, "shape"):
        embedding_shape = tuple(int(dim) for dim in bundle.caption_embeddings.shape)
        embedding_dim = embedding_shape[1]
    else:
        embedding_shape = (
            len(bundle.caption_embeddings),
            len(bundle.caption_embeddings[0]) if bundle.caption_embeddings else 0,
        )
        embedding_dim = embedding_shape[1]

    selector = MLPSelector(
        question_dim=embedding_dim,
        frame_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    scorer = MLPScorer(
        question_dim=embedding_dim,
        candidate_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    supervisor = MLPSupervisor(scorer)

    print("=" * 80)
    print("Selector-Scorer Smoke Test")
    print("=" * 80)
    print(f"Num frames: {len(bundle.frame_records)}")
    print(f"Embedding shape: {embedding_shape}")
    print(f"Selector class: {selector.__class__.__name__}")
    print(f"Selector framework: {selector.framework}")
    print(f"Scorer class: {scorer.__class__.__name__}")
    print(f"Scorer framework: {scorer.framework}")
    print(f"Supervisor class: {supervisor.__class__.__name__}")
    print(f"Supervisor framework: {supervisor.framework}")


if __name__ == "__main__":
    main()
