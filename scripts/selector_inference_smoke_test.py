from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.inference.selector_inference import (
    is_torch_checkpoint,
    select_top_k_frames,
    select_top_k_frames_torch,
)


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required for --config support. pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    config_defaults: dict = {}
    if pre_args.config:
        config_defaults = _load_yaml(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Smoke test selector inference and top-k frame selection."
    )
    parser.set_defaults(**config_defaults)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episode_dir", type=str, default=None)
    parser.add_argument("--question_id", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="SBERT model name (torch checkpoint only). If omitted, read from cache metadata.",
    )
    parser.add_argument(
        "--embedding_subdir",
        type=str,
        default="sentence-transformers_all-MiniLM-L6-v2",
        help="Subdirectory under episode_dir/embeddings/ for cached frame embeddings.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for field in ("checkpoint", "episode_dir", "question_id"):
        if getattr(args, field) is None:
            print(
                f"ERROR: --{field} is required (or set it in --config).",
                file=sys.stderr,
            )
            sys.exit(1)

    if is_torch_checkpoint(args.checkpoint):
        selected = select_top_k_frames_torch(
            checkpoint_path=args.checkpoint,
            episode_dir=args.episode_dir,
            question_id=args.question_id,
            top_k=args.top_k,
            embedding_model=args.embedding_model,
            embedding_subdir=args.embedding_subdir,
            device=args.device,
        )
        backend = "torch"
    else:
        selected = select_top_k_frames(
            checkpoint_path=args.checkpoint,
            episode_dir=args.episode_dir,
            question_id=args.question_id,
            top_k=args.top_k,
        )
        backend = "fallback"

    print("=" * 80)
    print(f"Selector Inference Smoke Test (backend={backend})")
    print("=" * 80)
    print(f"Question id: {args.question_id}")
    print(f"Top-k: {args.top_k}")
    for item in selected:
        print(
            f"{item['frame_id']} | score={item['score']:.6f} | logit={item['logit']:.6f} | {item['caption']}"
        )


if __name__ == "__main__":
    main()
