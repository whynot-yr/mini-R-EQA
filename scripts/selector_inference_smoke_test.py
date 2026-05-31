from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.inference.selector_inference import select_top_k_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test selector inference and top-k frame selection."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episode_dir", type=str, required=True)
    parser.add_argument("--question_id", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = select_top_k_frames(
        checkpoint_path=args.checkpoint,
        episode_dir=args.episode_dir,
        question_id=args.question_id,
        top_k=args.top_k,
    )
    print("=" * 80)
    print("Selector Inference Smoke Test")
    print("=" * 80)
    print(f"Question id: {args.question_id}")
    print(f"Top-k: {args.top_k}")
    for item in selected:
        print(
            f"{item['frame_id']} | score={item['score']:.6f} | logit={item['logit']:.6f} | {item['caption']}"
        )


if __name__ == "__main__":
    main()
