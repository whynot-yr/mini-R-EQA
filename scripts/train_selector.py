from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.training.selector_pseudo_labels import build_selector_training_examples
from mini_eqa.training.train_selector import save_selector_outputs, train_fallback_selector
from mini_eqa.utils.io_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a selector model from frame-level pseudo labels."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="reports/candidate_reward_dataset.jsonl",
    )
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument("--output", type=str, default="reports/selector.pt")
    parser.add_argument(
        "--metrics_output",
        type=str,
        default="reports/selector_train_metrics.json",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default="reports/selector_pseudo_label_summary.json",
    )
    parser.add_argument("--question_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples, summaries = build_selector_training_examples(
        dataset_path=args.dataset_path,
        episode_dir=args.episode_dir,
        question_dim=args.question_dim,
    )
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    checkpoint, metrics = train_fallback_selector(
        examples=examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    save_selector_outputs(
        checkpoint=checkpoint,
        metrics=metrics,
        checkpoint_path=args.output,
        metrics_path=args.metrics_output,
    )
    save_json(summaries, args.summary_output)

    print("=" * 80)
    print("Selector Training")
    print("=" * 80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Episode dir: {args.episode_dir}")
    print(f"Examples used: {metrics['num_examples']}")
    print(f"Epochs: {metrics['epochs']}")
    print(f"Final train loss: {metrics['final_train_loss']:.6f}")
    print(f"Checkpoint: {args.output}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Pseudo-label summary: {args.summary_output}")
    if args.dry_run:
        print("Checkpoint summary:")
        print(
            {
                "framework": checkpoint["framework"],
                "question_dim": checkpoint["question_dim"],
                "frame_dim": checkpoint["frame_dim"],
                "input_dim": checkpoint["input_dim"],
            }
        )


if __name__ == "__main__":
    main()
