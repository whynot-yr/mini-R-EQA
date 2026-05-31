from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.inference.selector_inference import select_top_k_frames
from mini_eqa.training.train_dual_network import run_dual_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first-stage dual-network selector training with scorer auxiliary guidance."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="reports/candidate_reward_dataset.jsonl",
    )
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument("--scorer_checkpoint", type=str, default="reports/scorer.pt")
    parser.add_argument("--selector_checkpoint", type=str, default="reports/selector.pt")
    parser.add_argument("--output", type=str, default="reports/selector_dual.pt")
    parser.add_argument(
        "--metrics_output",
        type=str,
        default="reports/dual_train_metrics.json",
    )
    parser.add_argument("--question_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--auxiliary_weight", type=float, default=0.25)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--smoke_question_id", type=str, default="q1")
    parser.add_argument("--smoke_top_k", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint, metrics = run_dual_training(
        dataset_path=args.dataset_path,
        episode_dir=args.episode_dir,
        scorer_checkpoint_path=args.scorer_checkpoint,
        selector_checkpoint_path=args.selector_checkpoint,
        output_checkpoint_path=args.output,
        metrics_output_path=args.metrics_output,
        question_dim=args.question_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        auxiliary_weight=args.auxiliary_weight,
        max_examples=args.max_examples,
    )
    selected = select_top_k_frames(
        checkpoint_path=args.output,
        episode_dir=args.episode_dir,
        question_id=args.smoke_question_id,
        top_k=args.smoke_top_k,
    )

    print("=" * 80)
    print("Dual-Network Training")
    print("=" * 80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Scorer checkpoint: {args.scorer_checkpoint}")
    print(f"Selector checkpoint: {args.selector_checkpoint}")
    print(f"Output checkpoint: {args.output}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Final train loss: {metrics['final_train_loss']:.6f}")
    print(f"Final scorer auxiliary signal: {metrics['final_scorer_auxiliary_signal']:.6f}")
    print("Selected frames:")
    for item in selected:
        print(f"{item['frame_id']} | score={item['score']:.6f} | {item['caption']}")
    if args.dry_run:
        print("Checkpoint summary:")
        print(
            {
                "framework": checkpoint["framework"],
                "question_dim": checkpoint["question_dim"],
                "frame_dim": checkpoint["frame_dim"],
                "input_dim": checkpoint["input_dim"],
                "dual_training": checkpoint["dual_training"],
            }
        )


if __name__ == "__main__":
    main()
