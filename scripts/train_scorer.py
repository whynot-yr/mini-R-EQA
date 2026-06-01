from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.training.scorer_dataset import build_scorer_training_examples
from mini_eqa.training.train_scorer import (
    save_scorer_outputs,
    train_fallback_scorer,
    train_torch_scorer,
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
        description="Train a scorer model from candidate reward dataset supervision."
    )
    parser.set_defaults(**config_defaults)

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "fallback"],
        default="torch",
        help="Training backend. 'torch' uses MLPScorer; 'fallback' uses pure-Python linear.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="reports/candidate_reward_dataset.jsonl",
    )
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Checkpoint output path. Defaults to scorer_torch.pt or scorer_fallback.json.",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default="reports/scorer_train_metrics.json",
    )
    parser.add_argument("--question_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="SBERT model name for question encoding. If set, enables real SBERT question embeddings.",
    )
    parser.add_argument(
        "--embedding_subdir",
        type=str,
        default="sentence-transformers_all-MiniLM-L6-v2",
        help="Subdirectory under episode_dir/embeddings/ for cached frame embeddings.",
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _check_torch_available() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "ERROR: --backend torch requires PyTorch, but torch is not installed.\n"
            "Install torch or run with --backend fallback.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    args = parse_args()

    if args.backend == "torch":
        _check_torch_available()

    if args.output is None:
        if args.backend == "torch":
            args.output = "reports/scorer_torch.pt"
        else:
            args.output = "reports/scorer_fallback.json"

    sbert_model = args.embedding_model if args.backend == "torch" else None

    examples = build_scorer_training_examples(
        dataset_path=args.dataset_path,
        episode_dir=args.episode_dir,
        question_dim=args.question_dim,
        sbert_model_name=sbert_model,
        embedding_subdir=args.embedding_subdir,
    )
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    if args.backend == "torch":
        _, checkpoint, metrics = train_torch_scorer(
            examples=examples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            hidden_dim=args.hidden_dim,
        )
    else:
        checkpoint, metrics = train_fallback_scorer(
            examples=examples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

    save_scorer_outputs(
        checkpoint=checkpoint,
        metrics=metrics,
        checkpoint_path=args.output,
        metrics_path=args.metrics_output,
        backend=args.backend,
    )

    print("=" * 80)
    print("Scorer Training")
    print("=" * 80)
    print(f"Backend:      {args.backend}")
    print(f"Dataset:      {args.dataset_path}")
    print(f"Episode dir:  {args.episode_dir}")
    print(f"Examples:     {metrics['num_examples']}")
    print(f"Epochs:       {metrics['epochs']}")
    print(f"Final loss:   {metrics['final_train_loss']:.6f}")
    print(f"Checkpoint:   {args.output}")
    print(f"Metrics:      {args.metrics_output}")
    if args.dry_run:
        print("Checkpoint summary:")
        print(
            {
                "framework": checkpoint["framework"],
                "question_dim": checkpoint["question_dim"],
                "candidate_dim": checkpoint["candidate_dim"],
            }
        )


if __name__ == "__main__":
    main()
