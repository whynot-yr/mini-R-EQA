from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.training.selector_pseudo_labels import build_selector_training_examples
from mini_eqa.training.train_selector import (
    save_selector_outputs,
    train_fallback_selector,
    train_torch_selector,
)
from mini_eqa.utils.io_utils import save_json
from mini_eqa.utils.io_utils import load_json


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
        description="Train a selector model from frame-level pseudo labels."
    )
    parser.set_defaults(**config_defaults)

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "fallback"],
        default="torch",
        help="Training backend. 'torch' uses MLPSelector; 'fallback' uses pure-Python linear.",
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
        help="Checkpoint output path. Defaults to selector_torch.pt or selector_fallback.json.",
    )
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
    parser.add_argument(
        "--min_reward_gap",
        type=float,
        default=0.25,
        help=(
            "Minimum gap between high-reward and low-reward candidates for a question "
            "to generate pseudo labels. Questions with smaller gaps are skipped entirely. "
            "Set to 0.0 to disable filtering (not recommended)."
        ),
    )
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


def _canonical_model_name(model_name: str) -> str:
    return model_name.strip().split("/")[-1]


def _resolve_torch_embedding_model(
    episode_dir: str,
    embedding_subdir: str,
    requested_model: str | None,
) -> str:
    metadata_path = (
        Path(episode_dir)
        / "embeddings"
        / embedding_subdir
        / "caption_embedding_meta.json"
    )
    if not metadata_path.exists():
        print(
            "ERROR: --backend torch requires caption embedding metadata at "
            f"{metadata_path}. Build embeddings first or use --backend fallback.",
            file=sys.stderr,
        )
        sys.exit(1)

    metadata = load_json(metadata_path)
    metadata_model = metadata.get("model_name")
    if not metadata_model:
        print(
            "ERROR: caption embedding metadata is missing model_name. "
            f"Cannot verify question embedding model for {metadata_path}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if requested_model is not None:
        if _canonical_model_name(requested_model) != _canonical_model_name(metadata_model):
            print(
                "ERROR: --embedding_model does not match the caption embedding model family.\n"
                f"Requested: {requested_model}\n"
                f"Cached captions: {metadata_model}",
                file=sys.stderr,
            )
            sys.exit(1)
        return str(metadata_model)

    return str(metadata_model)


def main() -> None:
    args = parse_args()

    if args.backend == "torch":
        _check_torch_available()

    if args.output is None:
        if args.backend == "torch":
            args.output = "reports/selector_torch.pt"
        else:
            args.output = "reports/selector_fallback.json"

    sbert_model = None
    if args.backend == "torch":
        sbert_model = _resolve_torch_embedding_model(
            episode_dir=args.episode_dir,
            embedding_subdir=args.embedding_subdir,
            requested_model=args.embedding_model,
        )

    examples, summaries = build_selector_training_examples(
        dataset_path=args.dataset_path,
        episode_dir=args.episode_dir,
        question_dim=args.question_dim,
        sbert_model_name=sbert_model,
        embedding_subdir=args.embedding_subdir,
        min_reward_gap=args.min_reward_gap,
    )
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    if args.backend == "torch":
        _, checkpoint, metrics = train_torch_selector(
            examples=examples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            hidden_dim=args.hidden_dim,
        )
    else:
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
        backend=args.backend,
    )
    save_json(summaries, args.summary_output)

    print("=" * 80)
    print("Selector Training")
    print("=" * 80)
    print(f"Backend:           {args.backend}")
    print(f"Dataset:           {args.dataset_path}")
    print(f"Episode dir:       {args.episode_dir}")
    print(f"Min reward gap:    {args.min_reward_gap}")
    print(f"Questions used:    {summaries['num_questions_used']} / {summaries['num_questions_total']}")
    print(f"Questions skipped: {summaries['num_questions_skipped']}")
    print(f"Positive frames:   {summaries['num_positive_frames']}")
    print(f"Negative frames:   {summaries['num_negative_frames']}")
    print(f"Examples:          {metrics['num_examples']}")
    print(f"Epochs:            {metrics['epochs']}")
    print(f"Final loss:        {metrics['final_train_loss']:.6f}")
    print(f"Checkpoint:        {args.output}")
    print(f"Metrics:           {args.metrics_output}")
    print(f"Pseudo-label summary: {args.summary_output}")
    if args.dry_run:
        print("Checkpoint summary:")
        print(
            {
                "framework": checkpoint["framework"],
                "question_dim": checkpoint["question_dim"],
                "frame_dim": checkpoint["frame_dim"],
            }
        )


if __name__ == "__main__":
    main()
