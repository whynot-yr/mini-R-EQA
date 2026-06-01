from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.training.selector_pseudo_labels import build_selector_training_examples
from mini_eqa.training.train_dual_network import (
    run_dual_training,
    train_torch_dual_network,
)
from mini_eqa.utils.io_utils import load_json, load_jsonl, save_json


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
        description="Train a dual-network (frozen scorer + fine-tuned selector)."
    )
    parser.set_defaults(**config_defaults)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "fallback"],
        default="torch",
        help=(
            "'torch' loads MLPScorer/MLPSelector from .pt files, freezes scorer, "
            "fine-tunes selector with soft selection. "
            "'fallback' uses the old pure-Python linear dual training (JSON checkpoints)."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="reports/candidate_reward_dataset.jsonl",
    )
    parser.add_argument(
        "--episode_dir",
        type=str,
        default=None,
        help="Path to a single episode directory. Mutually exclusive with --prepared_root.",
    )
    parser.add_argument(
        "--prepared_root",
        type=str,
        default=None,
        help=(
            "Root directory containing episode subdirectories for multi-episode training. "
            "Each JSONL row must have an 'episode_id' field."
        ),
    )
    parser.add_argument(
        "--scorer_checkpoint",
        type=str,
        default=None,
        help="For torch: path to scorer_torch.pt. For fallback: path to scorer JSON.",
    )
    parser.add_argument(
        "--selector_checkpoint",
        type=str,
        default=None,
        help="For torch: path to selector_torch.pt. For fallback: path to selector JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output checkpoint path. Defaults to selector_dual_torch.pt or selector_dual.pt.",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default="reports/dual_train_metrics.json",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--lambda_aux",
        type=float,
        default=0.1,
        help="Weight of the scorer auxiliary loss relative to BCE pseudo-label loss.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for soft frame selection during training.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="SBERT model name for question encoding in torch mode.",
    )
    parser.add_argument(
        "--embedding_subdir",
        type=str,
        default="sentence-transformers_all-MiniLM-L6-v2",
        help="Subdirectory under episode_dir/embeddings/ for cached frame embeddings.",
    )
    parser.add_argument(
        "--min_reward_gap",
        type=float,
        default=0.25,
        help=(
            "Minimum gap between high-reward and low-reward candidates for a question "
            "to generate pseudo labels. Questions with smaller gaps are skipped. "
            "Set to 0.0 to disable filtering."
        ),
    )
    # Fallback-only args
    parser.add_argument("--question_dim", type=int, default=64)
    parser.add_argument("--auxiliary_weight", type=float, default=0.25)
    parser.add_argument("--max_examples", type=int, default=None)
    # Smoke test args
    parser.add_argument("--smoke_question_id", type=str, default=None)
    parser.add_argument("--smoke_top_k", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _check_torch_available() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "ERROR: --backend torch requires PyTorch, but torch is not installed.\n"
            "Install torch or use --backend fallback.",
            file=sys.stderr,
        )
        sys.exit(1)


def _canonical_model_name(model_name: str) -> str:
    return model_name.strip().split("/")[-1]


def _resolve_meta_dir(
    episode_dir: str | None,
    prepared_root: str | None,
    embedding_subdir: str,
    dataset_path: str,
) -> Path:
    if episode_dir is not None:
        return Path(episode_dir) / "embeddings" / embedding_subdir
    rows = load_jsonl(dataset_path)
    if not rows:
        print("ERROR: dataset is empty; cannot resolve embedding model.", file=sys.stderr)
        sys.exit(1)
    ep_id = rows[0].get("episode_id")
    if not ep_id:
        print(
            "ERROR: first JSONL row is missing episode_id; "
            "cannot auto-resolve embedding model for --prepared_root mode. "
            "Ensure JSONL rows have episode_id (re-generate or repair), "
            "or supply --embedding_model explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(prepared_root) / ep_id / "embeddings" / embedding_subdir  # type: ignore[arg-type]


def _resolve_torch_embedding_model(
    episode_dir: str | None,
    embedding_subdir: str,
    requested_model: str | None,
    prepared_root: str | None = None,
    dataset_path: str = "",
) -> str:
    meta_dir = _resolve_meta_dir(episode_dir, prepared_root, embedding_subdir, dataset_path)
    metadata_path = meta_dir / "caption_embedding_meta.json"

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


def main() -> None:
    args = parse_args()

    if args.episode_dir is None and args.prepared_root is None:
        print(
            "ERROR: Provide --episode_dir (single episode) or "
            "--prepared_root (multi-episode training).",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Apply defaults ────────────────────────────────────────────────────────
    if args.backend == "torch":
        _check_torch_available()
        if args.scorer_checkpoint is None:
            args.scorer_checkpoint = "reports/scorer_torch.pt"
        if args.selector_checkpoint is None:
            args.selector_checkpoint = "reports/selector_torch.pt"
        if args.output is None:
            args.output = "reports/selector_dual_torch.pt"
    else:
        if args.scorer_checkpoint is None:
            args.scorer_checkpoint = "reports/scorer_fallback.json"
        if args.selector_checkpoint is None:
            args.selector_checkpoint = "reports/selector_fallback.json"
        if args.output is None:
            args.output = "reports/selector_dual.pt"

    # ── Torch backend ─────────────────────────────────────────────────────────
    if args.backend == "torch":
        sbert_model = _resolve_torch_embedding_model(
            episode_dir=args.episode_dir,
            embedding_subdir=args.embedding_subdir,
            requested_model=args.embedding_model,
            prepared_root=args.prepared_root,
            dataset_path=args.dataset_path,
        )

        examples, _ = build_selector_training_examples(
            dataset_path=args.dataset_path,
            episode_dir=args.episode_dir,
            sbert_model_name=sbert_model,
            embedding_subdir=args.embedding_subdir,
            min_reward_gap=args.min_reward_gap,
            prepared_root=args.prepared_root,
        )
        if args.max_examples is not None:
            examples = examples[: args.max_examples]

        _, checkpoint, metrics = train_torch_dual_network(
            examples=examples,
            scorer_checkpoint_path=args.scorer_checkpoint,
            selector_checkpoint_path=args.selector_checkpoint,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            lambda_aux=args.lambda_aux,
            temperature=args.temperature,
            device=args.device,
        )

        import torch

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_path)
        save_json(metrics, args.metrics_output)

        print("=" * 80)
        print("Dual-Network Training (torch backend)")
        print("=" * 80)
        print(f"Dataset:           {args.dataset_path}")
        print(f"Scorer:            {args.scorer_checkpoint}")
        print(f"Selector:          {args.selector_checkpoint}")
        print(f"Output:            {args.output}")
        print(f"Metrics:           {args.metrics_output}")
        print(f"Examples:          {metrics['num_examples']}")
        print(f"Questions:         {metrics['num_questions']}")
        print(f"Epochs:            {metrics['epochs']}")
        print(f"Final loss:        {metrics['final_train_loss']:.6f}")
        print(f"  BCE component:   {metrics['final_bce_loss']:.6f}")
        print(f"  Aux component:   {metrics['final_aux_loss']:.6f}")
        print(f"Lambda_aux:        {args.lambda_aux}")
        print(f"Temperature:       {args.temperature}")

        if args.smoke_question_id is not None:
            from mini_eqa.inference.selector_inference import select_top_k_frames_torch

            selected = select_top_k_frames_torch(
                checkpoint_path=args.output,
                episode_dir=args.episode_dir,
                question_id=args.smoke_question_id,
                top_k=args.smoke_top_k,
                embedding_model=sbert_model,
                embedding_subdir=args.embedding_subdir,
                device=args.device,
            )
            print(f"\nSmoke inference (question_id={args.smoke_question_id}):")
            for item in selected:
                print(
                    f"  {item['frame_id']} | score={item['score']:.4f} | {item['caption'][:60]}"
                )

        if args.dry_run:
            print("Checkpoint summary:")
            print(
                {
                    "framework": checkpoint["framework"],
                    "question_dim": checkpoint["question_dim"],
                    "frame_dim": checkpoint["frame_dim"],
                    "dual_training": checkpoint["dual_training"],
                }
            )

    # ── Fallback backend ──────────────────────────────────────────────────────
    else:
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
            min_reward_gap=args.min_reward_gap,
            prepared_root=args.prepared_root,
        )

        smoke_qid = args.smoke_question_id or "q1"
        from mini_eqa.inference.selector_inference import select_top_k_frames

        selected = select_top_k_frames(
            checkpoint_path=args.output,
            episode_dir=args.episode_dir,
            question_id=smoke_qid,
            top_k=args.smoke_top_k,
        )

        print("=" * 80)
        print("Dual-Network Training (fallback backend)")
        print("=" * 80)
        print(f"Dataset:           {args.dataset_path}")
        print(f"Scorer:            {args.scorer_checkpoint}")
        print(f"Selector:          {args.selector_checkpoint}")
        print(f"Output:            {args.output}")
        print(f"Metrics:           {args.metrics_output}")
        print(f"Final train loss:  {metrics['final_train_loss']:.6f}")
        print(f"Final aux signal:  {metrics['final_scorer_auxiliary_signal']:.6f}")
        print(f"Selected frames (question_id={smoke_qid}):")
        for item in selected:
            print(f"  {item['frame_id']} | score={item['score']:.6f} | {item['caption']}")

        if args.dry_run:
            print("Checkpoint summary:")
            print(
                {
                    "framework": checkpoint.get("framework"),
                    "question_dim": checkpoint.get("question_dim"),
                    "frame_dim": checkpoint.get("frame_dim"),
                    "input_dim": checkpoint.get("input_dim"),
                    "dual_training": checkpoint.get("dual_training"),
                }
            )


if __name__ == "__main__":
    main()
