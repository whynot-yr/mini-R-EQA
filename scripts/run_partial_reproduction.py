from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.evaluation.evaluate_answers import evaluate_prediction_item
from mini_eqa.evaluation.run_predictions import run_prediction_pipeline
from mini_eqa.preprocess.build_caption_embeddings import (
    DEFAULT_MODEL_NAME,
    resolve_output_dir,
)
from mini_eqa.utils.io_utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a partial R-EQA reproduction on prepared episodes."
    )
    parser.add_argument("--prepared_root", type=str, required=True)
    parser.add_argument("--runner", type=str, default="mock")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--limit_episodes", type=int, default=None)
    parser.add_argument("--limit_questions", type=int, default=None)
    parser.add_argument("--prompt", type=str, default="mini_rag")
    parser.add_argument("--max_output_tokens", type=int, default=128)
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def list_episode_dirs(prepared_root: Path, limit_episodes: int | None) -> list[Path]:
    episode_dirs = sorted(path for path in prepared_root.iterdir() if path.is_dir())
    if limit_episodes is not None:
        if limit_episodes <= 0:
            raise ValueError(
                f"limit_episodes must be positive when provided, got {limit_episodes}"
            )
        episode_dirs = episode_dirs[:limit_episodes]
    return episode_dirs


def build_missing_captions_message(episode_dir: Path) -> str:
    episode_meta = episode_dir / "episode_meta.json"
    frames_dir = None
    if episode_meta.exists():
        meta = load_json(episode_meta)
        frames_dir = meta.get("frames_dir")

    frames_hint = frames_dir or "path/to/episode_frames"
    return (
        "Missing captions.json. Run captioning first: "
        f"python -m mini_eqa.captioning.caption_frames --frames_dir {frames_hint} "
        f"--output {episode_dir / 'captions.json'} --backend filename_stub --overwrite"
    )


def build_missing_embeddings_message(episode_dir: Path) -> str:
    return (
        "Missing caption embeddings. Run build_caption_embeddings first: "
        "python -m mini_eqa.preprocess.build_caption_embeddings "
        f"--episode_dir {episode_dir} "
        f"--model_name {DEFAULT_MODEL_NAME} --overwrite"
    )


def evaluate_predictions_report(predictions_path: Path, output_path: Path) -> dict:
    report = load_json(predictions_path)
    prediction_items = report["predictions"]
    results = [evaluate_prediction_item(item) for item in prediction_items]

    if results:
        average_metrics = {
            "exact_match": sum(item["metrics"]["exact_match"] for item in results) / len(results),
            "contains_gold": sum(item["metrics"]["contains_gold"] for item in results) / len(results),
            "token_f1": sum(item["metrics"]["token_f1"] for item in results) / len(results),
        }
    else:
        average_metrics = {
            "exact_match": 0.0,
            "contains_gold": 0.0,
            "token_f1": 0.0,
        }

    eval_report = {
        "input_predictions": str(predictions_path),
        "num_predictions": len(results),
        "average_metrics": average_metrics,
        "results": results,
    }
    save_json(eval_report, output_path)
    return eval_report


def run_one_method(
    episode_dir: Path,
    episode_name: str,
    method: str,
    retriever: str,
    top_k: int,
    runner: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    limit_questions: int | None,
    output_dir: Path,
    cache_dir: Path | None,
) -> dict:
    predictions_path = output_dir / f"predictions_{method}_{episode_name}.json"
    report = run_prediction_pipeline(
        episode_dir=episode_dir,
        retriever=retriever,
        runner=runner,
        top_k=top_k,
        cache_dir=cache_dir,
        prompt_name=prompt,
        model=model,
        max_output_tokens=max_output_tokens,
        output=predictions_path,
        limit=limit_questions,
    )

    answer_eval_path = output_dir / f"answer_eval_{method}_{episode_name}.json"
    answer_eval = evaluate_predictions_report(predictions_path, answer_eval_path)

    return {
        "episode": episode_name,
        "method": method,
        "status": "ok",
        "num_questions": report["num_questions"],
        "metrics": answer_eval["average_metrics"],
        "predictions_path": str(predictions_path),
        "answer_eval_path": str(answer_eval_path),
        "message": None,
    }


def run_episode(
    episode_dir: Path,
    output_dir: Path,
    runner: str,
    model: str,
    prompt: str,
    max_output_tokens: int,
    limit_questions: int | None,
    embedding_model: str,
) -> list[dict]:
    episode_name = episode_dir.name
    questions_path = episode_dir / "questions.json"
    captions_path = episode_dir / "captions.json"

    if not questions_path.exists():
        return [
            {
                "episode": episode_name,
                "method": "reqa",
                "status": "error",
                "num_questions": 0,
                "metrics": None,
                "predictions_path": None,
                "answer_eval_path": None,
                "message": f"Missing questions.json in {episode_dir}",
            },
            {
                "episode": episode_name,
                "method": "uniform",
                "status": "error",
                "num_questions": 0,
                "metrics": None,
                "predictions_path": None,
                "answer_eval_path": None,
                "message": f"Missing questions.json in {episode_dir}",
            },
        ]

    if not captions_path.exists():
        message = build_missing_captions_message(episode_dir)
        return [
            {
                "episode": episode_name,
                "method": "reqa",
                "status": "error",
                "num_questions": 0,
                "metrics": None,
                "predictions_path": None,
                "answer_eval_path": None,
                "message": message,
            },
            {
                "episode": episode_name,
                "method": "uniform",
                "status": "error",
                "num_questions": 0,
                "metrics": None,
                "predictions_path": None,
                "answer_eval_path": None,
                "message": message,
            },
        ]

    cache_dir = resolve_output_dir(episode_dir, embedding_model, None)
    embeddings_path = cache_dir / "caption_embeddings.npy"
    embedding_meta_path = cache_dir / "caption_embedding_meta.json"

    reqa_message = None
    if not embeddings_path.exists() or not embedding_meta_path.exists():
        reqa_message = build_missing_embeddings_message(episode_dir)

    results = []
    if reqa_message is None:
        results.append(
            run_one_method(
                episode_dir=episode_dir,
                episode_name=episode_name,
                method="reqa",
                retriever="cached_sbert",
                top_k=3,
                runner=runner,
                model=model,
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                limit_questions=limit_questions,
                output_dir=output_dir,
                cache_dir=cache_dir,
            )
        )
    else:
        results.append(
            {
                "episode": episode_name,
                "method": "reqa",
                "status": "error",
                "num_questions": 0,
                "metrics": None,
                "predictions_path": None,
                "answer_eval_path": None,
                "message": reqa_message,
            }
        )

    results.append(
        run_one_method(
            episode_dir=episode_dir,
            episode_name=episode_name,
            method="uniform",
            retriever="uniform",
            top_k=10,
            runner=runner,
            model=model,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            limit_questions=limit_questions,
            output_dir=output_dir,
            cache_dir=None,
        )
    )

    return results


def make_summary_markdown(summary_rows: list[dict]) -> str:
    lines = [
        "# Partial Reproduction Summary",
        "",
        "Simple answer metrics are sanity checks, not official OpenEQA LLM-Match.",
        "",
        "| episode | method | num_questions | exact_match | contains_gold | token_f1 | predictions_path |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for row in summary_rows:
        metrics = row.get("metrics") or {}
        exact_match = f"{metrics.get('exact_match', 'n/a'):.4f}" if metrics else "n/a"
        contains_gold = f"{metrics.get('contains_gold', 'n/a'):.4f}" if metrics else "n/a"
        token_f1 = f"{metrics.get('token_f1', 'n/a'):.4f}" if metrics else "n/a"
        predictions_path = row.get("predictions_path") or row.get("message") or "n/a"
        lines.append(
            f"| {row['episode']} | {row['method']} | {row['num_questions']} | "
            f"{exact_match} | {contains_gold} | {token_f1} | {predictions_path} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    prepared_root = Path(args.prepared_root)
    output_dir = Path(args.output_dir)

    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared root does not exist: {prepared_root}")
    if not prepared_root.is_dir():
        raise NotADirectoryError(f"Prepared root is not a directory: {prepared_root}")

    episode_dirs = list_episode_dirs(prepared_root, args.limit_episodes)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for episode_dir in episode_dirs:
        summary_rows.extend(
            run_episode(
                episode_dir=episode_dir,
                output_dir=output_dir,
                runner=args.runner,
                model=args.model,
                prompt=args.prompt,
                max_output_tokens=args.max_output_tokens,
                limit_questions=args.limit_questions,
                embedding_model=args.embedding_model,
            )
        )

    summary = {
        "prepared_root": str(prepared_root),
        "output_dir": str(output_dir),
        "num_episodes": len(episode_dirs),
        "rows": summary_rows,
    }
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    save_json(summary, summary_json_path)
    summary_md_path.write_text(make_summary_markdown(summary_rows), encoding="utf-8")

    print("=" * 80)
    print("Partial R-EQA Reproduction")
    print("=" * 80)
    print(f"Prepared root: {prepared_root}")
    print(f"Output dir: {output_dir}")
    print(f"Num episodes: {len(episode_dirs)}")
    print(f"Saved summary to: {summary_json_path}")
    print(f"Saved summary to: {summary_md_path}")
    for row in summary_rows:
        print("-" * 80)
        print(f"Episode: {row['episode']}")
        print(f"Method: {row['method']}")
        print(f"Status: {row['status']}")
        print(f"Message: {row.get('message')}")
        print(f"Predictions path: {row.get('predictions_path')}")


if __name__ == "__main__":
    main()
