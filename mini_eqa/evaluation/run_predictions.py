from __future__ import annotations

import argparse
from pathlib import Path

from mini_eqa.baselines.rag import run_answer_generation, run_retrieval
from mini_eqa.retrieval.registry import RETRIEVER_NAMES
from mini_eqa.runners.registry import RUNNER_NAMES
from mini_eqa.utils.io_utils import load_json, save_json
from mini_eqa.utils.prompt_utils import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch question answering and save prediction reports."
    )
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument("--retriever", type=str, default="cached_sbert", choices=RETRIEVER_NAMES)
    parser.add_argument("--runner", type=str, default="mock", choices=RUNNER_NAMES)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="mini_rag")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_output_tokens", type=int, default=128)
    parser.add_argument("--output", type=str, default="reports/predictions_v0.7.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError(f"top_k must be positive, got {args.top_k}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"limit must be positive when provided, got {args.limit}")

    episode_dir = Path(args.episode_dir)
    captions = load_json(episode_dir / "captions.json")
    questions = load_json(episode_dir / "questions.json")

    if args.limit is not None:
        questions = questions[:args.limit]

    predictions = []
    resolved_cache_dir = None

    for question_item in questions:
        question = question_item["question"]
        retrieved, resolved_cache_dir = run_retrieval(
            retriever_name=args.retriever,
            captions=captions,
            question=question,
            top_k=args.top_k,
            episode_dir=episode_dir,
            cache_dir=args.cache_dir,
        )
        prompt = build_prompt(
            question=question,
            retrieved=retrieved,
            prompt_name=args.prompt,
        )
        predicted_answer = run_answer_generation(
            runner_name=args.runner,
            question=question,
            retrieved=retrieved,
            prompt=prompt,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        predictions.append(
            {
                "question_id": question_item["question_id"],
                "question": question,
                "gold_answer": question_item.get("answer"),
                "gold_frame_ids": question_item.get("gold_frame_ids"),
                "retrieved": retrieved,
                "prompt": prompt,
                "predicted_answer": predicted_answer,
            }
        )

    report = {
        "episode_dir": str(episode_dir),
        "retriever": args.retriever,
        "runner": args.runner,
        "top_k": args.top_k,
        "model": args.model,
        "cache_dir": str(resolved_cache_dir) if resolved_cache_dir is not None else None,
        "num_questions": len(predictions),
        "predictions": predictions,
    }

    save_json(report, args.output)

    print("=" * 80)
    print("Prediction Run")
    print("=" * 80)
    print(f"Episode dir: {report['episode_dir']}")
    print(f"Retriever: {report['retriever']}")
    print(f"Runner: {report['runner']}")
    print(f"Model: {report['model']}")
    print(f"Top-K: {report['top_k']}")
    print(f"Number of questions: {report['num_questions']}")
    print(f"Saved report to: {args.output}")


if __name__ == "__main__":
    main()
