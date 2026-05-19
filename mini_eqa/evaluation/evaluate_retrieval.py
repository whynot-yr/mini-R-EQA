from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from mini_eqa.evaluation.retrieval_metrics import (
    mrr,
    precision_at_k,
    recall_at_k,
)
from mini_eqa.retrieval.tfidf import retrieve_topk as retrieve_tfidf_topk
from mini_eqa.utils.io_utils import load_json, save_json


def get_retriever(name: str):
    """
    Return a retriever function by name.

    For v0.2, we only support TF-IDF.
    Later, you can add:
        - sbert
        - hybrid
        - uniform
    """
    if name == "tfidf":
        return retrieve_tfidf_topk

    raise ValueError(f"Unsupported retriever: {name}")


def evaluate_one_question(
    captions: list[dict[str, Any]],
    question_item: dict[str, Any],
    retriever_name: str,
    top_k: int,
) -> dict[str, Any]:
    """
    Run retrieval for one question and compute retrieval metrics.
    """
    if "gold_frame_ids" not in question_item:
        raise KeyError(
            f"Question {question_item.get('question_id')} does not have gold_frame_ids. "
            "Please add gold_frame_ids before running retrieval evaluation."
        )

    question_id = question_item["question_id"]
    question = question_item["question"]
    gold_answer = question_item.get("answer")
    gold_frame_ids = question_item["gold_frame_ids"]

    retriever = get_retriever(retriever_name)

    retrieved = retriever(
        captions=captions,
        question=question,
        top_k=top_k,
    )

    retrieved_frame_ids = [item["frame_id"] for item in retrieved]

    recall = recall_at_k(
        retrieved_frame_ids=retrieved_frame_ids,
        gold_frame_ids=gold_frame_ids,
    )
    precision = precision_at_k(
        retrieved_frame_ids=retrieved_frame_ids,
        gold_frame_ids=gold_frame_ids,
    )
    reciprocal_rank = mrr(
        retrieved_frame_ids=retrieved_frame_ids,
        gold_frame_ids=gold_frame_ids,
    )

    return {
        "question_id": question_id,
        "question": question,
        "gold_answer": gold_answer,
        "gold_frame_ids": gold_frame_ids,
        "retrieved_frame_ids": retrieved_frame_ids,
        "retrieved": retrieved,
        "metrics": {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "mrr": reciprocal_rank,
        },
    }


def evaluate_episode(
    episode_dir: str | Path,
    retriever_name: str,
    top_k: int,
) -> dict[str, Any]:
    """
    Evaluate retrieval on all questions in one toy episode.
    """
    episode_dir = Path(episode_dir)

    captions_path = episode_dir / "captions.json"
    questions_path = episode_dir / "questions.json"

    captions = load_json(captions_path)
    questions = load_json(questions_path)

    results = []

    for question_item in questions:
        result = evaluate_one_question(
            captions=captions,
            question_item=question_item,
            retriever_name=retriever_name,
            top_k=top_k,
        )
        results.append(result)

    recall_scores = [item["metrics"]["recall_at_k"] for item in results]
    precision_scores = [item["metrics"]["precision_at_k"] for item in results]
    mrr_scores = [item["metrics"]["mrr"] for item in results]

    average_metrics = {
        "recall_at_k": mean(recall_scores) if recall_scores else 0.0,
        "precision_at_k": mean(precision_scores) if precision_scores else 0.0,
        "mrr": mean(mrr_scores) if mrr_scores else 0.0,
    }

    return {
        "episode_dir": str(episode_dir),
        "retriever": retriever_name,
        "top_k": top_k,
        "num_questions": len(questions),
        "average_metrics": average_metrics,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality for mini-R-EQA."
    )

    parser.add_argument(
        "--episode_dir",
        type=str,
        default="data/sample_episode",
        help="Path to an episode folder containing captions.json and questions.json.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="tfidf",
        choices=["tfidf"],
        help="Retriever name. For v0.2, only tfidf is supported.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of retrieved evidence frames.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the evaluation result as JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError(f"top_k must be positive, got {args.top_k}")

    report = evaluate_episode(
        episode_dir=args.episode_dir,
        retriever_name=args.retriever,
        top_k=args.top_k,
    )

    print("=" * 80)
    print("Retrieval Evaluation")
    print("=" * 80)
    print(f"Episode dir: {report['episode_dir']}")
    print(f"Retriever: {report['retriever']}")
    print(f"Top-K: {report['top_k']}")
    print(f"Number of questions: {report['num_questions']}")
    print()
    print("Average metrics:")
    for metric_name, value in report["average_metrics"].items():
        print(f"  {metric_name}: {value:.4f}")

    print()
    print("Per-question results:")
    for item in report["results"]:
        metrics = item["metrics"]
        print("-" * 80)
        print(f"Question ID: {item['question_id']}")
        print(f"Question: {item['question']}")
        print(f"Gold answer: {item['gold_answer']}")
        print(f"Gold frame ids: {item['gold_frame_ids']}")
        print(f"Retrieved frame ids: {item['retrieved_frame_ids']}")
        print(
            "Metrics: "
            f"Recall@K={metrics['recall_at_k']:.4f}, "
            f"Precision@K={metrics['precision_at_k']:.4f}, "
            f"MRR={metrics['mrr']:.4f}"
        )

    if args.output is not None:
        save_json(report, args.output)
        print()
        print(f"Saved report to: {args.output}")


if __name__ == "__main__":
    main()