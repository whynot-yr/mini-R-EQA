from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from mini_eqa.evaluation.answer_metrics import contains_gold, exact_match, token_f1
from mini_eqa.utils.io_utils import load_json, save_json


def evaluate_prediction_item(item: dict) -> dict:
    if "gold_answer" not in item:
        raise KeyError(
            f"Prediction {item.get('question_id')} is missing gold_answer."
        )
    if "predicted_answer" not in item:
        raise KeyError(
            f"Prediction {item.get('question_id')} is missing predicted_answer."
        )

    gold_answer = item["gold_answer"]
    predicted_answer = item["predicted_answer"]

    if gold_answer is None:
        raise ValueError(
            f"Prediction {item.get('question_id')} has gold_answer=None."
        )
    if predicted_answer is None:
        raise ValueError(
            f"Prediction {item.get('question_id')} has predicted_answer=None."
        )

    metrics = {
        "exact_match": exact_match(predicted_answer, gold_answer),
        "contains_gold": contains_gold(predicted_answer, gold_answer),
        "token_f1": token_f1(predicted_answer, gold_answer),
    }

    return {
        "question_id": item.get("question_id"),
        "question": item.get("question"),
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted answers against gold answers."
    )
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictions_path = Path(args.predictions)
    report = load_json(predictions_path)

    if "predictions" not in report:
        raise KeyError(
            f"Prediction report {predictions_path} does not contain a predictions field."
        )

    prediction_items = report["predictions"]
    results = [evaluate_prediction_item(item) for item in prediction_items]

    average_metrics = {
        "exact_match": mean(item["metrics"]["exact_match"] for item in results) if results else 0.0,
        "contains_gold": mean(item["metrics"]["contains_gold"] for item in results) if results else 0.0,
        "token_f1": mean(item["metrics"]["token_f1"] for item in results) if results else 0.0,
    }

    answer_report = {
        "input_predictions": str(predictions_path),
        "num_predictions": len(results),
        "average_metrics": average_metrics,
        "results": results,
    }

    print("=" * 80)
    print("Answer Evaluation")
    print("=" * 80)
    print(f"Input predictions: {answer_report['input_predictions']}")
    print(f"Number of predictions: {answer_report['num_predictions']}")
    print()
    print("Average metrics:")
    for metric_name, value in average_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    print()
    print("Per-question results:")
    for item in results:
        metrics = item["metrics"]
        print("-" * 80)
        print(f"Question ID: {item['question_id']}")
        print(f"Question: {item['question']}")
        print(f"Gold answer: {item['gold_answer']}")
        print(f"Predicted answer: {item['predicted_answer']}")
        print(
            "Metrics: "
            f"EM={metrics['exact_match']:.4f}, "
            f"ContainsGold={metrics['contains_gold']:.4f}, "
            f"TokenF1={metrics['token_f1']:.4f}"
        )

    if args.output is not None:
        save_json(answer_report, args.output)
        print()
        print(f"Saved report to: {args.output}")


if __name__ == "__main__":
    main()
