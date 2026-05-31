from __future__ import annotations

from pathlib import Path

from mini_eqa.training.feature_utils import dot_product
from mini_eqa.training.scorer_dataset import ScorerTrainingExample
from mini_eqa.utils.io_utils import save_json


def _build_feature_vector(example: ScorerTrainingExample) -> list[float]:
    return list(example.question_embedding) + list(example.candidate_embedding)


def train_fallback_scorer(
    examples: list[ScorerTrainingExample],
    epochs: int = 5,
    learning_rate: float = 0.05,
) -> tuple[dict, dict]:
    if not examples:
        raise ValueError("No scorer training examples were provided.")

    input_dim = len(_build_feature_vector(examples[0]))
    weights = [0.0] * input_dim
    bias = 0.0
    losses: list[float] = []

    for _ in range(epochs):
        total_loss = 0.0
        for example in examples:
            features = _build_feature_vector(example)
            prediction = dot_product(weights, features) + bias
            error = prediction - example.reward
            loss = error * error
            total_loss += loss

            gradient_scale = 2.0 * error
            for index, value in enumerate(features):
                weights[index] -= learning_rate * gradient_scale * value
            bias -= learning_rate * gradient_scale

        losses.append(total_loss / len(examples))

    checkpoint = {
        "framework": "python_fallback_linear",
        "question_dim": len(examples[0].question_embedding),
        "candidate_dim": len(examples[0].candidate_embedding),
        "input_dim": input_dim,
        "weights": weights,
        "bias": bias,
    }
    metrics = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "num_examples": len(examples),
        "train_loss_history": losses,
        "final_train_loss": losses[-1],
    }
    return checkpoint, metrics


def save_scorer_outputs(
    checkpoint: dict,
    metrics: dict,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
) -> None:
    save_json(checkpoint, checkpoint_path)
    save_json(metrics, metrics_path)
