from __future__ import annotations

from math import exp, log
from pathlib import Path

from mini_eqa.training.feature_utils import dot_product
from mini_eqa.training.selector_pseudo_labels import SelectorTrainingExample
from mini_eqa.utils.io_utils import save_json


def _build_feature_vector(example: SelectorTrainingExample) -> list[float]:
    return list(example.question_embedding) + list(example.frame_embedding)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def train_fallback_selector(
    examples: list[SelectorTrainingExample],
    epochs: int = 5,
    learning_rate: float = 0.05,
) -> tuple[dict, dict]:
    if not examples:
        raise ValueError("No selector training examples were provided.")

    input_dim = len(_build_feature_vector(examples[0]))
    weights = [0.0] * input_dim
    bias = 0.0
    losses: list[float] = []

    for _ in range(epochs):
        total_loss = 0.0
        for example in examples:
            features = _build_feature_vector(example)
            logit = dot_product(weights, features) + bias
            probability = _sigmoid(logit)
            label = example.label

            loss = -(
                label * log(max(probability, 1e-8))
                + (1.0 - label) * log(max(1.0 - probability, 1e-8))
            )
            total_loss += loss

            error = probability - label
            for index, value in enumerate(features):
                weights[index] -= learning_rate * error * value
            bias -= learning_rate * error

        losses.append(total_loss / len(examples))

    checkpoint = {
        "framework": "python_fallback_linear",
        "question_dim": len(examples[0].question_embedding),
        "frame_dim": len(examples[0].frame_embedding),
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


def save_selector_outputs(
    checkpoint: dict,
    metrics: dict,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
) -> None:
    save_json(checkpoint, checkpoint_path)
    save_json(metrics, metrics_path)
