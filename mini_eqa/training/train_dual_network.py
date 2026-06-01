from __future__ import annotations

import math
from pathlib import Path

from mini_eqa.inference.scorer_inference import build_frame_auxiliary_targets
from mini_eqa.training.feature_utils import dot_product
from mini_eqa.training.selector_pseudo_labels import (
    SelectorTrainingExample,
    build_selector_training_examples,
)
from mini_eqa.utils.io_utils import load_json, save_json


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _build_feature_vector(example: SelectorTrainingExample) -> list[float]:
    return list(example.question_embedding) + list(example.frame_embedding)


def train_dual_selector(
    selector_checkpoint: dict,
    examples: list[SelectorTrainingExample],
    auxiliary_targets: dict[tuple[str, str], float],
    epochs: int = 3,
    learning_rate: float = 0.05,
    auxiliary_weight: float = 0.25,
) -> tuple[dict, dict]:
    weights = list(selector_checkpoint["weights"])
    bias = float(selector_checkpoint["bias"])
    loss_history: list[float] = []
    auxiliary_history: list[float] = []

    for _ in range(epochs):
        total_loss = 0.0
        total_auxiliary = 0.0
        auxiliary_count = 0

        for example in examples:
            features = _build_feature_vector(example)
            logit = dot_product(weights, features) + bias
            probability = _sigmoid(logit)
            label = example.label
            auxiliary_target = auxiliary_targets.get((example.question_id, example.frame_id), 0.0)

            bce_loss = -(
                label * math.log(max(probability, 1e-8))
                + (1.0 - label) * math.log(max(1.0 - probability, 1e-8))
            )
            auxiliary_loss = (probability - auxiliary_target) ** 2
            loss = bce_loss + auxiliary_weight * auxiliary_loss
            total_loss += loss
            total_auxiliary += auxiliary_target
            auxiliary_count += 1

            gradient = (probability - label) + (
                auxiliary_weight * 2.0 * (probability - auxiliary_target) * probability * (1.0 - probability)
            )
            for index, value in enumerate(features):
                weights[index] -= learning_rate * gradient * value
            bias -= learning_rate * gradient

        loss_history.append(total_loss / len(examples))
        auxiliary_history.append(total_auxiliary / max(auxiliary_count, 1))

    new_checkpoint = {
        **selector_checkpoint,
        "weights": weights,
        "bias": bias,
        "dual_training": {
            "strategy": "staged_selector_finetune_with_scorer_auxiliary",
            "auxiliary_weight": auxiliary_weight,
        },
    }
    metrics = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "auxiliary_weight": auxiliary_weight,
        "num_examples": len(examples),
        "train_loss_history": loss_history,
        "final_train_loss": loss_history[-1],
        "scorer_auxiliary_signal_history": auxiliary_history,
        "final_scorer_auxiliary_signal": auxiliary_history[-1],
    }
    return new_checkpoint, metrics


def run_dual_training(
    dataset_path: str | Path,
    episode_dir: str | Path,
    scorer_checkpoint_path: str | Path,
    selector_checkpoint_path: str | Path,
    output_checkpoint_path: str | Path,
    metrics_output_path: str | Path,
    question_dim: int = 64,
    epochs: int = 3,
    learning_rate: float = 0.05,
    auxiliary_weight: float = 0.25,
    max_examples: int | None = None,
) -> tuple[dict, dict]:
    examples, _ = build_selector_training_examples(
        dataset_path=dataset_path,
        episode_dir=episode_dir,
        question_dim=question_dim,
    )
    if max_examples is not None:
        examples = examples[:max_examples]

    auxiliary_targets = build_frame_auxiliary_targets(
        checkpoint_path=scorer_checkpoint_path,
        dataset_path=dataset_path,
        episode_dir=episode_dir,
    )
    selector_checkpoint = load_json(selector_checkpoint_path)
    new_checkpoint, metrics = train_dual_selector(
        selector_checkpoint=selector_checkpoint,
        examples=examples,
        auxiliary_targets=auxiliary_targets,
        epochs=epochs,
        learning_rate=learning_rate,
        auxiliary_weight=auxiliary_weight,
    )
    save_json(new_checkpoint, output_checkpoint_path)
    save_json(metrics, metrics_output_path)
    return new_checkpoint, metrics
