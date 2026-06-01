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


def train_torch_scorer(
    examples: list[ScorerTrainingExample],
    epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    hidden_dim: int = 256,
) -> tuple[object, dict, dict]:
    try:
        import torch
        import torch.nn as nn
        from torch.optim import Adam
    except ImportError:
        raise RuntimeError(
            "backend=torch requires torch. Install PyTorch or use --backend fallback."
        )

    from mini_eqa.scorer.mlp_scorer import MLPScorer

    if not examples:
        raise ValueError("No scorer training examples were provided.")

    question_dim = len(examples[0].question_embedding)
    candidate_dim = len(examples[0].candidate_embedding)

    model = MLPScorer(
        question_dim=question_dim,
        candidate_dim=candidate_dim,
        hidden_dim=hidden_dim,
    )
    if model.framework == "unavailable":
        raise RuntimeError(
            "MLPScorer failed to initialize: torch is not available inside the module."
        )
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        for example in examples:
            q = torch.tensor(example.question_embedding, dtype=torch.float32, device=device)
            c = torch.tensor(example.candidate_embedding, dtype=torch.float32, device=device)
            target = torch.tensor([example.reward], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            output = model(q, c)
            if output.dim() == 0:
                output = output.unsqueeze(0)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(total_loss / len(examples))

    checkpoint = {
        "framework": "torch",
        "question_dim": question_dim,
        "candidate_dim": candidate_dim,
        "hidden_dim": hidden_dim,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    metrics = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device": str(device),
        "num_examples": len(examples),
        "train_loss_history": losses,
        "final_train_loss": losses[-1],
    }
    return model, checkpoint, metrics


def save_scorer_outputs(
    checkpoint: dict,
    metrics: dict,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    backend: str = "fallback",
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "torch":
        import torch

        torch.save(checkpoint, checkpoint_path)
    else:
        save_json(checkpoint, checkpoint_path)
    save_json(metrics, metrics_path)
