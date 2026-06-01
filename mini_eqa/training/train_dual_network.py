from __future__ import annotations

import math
from collections import defaultdict
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
    min_reward_gap: float = 0.25,
) -> tuple[dict, dict]:
    examples, _ = build_selector_training_examples(
        dataset_path=dataset_path,
        episode_dir=episode_dir,
        question_dim=question_dim,
        min_reward_gap=min_reward_gap,
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


# ── Torch dual-network training ───────────────────────────────────────────────


def _load_torch_checkpoint(checkpoint_path: str | Path, role: str) -> dict:
    """Load a torch .pt checkpoint dict.

    Fails loudly if torch is unavailable or the file is not a torch checkpoint.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "backend=torch requires torch. Install PyTorch or use --backend fallback."
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{role} checkpoint not found: {checkpoint_path}. "
            f"Train {role} with --backend torch first."
        )

    try:
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    except Exception as exc:
        raise ValueError(
            f"Failed to load {role} checkpoint from {checkpoint_path}: {exc}\n"
            "Ensure it was saved with torch.save (backend=torch), not JSON."
        ) from exc

    if not isinstance(ckpt, dict) or ckpt.get("framework") != "torch":
        raise ValueError(
            f"Expected a torch-format checkpoint at {checkpoint_path}, "
            f"got framework='{ckpt.get('framework') if isinstance(ckpt, dict) else type(ckpt)}'.\n"
            "Use --backend fallback for JSON checkpoints, or train with --backend torch first."
        )
    return ckpt


def train_torch_dual_network(
    examples: list[SelectorTrainingExample],
    scorer_checkpoint_path: str | Path,
    selector_checkpoint_path: str | Path,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    lambda_aux: float = 0.1,
    temperature: float = 1.0,
    device: str = "cpu",
) -> tuple[object, dict, dict]:
    """Fine-tune selector against a frozen scorer using soft candidate selection.

    Training objective per question:
        loss = BCE(selector_logits, pseudo_labels)
             + lambda_aux * (-mean_scorer_utility)

    The scorer utility is computed via differentiable soft selection:
        weights = softmax(selector_logits / temperature)
        pooled  = sum_i(weights_i * frame_emb_i)   ← frames stripped of cosine feature
        utility = scorer(q_emb, pooled)

    The scorer is frozen (parameters excluded from optimizer). Gradients flow
    through the soft selection back into the selector.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.optim import Adam
    except ImportError:
        raise RuntimeError(
            "backend=torch requires torch. Install PyTorch or use --backend fallback."
        )

    from mini_eqa.scorer.mlp_scorer import MLPScorer
    from mini_eqa.selector.mlp_selector import MLPSelector

    if not examples:
        raise ValueError("No selector training examples provided.")

    # ── Load checkpoints ──────────────────────────────────────────────────────
    scorer_ckpt = _load_torch_checkpoint(scorer_checkpoint_path, "scorer")
    selector_ckpt = _load_torch_checkpoint(selector_checkpoint_path, "selector")

    scorer_q_dim = scorer_ckpt["question_dim"]
    scorer_cand_dim = scorer_ckpt["candidate_dim"]
    selector_q_dim = selector_ckpt["question_dim"]
    selector_frame_dim = selector_ckpt["frame_dim"]

    if scorer_q_dim != selector_q_dim:
        raise ValueError(
            f"Question dim mismatch: scorer={scorer_q_dim}, selector={selector_q_dim}. "
            "Both must be trained with the same embedding mode (torch+sbert or fallback)."
        )
    if scorer_cand_dim > selector_frame_dim:
        raise ValueError(
            f"Scorer candidate_dim ({scorer_cand_dim}) > selector frame_dim ({selector_frame_dim}). "
            "Cannot extract frame embeddings for the scorer auxiliary path."
        )

    actual_frame_dim = len(examples[0].frame_embedding)
    if actual_frame_dim != selector_frame_dim:
        raise ValueError(
            f"Training example frame_dim ({actual_frame_dim}) does not match "
            f"selector checkpoint frame_dim ({selector_frame_dim}). "
            "Ensure the training examples were built with the same embedding settings "
            "as the selector checkpoint."
        )

    scorer = MLPScorer(scorer_q_dim, scorer_cand_dim, scorer_ckpt["hidden_dim"])
    scorer.load_state_dict(scorer_ckpt["state_dict"])
    scorer = scorer.to(device)
    scorer.eval()
    for param in scorer.parameters():
        param.requires_grad_(False)

    selector = MLPSelector(selector_q_dim, selector_frame_dim, selector_ckpt["hidden_dim"])
    selector.load_state_dict(selector_ckpt["state_dict"])
    selector = selector.to(device)
    selector.train()

    optimizer = Adam(selector.parameters(), lr=learning_rate)
    bce_criterion = nn.BCEWithLogitsLoss()

    # ── Group examples by question ────────────────────────────────────────────
    question_groups: dict[str, list[SelectorTrainingExample]] = defaultdict(list)
    for ex in examples:
        question_groups[ex.question_id].append(ex)
    questions = list(question_groups.items())

    bce_history: list[float] = []
    aux_history: list[float] = []
    loss_history: list[float] = []

    for _ in range(epochs):
        total_bce = 0.0
        total_aux = 0.0

        for _question_id, q_examples in questions:
            q_tensor = torch.tensor(
                q_examples[0].question_embedding, dtype=torch.float32, device=device
            )
            frame_tensor = torch.stack(
                [
                    torch.tensor(ex.frame_embedding, dtype=torch.float32, device=device)
                    for ex in q_examples
                ]
            )  # [N, frame_dim]
            label_tensor = torch.tensor(
                [ex.label for ex in q_examples], dtype=torch.float32, device=device
            )  # [N]

            # Score all frames differentiably
            logits = selector(q_tensor, frame_tensor)  # [N]

            # BCE pseudo-label loss
            bce_loss = bce_criterion(logits, label_tensor)

            # Soft selection → scorer auxiliary loss
            weights = torch.softmax(logits / temperature, dim=0)  # [N]
            # Strip any extra features (e.g. cosine_sim) to match scorer candidate_dim
            scorer_frame_tensor = frame_tensor[:, :scorer_cand_dim]  # [N, scorer_cand_dim]
            pooled = (weights.unsqueeze(1) * scorer_frame_tensor).sum(0)  # [scorer_cand_dim]

            predicted_utility = scorer(q_tensor, pooled)
            if predicted_utility.dim() == 0:
                predicted_utility = predicted_utility.unsqueeze(0)
            aux_loss = -predicted_utility.mean()  # maximize predicted utility

            loss = bce_loss + lambda_aux * aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_bce += bce_loss.item()
            total_aux += aux_loss.item()

        n = max(len(questions), 1)
        avg_bce = total_bce / n
        avg_aux = total_aux / n
        bce_history.append(avg_bce)
        aux_history.append(avg_aux)
        loss_history.append(avg_bce + lambda_aux * avg_aux)

    checkpoint = {
        "framework": "torch",
        "question_dim": selector_q_dim,
        "frame_dim": selector_frame_dim,
        "hidden_dim": selector_ckpt["hidden_dim"],
        "state_dict": {k: v.cpu() for k, v in selector.state_dict().items()},
        "dual_training": {
            "strategy": "soft_selection_with_frozen_scorer",
            "lambda_aux": lambda_aux,
            "temperature": temperature,
            "scorer_question_dim": scorer_q_dim,
            "scorer_candidate_dim": scorer_cand_dim,
        },
    }
    metrics = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "lambda_aux": lambda_aux,
        "temperature": temperature,
        "device": str(device),
        "num_examples": len(examples),
        "num_questions": len(questions),
        "train_loss_history": loss_history,
        "final_train_loss": loss_history[-1],
        "bce_loss_history": bce_history,
        "final_bce_loss": bce_history[-1],
        "aux_loss_history": aux_history,
        "final_aux_loss": aux_history[-1],
    }
    return selector, checkpoint, metrics
