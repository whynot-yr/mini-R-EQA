from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in dependency-light environments
    torch = None
    nn = None

from mini_eqa.scorer.mlp_scorer import MLPScorer


@dataclass
class SupervisorBatch:
    question_embeddings: Any
    candidate_embeddings: Any
    rewards: Any


class MLPSupervisor(nn.Module if nn is not None else object):
    """Minimal supervision wrapper for future scorer training."""

    def __init__(self, scorer: MLPScorer) -> None:
        self.scorer = scorer
        self.framework = "torch" if nn is not None else "unavailable"

        if nn is None:
            self.loss_fn = None
            return

        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, batch: SupervisorBatch) -> Any:
        if self.loss_fn is None:
            raise RuntimeError("MLPSupervisor requires torch to run forward().")
        predicted_rewards = self.scorer(
            question_embedding=batch.question_embeddings,
            candidate_embedding=batch.candidate_embeddings,
        )
        return self.loss_fn(predicted_rewards, batch.rewards)
