from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in dependency-light environments
    torch = None
    nn = None


class MLPScorer(nn.Module if nn is not None else object):
    """Minimal scorer skeleton for predicting candidate-set reward."""

    def __init__(
        self,
        question_dim: int,
        candidate_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        self.question_dim = question_dim
        self.candidate_dim = candidate_dim
        self.hidden_dim = hidden_dim
        self.framework = "torch" if nn is not None else "unavailable"

        if nn is None:
            self.network = None
            return

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(question_dim + candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        question_embedding: Any,
        candidate_embedding: Any,
    ) -> Any:
        if self.network is None:
            raise RuntimeError("MLPScorer requires torch to run forward().")
        if question_embedding.ndim == 1:
            question_embedding = question_embedding.unsqueeze(0)
        if candidate_embedding.ndim == 1:
            candidate_embedding = candidate_embedding.unsqueeze(0)

        features = torch.cat([question_embedding, candidate_embedding], dim=-1)
        return self.network(features).squeeze(-1)
