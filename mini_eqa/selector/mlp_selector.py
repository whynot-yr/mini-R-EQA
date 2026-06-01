from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in dependency-light environments
    torch = None
    nn = None


class MLPSelector(nn.Module if nn is not None else object):
    """Minimal selector skeleton for scoring question-frame pairs."""

    def __init__(
        self,
        question_dim: int,
        frame_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        self.question_dim = question_dim
        self.frame_dim = frame_dim
        self.hidden_dim = hidden_dim
        self.framework = "torch" if nn is not None else "unavailable"

        if nn is None:
            self.network = None
            return

        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(question_dim + frame_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        question_embedding: Any,
        frame_embeddings: Any,
    ) -> Any:
        if self.network is None:
            raise RuntimeError("MLPSelector requires torch to run forward().")
        if question_embedding.ndim == 1:
            question_embedding = question_embedding.unsqueeze(0)
        if frame_embeddings.ndim == 1:
            frame_embeddings = frame_embeddings.unsqueeze(0)

        if question_embedding.shape[0] == 1 and frame_embeddings.shape[0] > 1:
            question_embedding = question_embedding.expand(frame_embeddings.shape[0], -1)

        features = torch.cat([question_embedding, frame_embeddings], dim=-1)
        return self.network(features).squeeze(-1)
