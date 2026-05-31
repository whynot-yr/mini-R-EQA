from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrameRecord:
    frame_id: str
    caption: str
    image_path: str | None = None
    embedding_index: int | None = None


@dataclass
class QuestionRecord:
    question_id: str
    question: str
    gold_answer: str | None = None
    episode_history: str | None = None
    scene_id: str | None = None


@dataclass
class CandidateSetRecord:
    question_id: str
    candidate_type: str
    frames: list[FrameRecord] = field(default_factory=list)
    frame_ids: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    question_id: str
    selected_frame_ids: list[str]
    prompt: str | None = None
    predicted_answer: str | None = None
    score: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class CandidateRewardRecord:
    question_id: str
    question: str
    gold_answer: str | None
    candidate_frames: list[str]
    candidate_type: str
    predicted_answer: str
    reward: float
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
