from __future__ import annotations

from typing import Any

from mini_eqa.evaluation.answer_metrics import contains_gold, exact_match, token_f1

REWARD_MODES = ("local", "deepseek_judge", "hybrid")


def compute_reward_breakdown(
    prediction: str,
    gold_answer: str | None,
    reward_mode: str = "local",
    judge_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a reward breakdown dict for a predicted answer.

    Args:
        prediction: The predicted answer string.
        gold_answer: The gold / reference answer, or None if unavailable.
        reward_mode: One of "local", "deepseek_judge", or "hybrid".
            - local: reward = max(exact_match, token_f1)
            - deepseek_judge: reward = judge_score from judge_result
            - hybrid: reward = max(judge_score, exact_match)
        judge_result: Required when reward_mode is "deepseek_judge" or "hybrid".
            A dict with keys: score, label, rationale, raw_response.

    Returns:
        A JSON-serializable dict with all metric fields and the final "reward".
        contains_gold is always included for diagnostics but never used as reward.
    """
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")

    if gold_answer is None:
        base = {
            "exact_match": 0.0,
            "contains_gold": 0.0,
            "token_f1": 0.0,
            "local_reward": 0.0,
            "judge_score": 0.0,
            "judge_label": "incorrect",
            "judge_rationale": "",
            "reward": 0.0,
            "reward_mode": reward_mode,
        }
        if judge_result is not None:
            base["judge_score"] = float(judge_result.get("score", 0.0))
            base["judge_label"] = str(judge_result.get("label", "incorrect"))
            base["judge_rationale"] = str(judge_result.get("rationale", ""))
        return base

    exact = exact_match(prediction, gold_answer)
    contains = contains_gold(prediction, gold_answer)
    overlap = token_f1(prediction, gold_answer)
    local_reward = max(exact, overlap)

    judge_score = 0.0
    judge_label = "incorrect"
    judge_rationale = ""

    if judge_result is not None:
        judge_score = float(judge_result.get("score", 0.0))
        judge_label = str(judge_result.get("label", "incorrect"))
        judge_rationale = str(judge_result.get("rationale", ""))

    if reward_mode == "local":
        reward = local_reward
    elif reward_mode == "deepseek_judge":
        reward = judge_score
    elif reward_mode == "hybrid":
        reward = max(judge_score, exact)
    else:
        reward = local_reward

    return {
        "exact_match": exact,
        "contains_gold": contains,
        "token_f1": overlap,
        "local_reward": local_reward,
        "judge_score": judge_score,
        "judge_label": judge_label,
        "judge_rationale": judge_rationale,
        "reward": reward,
        "reward_mode": reward_mode,
    }
