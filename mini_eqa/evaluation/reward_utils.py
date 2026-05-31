from __future__ import annotations

from mini_eqa.evaluation.answer_metrics import contains_gold, exact_match, token_f1


def compute_reward_breakdown(prediction: str, gold_answer: str | None) -> dict[str, float]:
    if gold_answer is None:
        return {
            "exact_match": 0.0,
            "contains_gold": 0.0,
            "token_f1": 0.0,
            "reward": 0.0,
        }

    exact = exact_match(prediction, gold_answer)
    contains = contains_gold(prediction, gold_answer)
    overlap = token_f1(prediction, gold_answer)
    reward = max(exact, contains, overlap)
    return {
        "exact_match": exact,
        "contains_gold": contains,
        "token_f1": overlap,
        "reward": reward,
    }
