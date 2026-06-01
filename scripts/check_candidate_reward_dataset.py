from __future__ import annotations

"""Validate a candidate reward dataset JSONL and print diagnostic stats."""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.utils.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a candidate reward dataset JSONL and print diagnostic stats."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to JSONL file.")
    parser.add_argument(
        "--allow_zero_variance",
        action="store_true",
        help="Do not exit nonzero when reward variance is zero.",
    )
    return parser.parse_args()


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    exit_code = 0

    # ── File existence and non-empty ──────────────────────────────────────────
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_jsonl(input_path)

    if not rows:
        print(f"ERROR: file is empty: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ── Field presence checks ─────────────────────────────────────────────────
    missing_predicted = sum(1 for r in rows if not r.get("predicted_answer"))
    missing_gold = sum(1 for r in rows if not r.get("gold_answer"))
    missing_frames = sum(
        1 for r in rows if not (r.get("frame_ids") or r.get("candidate_frames"))
    )

    if missing_predicted:
        print(f"WARNING: {missing_predicted} rows missing predicted_answer", file=sys.stderr)
        exit_code = 1
    if missing_gold:
        print(f"WARNING: {missing_gold} rows missing gold_answer", file=sys.stderr)
        exit_code = 1
    if missing_frames:
        print(f"WARNING: {missing_frames} rows missing frame_ids", file=sys.stderr)
        exit_code = 1

    # ── Core stats ────────────────────────────────────────────────────────────
    rewards = [float(r.get("reward", 0.0)) for r in rows]
    question_ids = [str(r.get("question_id", "")) for r in rows]
    unique_questions = set(question_ids)
    candidate_types = Counter(str(r.get("candidate_type", "")) for r in rows)
    hard_neg_count = sum(1 for r in rows if r.get("is_hard_negative", False))
    zero_count = sum(1 for r in rewards if r == 0.0)

    mean_r = sum(rewards) / len(rewards)
    std_r = _std(rewards)

    # Usable question ratio: questions with > 1 unique reward value across their candidates
    q_rewards: dict[str, set[float]] = {}
    for r in rows:
        qid = str(r.get("question_id", ""))
        q_rewards.setdefault(qid, set()).add(float(r.get("reward", 0.0)))
    usable_questions = sum(1 for rset in q_rewards.values() if len(rset) > 1)
    usable_ratio = usable_questions / max(len(unique_questions), 1)

    # ── Reward variance check ─────────────────────────────────────────────────
    if std_r == 0.0 and not args.allow_zero_variance:
        print(
            "ERROR: reward variance is zero (all rewards identical). "
            "Training will produce degenerate pseudo labels. "
            "Use --allow_zero_variance to suppress this check.",
            file=sys.stderr,
        )
        exit_code = 1

    # ── Judge stats (if present) ──────────────────────────────────────────────
    judge_labels = [
        str(r.get("reward_breakdown", {}).get("judge_label", ""))
        for r in rows
        if r.get("reward_breakdown", {}).get("judge_label")
    ]
    has_judge = len(judge_labels) > 0
    label_dist = dict(Counter(judge_labels)) if has_judge else {}
    parse_error_count = label_dist.get("judge_parse_error", 0)

    # ── Print report ──────────────────────────────────────────────────────────
    sep = "=" * 60
    print(sep)
    print("Candidate Reward Dataset Check")
    print(sep)
    print(f"File:              {input_path}")
    print(f"Records:           {len(rows)}")
    print(f"Questions:         {len(unique_questions)}")
    print(f"Reward min:        {min(rewards):.4f}")
    print(f"Reward mean:       {mean_r:.4f}")
    print(f"Reward max:        {max(rewards):.4f}")
    print(f"Reward std:        {std_r:.4f}")
    print(f"Zero rewards:      {zero_count} / {len(rows)} ({zero_count / len(rows):.1%})")
    print(f"Hard negatives:    {hard_neg_count}")
    print(f"Usable questions:  {usable_questions} / {len(unique_questions)} ({usable_ratio:.1%})")
    print(f"Candidate types:   {dict(candidate_types)}")

    if has_judge:
        print(f"Judge labels:      {label_dist}")
        if parse_error_count:
            print(f"Judge parse errors:{parse_error_count}")

    if missing_predicted or missing_gold or missing_frames:
        print(f"\nField issues:      predicted_answer={missing_predicted} missing, "
              f"gold_answer={missing_gold} missing, frame_ids={missing_frames} missing")

    if exit_code != 0:
        print("\nValidation FAILED — see warnings above.", file=sys.stderr)
    else:
        print("\nValidation PASSED.")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
