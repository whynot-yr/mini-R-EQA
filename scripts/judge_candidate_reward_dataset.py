from __future__ import annotations

"""Post-hoc DeepSeek judge for existing candidate reward datasets.

Takes an existing JSONL with predicted_answer already filled in and
adds/replaces reward fields using the DeepSeek semantic judge.
Writes output incrementally so an interrupted run can be resumed.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.evaluation.llm_judge_reward import judge_answer_with_deepseek
from mini_eqa.evaluation.reward_utils import compute_reward_breakdown
from mini_eqa.utils.io_utils import load_jsonl


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required for --config support. pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _candidate_key(row: dict) -> str:
    """Stable resume key: question_id + candidate_type + sorted(frame_ids)."""
    question_id = str(row.get("question_id", ""))
    candidate_type = str(row.get("candidate_type", ""))
    frame_ids = sorted(row.get("frame_ids") or row.get("candidate_frames", []))
    candidate_id = str(row.get("candidate_id") or "")
    parts = [question_id, candidate_type, ",".join(frame_ids)]
    if candidate_id:
        parts.append(candidate_id)
    return "|".join(parts)


def _recompute_hard_negative(
    row: dict,
    hard_negative_min_score: float,
    hard_negative_max_reward: float,
) -> bool:
    """Return True iff the row qualifies as a hard negative after reward update."""
    reward = float(row.get("reward", 0.0))
    if reward > hard_negative_max_reward:
        return False

    # If the judge says correct, it cannot be a hard negative.
    judge_label = row.get("reward_breakdown", {}).get("judge_label", "")
    if judge_label == "correct":
        return False

    retrieval_scores = row.get("retrieval_scores") or []
    if not retrieval_scores:
        return False
    mean_score = sum(retrieval_scores) / len(retrieval_scores)
    return mean_score >= hard_negative_min_score


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    config_defaults: dict = {}
    if pre_args.config:
        config_defaults = _load_yaml(pre_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Add or replace reward fields in an existing candidate reward JSONL "
            "using a DeepSeek semantic judge, without regenerating candidate answers."
        )
    )
    parser.set_defaults(**config_defaults)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to existing candidate_reward_dataset.jsonl with predicted_answer filled in.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for the judged output JSONL.",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="deepseek_judge",
        choices=["local", "deepseek_judge", "hybrid"],
        help="Reward mode for the output dataset.",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="deepseek-chat",
        help="DeepSeek model for judge calls.",
    )
    parser.add_argument("--judge_max_output_tokens", type=int, default=128)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_max_retries", type=int, default=5)
    parser.add_argument("--judge_retry_initial_sleep", type=float, default=3.0)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "If output file already exists, skip rows whose candidate_key is already "
            "present and write them through unchanged. Enables resuming interrupted runs."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many rows (useful for smoke tests).",
    )
    # ── episode_id ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--infer_episode_id_from_filename",
        action="store_true",
        help=(
            "If a row has missing or null episode_id, infer it from the input filename stem. "
            "Useful when the input file is a per-episode JSONL named after its episode."
        ),
    )
    # ── hard negative recomputation ────────────────────────────────────────────
    parser.add_argument(
        "--recompute_hard_negatives",
        action="store_true",
        default=True,
        help=(
            "Recompute is_hard_negative after updating reward. "
            "Default True — hard negatives must be consistent with the final reward."
        ),
    )
    parser.add_argument(
        "--no_recompute_hard_negatives",
        dest="recompute_hard_negatives",
        action="store_false",
        help="Disable hard-negative recomputation (not recommended).",
    )
    parser.add_argument(
        "--hard_negative_min_score",
        type=float,
        default=0.5,
        help="Mean retrieval score threshold to qualify as a hard negative.",
    )
    parser.add_argument(
        "--hard_negative_max_reward",
        type=float,
        default=0.2,
        help="Reward must be <= this value to qualify as a hard negative.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Do not write output or call judge.")
    return parser.parse_args()


def _load_existing_judged(output_path: Path) -> dict[str, dict]:
    """Load existing output JSONL as a dict keyed by candidate_key."""
    if not output_path.exists():
        return {}
    existing: dict[str, dict] = {}
    try:
        for row in load_jsonl(output_path):
            existing[_candidate_key(row)] = row
    except Exception:
        pass
    return existing


def _judge_row(
    row: dict,
    reward_mode: str,
    judge_model: str,
    judge_max_output_tokens: int,
    judge_temperature: float,
    judge_max_retries: int,
    judge_retry_initial_sleep: float,
    recompute_hard_negatives: bool,
    hard_negative_min_score: float,
    hard_negative_max_reward: float,
    fallback_episode_id: str | None,
) -> dict:
    """Apply judge to a single row and return an updated copy."""
    updated = dict(row)
    question = str(row.get("question", ""))
    gold_answer = row.get("gold_answer")
    predicted_answer = str(row.get("predicted_answer", ""))

    # ── Preserve / restore episode_id ─────────────────────────────────────────
    existing_ep_id = row.get("episode_id")
    if not existing_ep_id and fallback_episode_id:
        updated["episode_id"] = fallback_episode_id
    elif existing_ep_id:
        updated["episode_id"] = existing_ep_id
    # If still missing, leave as-is (will be caught by checker).

    # ── Judge call ─────────────────────────────────────────────────────────────
    judge_result = None
    if reward_mode in ("deepseek_judge", "hybrid") and gold_answer:
        try:
            judge_result = judge_answer_with_deepseek(
                question=question,
                gold_answer=str(gold_answer),
                predicted_answer=predicted_answer,
                model=judge_model,
                max_output_tokens=judge_max_output_tokens,
                temperature=judge_temperature,
                max_retries=judge_max_retries,
                retry_initial_sleep=judge_retry_initial_sleep,
            )
        except Exception as exc:
            judge_result = {
                "score": 0.0,
                "label": "judge_api_error",
                "rationale": str(exc),
                "raw_response": "",
            }

    reward_breakdown = compute_reward_breakdown(
        prediction=predicted_answer,
        gold_answer=gold_answer,
        reward_mode=reward_mode,
        judge_result=judge_result,
    )

    updated["reward"] = float(reward_breakdown["reward"])
    updated["reward_breakdown"] = reward_breakdown

    # ── Recompute is_hard_negative ─────────────────────────────────────────────
    if recompute_hard_negatives:
        updated["is_hard_negative"] = _recompute_hard_negative(
            updated,
            hard_negative_min_score=hard_negative_min_score,
            hard_negative_max_reward=hard_negative_max_reward,
        )

    # ── Persist debug fields ───────────────────────────────────────────────────
    debug = dict(row.get("debug") or {})
    debug["reward_mode"] = reward_mode
    if judge_result is not None:
        debug["judge_raw_response"] = judge_result.get("raw_response", "")
    updated["debug"] = debug

    return updated


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.reward_mode in ("deepseek_judge", "hybrid"):
        print(
            f"WARNING: --reward_mode {args.reward_mode} will call the DeepSeek judge API. "
            "Ensure DEEPSEEK_API_KEY is set and you have budget.",
            file=sys.stderr,
        )

    # episode_id fallback derived from the filename stem (per-episode files).
    fallback_episode_id: str | None = None
    if args.infer_episode_id_from_filename:
        fallback_episode_id = input_path.stem

    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    # Warn about rows with missing episode_id in the input.
    missing_ep_id_count = sum(1 for r in rows if not r.get("episode_id"))
    if missing_ep_id_count:
        print(
            f"WARNING: {missing_ep_id_count} input rows have missing/null episode_id.",
            file=sys.stderr,
        )
        if fallback_episode_id:
            print(
                f"         Will fill episode_id='{fallback_episode_id}' "
                f"(inferred from filename).",
                file=sys.stderr,
            )

    # Load existing judged rows for resume support.
    existing_judged: dict[str, dict] = {}
    if args.skip_existing and output_path.exists():
        existing_judged = _load_existing_judged(output_path)
        print(
            f"Resume mode: found {len(existing_judged)} already-judged rows in {output_path}.",
            file=sys.stderr,
        )

    n_skipped = 0
    n_judged = 0
    n_errors = 0
    parse_errors = 0
    n_ep_id_fixed = 0
    n_hard_neg_changed = 0

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = output_path.open("w", encoding="utf-8")
    else:
        out_file = None

    try:
        for i, row in enumerate(rows):
            key = _candidate_key(row)

            if args.skip_existing and key in existing_judged:
                out_row = existing_judged[key]
                n_skipped += 1
            elif args.dry_run:
                out_row = row
            else:
                old_ep_id = row.get("episode_id")
                old_hard_neg = row.get("is_hard_negative", False)

                out_row = _judge_row(
                    row=row,
                    reward_mode=args.reward_mode,
                    judge_model=args.judge_model,
                    judge_max_output_tokens=args.judge_max_output_tokens,
                    judge_temperature=args.judge_temperature,
                    judge_max_retries=args.judge_max_retries,
                    judge_retry_initial_sleep=args.judge_retry_initial_sleep,
                    recompute_hard_negatives=args.recompute_hard_negatives,
                    hard_negative_min_score=args.hard_negative_min_score,
                    hard_negative_max_reward=args.hard_negative_max_reward,
                    fallback_episode_id=fallback_episode_id,
                )

                if not old_ep_id and out_row.get("episode_id"):
                    n_ep_id_fixed += 1
                if out_row.get("is_hard_negative") != old_hard_neg:
                    n_hard_neg_changed += 1

                label = out_row.get("reward_breakdown", {}).get("judge_label", "")
                if label == "judge_api_error":
                    n_errors += 1
                elif label == "judge_parse_error":
                    parse_errors += 1
                n_judged += 1

            if out_file is not None:
                out_file.write(json.dumps(out_row, ensure_ascii=False))
                out_file.write("\n")
                out_file.flush()

            if (i + 1) % 10 == 0 or (i + 1) == len(rows):
                print(
                    f"[progress] {i + 1}/{len(rows)} rows processed "
                    f"(judged={n_judged}, skipped={n_skipped}, "
                    f"api_errors={n_errors}, parse_errors={parse_errors})",
                    file=sys.stderr,
                )
    finally:
        if out_file is not None:
            out_file.close()

    print("=" * 80)
    print("Judge Candidate Reward Dataset")
    print("=" * 80)
    print(f"Input:                {input_path}")
    print(f"Output:               {output_path}")
    print(f"Reward mode:          {args.reward_mode}")
    print(f"Rows processed:       {len(rows)}")
    print(f"  Judged:             {n_judged}")
    print(f"  Skipped:            {n_skipped}")
    print(f"  API errors:         {n_errors}")
    print(f"  Parse errors:       {parse_errors}")
    print(f"  episode_id fixed:   {n_ep_id_fixed}")
    print(f"  hard_neg changed:   {n_hard_neg_changed}")
    if args.dry_run:
        print("(dry_run — no output written, no API calls made)")


if __name__ == "__main__":
    main()
