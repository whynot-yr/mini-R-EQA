from __future__ import annotations

"""Repair data integrity issues in candidate reward dataset JSONL files.

Fixes:
  1. Missing/null episode_id — inferred from filename stem.
  2. Inconsistent is_hard_negative — recomputed from final reward and
     retrieval scores after judge reward has been applied.

Supports:
  - A single JSONL file.
  - A directory of per-episode JSONL files (each named <episode_id>.jsonl).
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.utils.io_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair episode_id and is_hard_negative integrity issues in a candidate "
            "reward dataset JSONL or a directory of per-episode JSONL files."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to a JSONL file or a directory containing per-episode JSONL files. "
            "Directory mode infers episode_id from each file's basename stem."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output path. If --input is a directory, this is the merged output JSONL. "
            "If --input is a file, this is the repaired output JSONL."
        ),
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
    parser.add_argument(
        "--no_recompute_hard_negatives",
        action="store_true",
        help="Skip is_hard_negative recomputation (only fix episode_id).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Report counts without writing.")
    return parser.parse_args()


def _recompute_hard_negative(
    row: dict,
    hard_negative_min_score: float,
    hard_negative_max_reward: float,
) -> bool:
    reward = float(row.get("reward", 0.0))
    if reward > hard_negative_max_reward:
        return False
    judge_label = row.get("reward_breakdown", {}).get("judge_label", "")
    if judge_label == "correct":
        return False
    retrieval_scores = row.get("retrieval_scores") or []
    if not retrieval_scores:
        return False
    mean_score = sum(retrieval_scores) / len(retrieval_scores)
    return mean_score >= hard_negative_min_score


def _collect_input_files(input_path: Path) -> list[tuple[Path, str | None]]:
    """Return list of (file_path, episode_id_hint) pairs.

    episode_id_hint is the file stem when input is a directory, else None.
    """
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        if not files:
            print(f"ERROR: no .jsonl files found in directory: {input_path}", file=sys.stderr)
            sys.exit(1)
        return [(f, f.stem) for f in files]
    else:
        return [(input_path, None)]


def repair_rows(
    rows: list[dict],
    episode_id_hint: str | None,
    hard_negative_min_score: float,
    hard_negative_max_reward: float,
    recompute_hard_negatives: bool,
) -> tuple[list[dict], dict]:
    """Repair rows in-place and return (repaired_rows, counts)."""
    n_ep_id_fixed = 0
    n_hard_neg_changed = 0
    n_inconsistent_found = 0

    repaired = []
    for row in rows:
        r = dict(row)

        # Fix episode_id
        existing_ep_id = r.get("episode_id")
        if not existing_ep_id and episode_id_hint:
            r["episode_id"] = episode_id_hint
            n_ep_id_fixed += 1

        # Recompute is_hard_negative
        if recompute_hard_negatives:
            old_hn = r.get("is_hard_negative", False)
            new_hn = _recompute_hard_negative(
                r,
                hard_negative_min_score=hard_negative_min_score,
                hard_negative_max_reward=hard_negative_max_reward,
            )
            if new_hn != old_hn:
                if old_hn and not new_hn:
                    n_inconsistent_found += 1
                r["is_hard_negative"] = new_hn
                n_hard_neg_changed += 1

        repaired.append(r)

    return repaired, {
        "n_ep_id_fixed": n_ep_id_fixed,
        "n_hard_neg_changed": n_hard_neg_changed,
        "n_inconsistent_found": n_inconsistent_found,
        "n_rows": len(rows),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    input_files = _collect_input_files(input_path)
    print(f"Input files: {len(input_files)}", file=sys.stderr)

    all_rows: list[dict] = []
    total_ep_id_fixed = 0
    total_hard_neg_changed = 0
    total_inconsistent = 0
    total_rows = 0

    for file_path, ep_hint in input_files:
        rows = load_jsonl(file_path)
        repaired, counts = repair_rows(
            rows=rows,
            episode_id_hint=ep_hint,
            hard_negative_min_score=args.hard_negative_min_score,
            hard_negative_max_reward=args.hard_negative_max_reward,
            recompute_hard_negatives=not args.no_recompute_hard_negatives,
        )
        all_rows.extend(repaired)
        total_ep_id_fixed += counts["n_ep_id_fixed"]
        total_hard_neg_changed += counts["n_hard_neg_changed"]
        total_inconsistent += counts["n_inconsistent_found"]
        total_rows += counts["n_rows"]
        if ep_hint:
            print(
                f"  {file_path.name}: {counts['n_rows']} rows, "
                f"ep_id_fixed={counts['n_ep_id_fixed']}, "
                f"hard_neg_changed={counts['n_hard_neg_changed']}",
                file=sys.stderr,
            )

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")

    print("=" * 70)
    print("Repair Candidate Reward Dataset")
    print("=" * 70)
    print(f"Input:                   {input_path}")
    print(f"Output:                  {output_path}")
    print(f"Rows processed:          {total_rows}")
    print(f"Missing episode_id fixed:{total_ep_id_fixed}")
    print(f"hard_neg changed:        {total_hard_neg_changed}")
    print(f"Inconsistent HN found:   {total_inconsistent}")
    if args.dry_run:
        print("(dry_run — no output written)")
    else:
        print(f"Written to:              {output_path}")


if __name__ == "__main__":
    main()
