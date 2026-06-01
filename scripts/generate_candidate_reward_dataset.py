from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.evaluation.llm_judge_reward import judge_answer_with_deepseek
from mini_eqa.evaluation.reward_utils import compute_reward_breakdown
from mini_eqa.inference.candidate_generation import build_candidate_sets
from mini_eqa.runners.registry import get_runner
from mini_eqa.schema import CandidateRewardRecord
from mini_eqa.utils.io_utils import load_json, save_json, save_jsonl


def _load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required for --config support. pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    config_defaults: dict = {}
    if pre_args.config:
        config_defaults = _load_yaml(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Generate candidate reward dataset rows for selector-scorer training."
    )
    parser.set_defaults(**config_defaults)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument(
        "--episode_id",
        type=str,
        default=None,
        help=(
            "episode_id to embed in every output row. "
            "If omitted, inferred from basename(episode_dir). "
            "Required for multi-episode training with --prepared_root."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/candidate_reward_dataset.jsonl",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default="reports/candidate_reward_summary.json",
    )
    parser.add_argument("--runner", type=str, default="mock", choices=["mock", "deepseek"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--max_output_tokens", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--embedding_cache_dir",
        type=str,
        default=None,
        help="Directory containing caption_embeddings.npy and caption_embedding_meta.json. "
        "If provided, uses cached SBERT ranking instead of lexical overlap.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="SBERT model name override. If omitted, read from cache metadata.",
    )
    parser.add_argument(
        "--hard_negative_min_score",
        type=float,
        default=0.5,
        help="Retrieval score threshold above which a low-reward candidate is a hard negative.",
    )
    parser.add_argument(
        "--hard_negative_max_reward",
        type=float,
        default=0.2,
        help="Reward threshold below which a high-retrieval-score candidate is a hard negative.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum retry attempts for transient deepseek API failures (SSL, connection, 429, 5xx).",
    )
    parser.add_argument(
        "--retry_initial_sleep",
        type=float,
        default=3.0,
        help="Initial sleep (seconds) before first retry; doubles on each subsequent attempt.",
    )
    # ── Judge / reward mode ────────────────────────────────────────────────────
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="local",
        choices=["local", "deepseek_judge", "hybrid"],
        help=(
            "Reward computation mode. "
            "'local' uses lexical metrics (default, no extra API calls). "
            "'deepseek_judge' calls DeepSeek once per candidate row to judge semantic correctness. "
            "'hybrid' = max(judge_score, exact_match)."
        ),
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="deepseek-chat",
        help="DeepSeek model for judge calls (only used when reward_mode != local).",
    )
    parser.add_argument("--judge_max_output_tokens", type=int, default=128)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument(
        "--judge_max_retries",
        type=int,
        default=5,
        help="Max retry attempts for judge API calls.",
    )
    parser.add_argument(
        "--judge_retry_initial_sleep",
        type=float,
        default=3.0,
        help="Initial sleep before judge retry.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_episode_inputs(episode_dir: Path) -> tuple[list[dict], list[dict]]:
    captions = load_json(episode_dir / "captions.json")
    questions = load_json(episode_dir / "questions.json")
    return captions, questions


def run_candidate_answer(
    runner_name: str,
    question: str,
    retrieved: list[dict],
    model: str,
    max_output_tokens: int,
    base_url: str | None,
    max_retries: int = 5,
    retry_initial_sleep: float = 3.0,
) -> str:
    runner = get_runner(runner_name)
    if runner_name == "mock":
        return runner(question=question, retrieved=retrieved)

    prompt_lines = [f"Question: {question}", "Evidence:"]
    for item in retrieved:
        prompt_lines.append(f"- {item['frame_id']}: {item['caption']}")
    prompt = "\n".join(prompt_lines)

    runner_kwargs = {
        "question": question,
        "retrieved": retrieved,
        "prompt": prompt,
        "model": model,
        "max_output_tokens": max_output_tokens,
    }
    if runner_name == "deepseek":
        runner_kwargs["max_retries"] = max_retries
        runner_kwargs["retry_initial_sleep"] = retry_initial_sleep
    if runner_name in {"openai_compatible", "llama_local"} and base_url is not None:
        runner_kwargs["base_url"] = base_url
    return runner(**runner_kwargs)


def _warn_if_zero_variance(rewards: list[float], runner: str) -> None:
    if len(rewards) < 2:
        return
    variance = sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)
    if variance == 0.0:
        warnings.warn(
            f"All rewards are identical ({rewards[0]:.4f}) across {len(rewards)} candidates. "
            f"This is expected with runner='{runner}' on trivial data but will cause degenerate "
            "pseudo labels in selector training. Use --runner deepseek for real training data.",
            stacklevel=2,
        )


def generate_rows(
    captions: list[dict],
    questions: list[dict],
    runner: str,
    model: str,
    base_url: str | None,
    max_output_tokens: int,
    top_k: int,
    seed: int,
    embedding_cache_dir: str | None,
    embedding_model: str | None,
    hard_negative_min_score: float,
    hard_negative_max_reward: float,
    max_retries: int = 5,
    retry_initial_sleep: float = 3.0,
    reward_mode: str = "local",
    judge_model: str = "deepseek-chat",
    judge_max_output_tokens: int = 128,
    judge_temperature: float = 0.0,
    judge_max_retries: int = 5,
    judge_retry_initial_sleep: float = 3.0,
    episode_id: str = "",
) -> list[dict]:
    rows: list[dict] = []
    for question_index, question_item in enumerate(questions):
        candidate_sets = build_candidate_sets(
            captions=captions,
            question=question_item["question"],
            sample_size=top_k,
            seed=seed + question_index,
            cache_dir=embedding_cache_dir,
            model_name=embedding_model,
        )
        question_rows: list[dict] = []
        for candidate in candidate_sets:
            predicted_answer = run_candidate_answer(
                runner_name=runner,
                question=question_item["question"],
                retrieved=candidate["frames"],
                model=model,
                max_output_tokens=max_output_tokens,
                base_url=base_url,
                max_retries=max_retries,
                retry_initial_sleep=retry_initial_sleep,
            )
            judge_result = None
            if reward_mode in ("deepseek_judge", "hybrid"):
                gold = question_item.get("answer")
                if gold:
                    try:
                        judge_result = judge_answer_with_deepseek(
                            question=question_item["question"],
                            gold_answer=gold,
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
                gold_answer=question_item.get("answer"),
                reward_mode=reward_mode,
                judge_result=judge_result,
            )

            frame_ids = [item["frame_id"] for item in candidate["frames"]]
            retrieval_scores = [float(item.get("score", 0.0)) for item in candidate["frames"]]
            retrieval_ranks = [int(item.get("rank", 0)) for item in candidate["frames"]]
            mean_retrieval_score = sum(retrieval_scores) / max(len(retrieval_scores), 1)
            reward_value = float(reward_breakdown["reward"])

            # is_hard_negative is only meaningful when SBERT cosine scores are in use;
            # lexical overlap counts are unbounded and cannot be compared to [0,1] thresholds.
            is_hard_negative = (
                embedding_cache_dir is not None
                and mean_retrieval_score >= hard_negative_min_score
                and reward_value <= hard_negative_max_reward
            )

            record = CandidateRewardRecord(
                question_id=question_item["question_id"],
                question=question_item["question"],
                gold_answer=question_item.get("answer"),
                frame_ids=frame_ids,
                candidate_type=candidate["candidate_type"],
                predicted_answer=predicted_answer,
                reward=reward_value,
                reward_breakdown=reward_breakdown,
                retrieval_scores=retrieval_scores,
                retrieval_ranks=retrieval_ranks,
                is_hard_negative=is_hard_negative,
                top_k=top_k,
                episode_id=episode_id or None,
                selected_items=[
                    {
                        "frame_id": item["frame_id"],
                        "caption": item["caption"],
                        "score": float(item.get("score", 0.0)),
                        "rank": int(item.get("rank", 0)),
                    }
                    for item in candidate["frames"]
                ],
                debug={
                    "runner": runner,
                    "seed": seed + question_index,
                    "rank_source": candidate["frames"][0].get("rank_source", "unknown")
                    if candidate["frames"]
                    else "unknown",
                },
                metadata={
                    "top_k": top_k,
                    "seed": seed + question_index,
                    "frame_captions": [item["caption"] for item in candidate["frames"]],
                    "frame_scores": [item.get("score", 0.0) for item in candidate["frames"]],
                },
            )
            question_rows.append(asdict(record))

        _warn_if_zero_variance(
            [row["reward"] for row in question_rows],
            runner=runner,
        )
        rows.extend(question_rows)
    return rows


def build_summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "num_questions": 0,
            "num_candidates": 0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "reward_min": 0.0,
            "reward_max": 0.0,
            "num_zero_rewards": 0,
            "num_hard_negatives": 0,
            "candidate_type_distribution": {},
        }

    rewards = [row["reward"] for row in rows]
    n = len(rewards)
    mean_r = sum(rewards) / n
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / n) ** 0.5

    from collections import Counter

    type_dist = dict(Counter(row["candidate_type"] for row in rows))
    question_ids = {row["question_id"] for row in rows}

    return {
        "num_questions": len(question_ids),
        "num_candidates": n,
        "reward_mean": round(mean_r, 6),
        "reward_std": round(std_r, 6),
        "reward_min": round(min(rewards), 6),
        "reward_max": round(max(rewards), 6),
        "num_zero_rewards": sum(1 for r in rewards if r == 0.0),
        "num_hard_negatives": sum(1 for row in rows if row.get("is_hard_negative", False)),
        "candidate_type_distribution": type_dist,
    }


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir)
    captions, questions = load_episode_inputs(episode_dir)

    # episode_id: explicit arg > basename of episode_dir.  Never null.
    episode_id: str = args.episode_id or episode_dir.name
    if not episode_id:
        print(
            "ERROR: could not determine episode_id. "
            "Pass --episode_id explicitly or use a non-empty --episode_dir path.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.runner == "deepseek":
        print(
            "WARNING: --runner deepseek will call the DeepSeek API. "
            "Ensure DEEPSEEK_API_KEY is set and you have budget.",
            file=sys.stderr,
        )
    if args.reward_mode in ("deepseek_judge", "hybrid"):
        print(
            f"WARNING: --reward_mode {args.reward_mode} calls DeepSeek judge once per candidate. "
            "This roughly doubles API usage when combined with --runner deepseek.",
            file=sys.stderr,
        )

    if args.limit is not None:
        questions = questions[: args.limit]

    embedding_cache_dir = args.embedding_cache_dir
    if embedding_cache_dir is None and (episode_dir / "embeddings").exists():
        subdirs = sorted((episode_dir / "embeddings").iterdir())
        if subdirs:
            embedding_cache_dir = str(subdirs[0])

    rows = generate_rows(
        captions=captions,
        questions=questions,
        runner=args.runner,
        model=args.model,
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
        top_k=args.top_k,
        seed=args.seed,
        embedding_cache_dir=embedding_cache_dir,
        embedding_model=args.embedding_model,
        hard_negative_min_score=args.hard_negative_min_score,
        hard_negative_max_reward=args.hard_negative_max_reward,
        max_retries=args.max_retries,
        retry_initial_sleep=args.retry_initial_sleep,
        reward_mode=args.reward_mode,
        judge_model=args.judge_model,
        judge_max_output_tokens=args.judge_max_output_tokens,
        judge_temperature=args.judge_temperature,
        judge_max_retries=args.judge_max_retries,
        judge_retry_initial_sleep=args.judge_retry_initial_sleep,
        episode_id=episode_id,
    )

    if not args.dry_run:
        save_jsonl(rows, args.output)
        summary = build_summary(rows)
        save_json(summary, args.summary_output)
    else:
        summary = build_summary(rows)

    print("=" * 80)
    print("Candidate Reward Dataset Generation")
    print("=" * 80)
    print(f"Episode dir:       {episode_dir}")
    print(f"Runner:            {args.runner}")
    print(f"Reward mode:       {args.reward_mode}")
    print(f"Embedding cache:   {embedding_cache_dir or 'lexical fallback'}")
    print(f"Top-k:             {args.top_k}")
    print(f"Questions:         {len(questions)}")
    print(f"Rows generated:    {len(rows)}")
    print(f"Hard negatives:    {summary['num_hard_negatives']}")
    print(f"Zero rewards:      {summary['num_zero_rewards']} / {len(rows)}")
    print(f"Reward mean/std:   {summary['reward_mean']:.4f} / {summary['reward_std']:.4f}")
    if args.reward_mode in ("deepseek_judge", "hybrid") and rows:
        from collections import Counter
        labels = [
            row.get("reward_breakdown", {}).get("judge_label", "")
            for row in rows
        ]
        label_counts = dict(Counter(l for l in labels if l))
        parse_errors = sum(1 for l in labels if l == "judge_parse_error")
        print(f"Judge labels:      {label_counts}")
        if parse_errors:
            print(f"Judge parse errors: {parse_errors}")
    if args.dry_run:
        print("(dry_run — outputs not written)")
        if rows:
            print("Sample row keys:", list(rows[0].keys()))
    else:
        print(f"Output:            {args.output}")
        print(f"Summary:           {args.summary_output}")


if __name__ == "__main__":
    main()
