from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.evaluation.reward_utils import compute_reward_breakdown
from mini_eqa.inference.candidate_generation import build_candidate_sets
from mini_eqa.runners.registry import get_runner
from mini_eqa.schema import CandidateRewardRecord
from mini_eqa.utils.io_utils import load_json, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate candidate reward dataset rows for selector-scorer training."
    )
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/candidate_reward_dataset.jsonl",
    )
    parser.add_argument("--runner", type=str, default="mock")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--max_output_tokens", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
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
    if runner_name in {"openai_compatible", "llama_local"} and base_url is not None:
        runner_kwargs["base_url"] = base_url
    return runner(**runner_kwargs)


def generate_rows(
    captions: list[dict],
    questions: list[dict],
    runner: str,
    model: str,
    base_url: str | None,
    max_output_tokens: int,
    top_k: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for question_index, question_item in enumerate(questions):
        candidate_sets = build_candidate_sets(
            captions=captions,
            question=question_item["question"],
            sample_size=top_k,
            seed=seed + question_index,
        )
        for candidate in candidate_sets:
            predicted_answer = run_candidate_answer(
                runner_name=runner,
                question=question_item["question"],
                retrieved=candidate["frames"],
                model=model,
                max_output_tokens=max_output_tokens,
                base_url=base_url,
            )
            reward_breakdown = compute_reward_breakdown(
                prediction=predicted_answer,
                gold_answer=question_item.get("answer"),
            )
            row = CandidateRewardRecord(
                question_id=question_item["question_id"],
                question=question_item["question"],
                gold_answer=question_item.get("answer"),
                candidate_frames=[item["frame_id"] for item in candidate["frames"]],
                candidate_type=candidate["candidate_type"],
                predicted_answer=predicted_answer,
                reward=reward_breakdown["reward"],
                reward_breakdown=reward_breakdown,
                metadata={
                    "runner": runner,
                    "top_k": top_k,
                    "seed": seed + question_index,
                    "frame_captions": [item["caption"] for item in candidate["frames"]],
                    "frame_scores": [item["score"] for item in candidate["frames"]],
                },
            )
            rows.append(asdict(row))
    return rows


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir)
    captions, questions = load_episode_inputs(episode_dir)

    if args.limit is not None:
        questions = questions[: args.limit]

    rows = generate_rows(
        captions=captions,
        questions=questions,
        runner=args.runner,
        model=args.model,
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
        top_k=args.top_k,
        seed=args.seed,
    )
    save_jsonl(rows, args.output)

    print("=" * 80)
    print("Candidate Reward Dataset Generation")
    print("=" * 80)
    print(f"Episode dir: {episode_dir}")
    print(f"Runner: {args.runner}")
    print(f"Top-k: {args.top_k}")
    print(f"Questions processed: {len(questions)}")
    print(f"Rows generated: {len(rows)}")
    print(f"Output: {args.output}")
    if args.dry_run and rows:
        print("Sample row:")
        print(rows[0])


if __name__ == "__main__":
    main()
