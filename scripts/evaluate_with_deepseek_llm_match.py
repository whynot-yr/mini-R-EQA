from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.evaluation.deepseek_llm_match import run_deepseek_llm_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a DeepSeek-based OpenEQA-style LLM-Match evaluator. "
            "This is not the official OpenEQA GPT-4 score."
        )
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_deepseek_llm_match(
        dataset_path=args.dataset,
        results_path=args.results,
        output_path=args.output,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        timeout=args.timeout,
        resume=args.resume,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    metadata = payload["metadata"]
    print("=" * 80)
    print("DeepSeek LLM-Match Evaluation")
    print("=" * 80)
    print("Note: this is not the official OpenEQA GPT-4 LLM-Match score.")
    print(f"Dataset path: {metadata['dataset_path']}")
    print(f"Results path: {metadata['results_path']}")
    print(f"Model: {metadata['model']}")
    print(f"Processed count: {metadata['processed_count']}")
    print(f"Evaluated count: {metadata['evaluated_count']}")
    print(f"Failed count: {metadata['failed_count']}")
    print(f"Dry-run count: {metadata['dry_run_count']}")
    print(f"Overall mapped score: {metadata['overall_mapped_score']}")
    print(f"Output path: {args.output}")


if __name__ == "__main__":
    main()
