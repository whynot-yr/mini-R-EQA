from __future__ import annotations

import argparse
from pathlib import Path

from mini_eqa.retrieval.registry import RETRIEVER_NAMES, get_retriever
from mini_eqa.runners.registry import RUNNER_NAMES, get_runner
from mini_eqa.utils.io_utils import load_json
from mini_eqa.utils.prompt_utils import build_prompt


DEFAULT_CACHE_MODEL_DIR = "sentence-transformers_all-MiniLM-L6-v2"


def find_question(questions: list[dict], question_id: str) -> dict:
    for q in questions:
        if q["question_id"] == question_id:
            return q
    raise ValueError(f"Question id not found: {question_id}")


def print_result(
    question: str,
    gold_answer: str | None,
    retrieved: list[dict],
    prompt: str,
    answer: str,
) -> None:
    print("=" * 80)
    print("Question:")
    print(question)
    print()
    print("Gold answer:")
    print(gold_answer)
    print()
    print("Retrieved evidence:")
    for item in retrieved:
        print(f"{item['frame_id']} | score={item['score']:.4f} | {item['caption']}")
    print()
    print("Prompt:")
    print(prompt)
    print()
    print("Answer:")
    print(answer)
    print("=" * 80)


def resolve_cache_dir(episode_dir: str | Path, cache_dir: str | Path | None) -> Path:
    episode_dir = Path(episode_dir)
    if cache_dir is not None:
        return Path(cache_dir)
    return episode_dir / "embeddings" / DEFAULT_CACHE_MODEL_DIR


def run_retrieval(
    retriever_name: str,
    captions: list[dict],
    question: str,
    top_k: int,
    episode_dir: str | Path,
    cache_dir: str | Path | None = None,
) -> tuple[list[dict], Path | None]:
    retriever = get_retriever(retriever_name)

    if retriever_name == "cached_sbert":
        resolved_cache_dir = resolve_cache_dir(episode_dir=episode_dir, cache_dir=cache_dir)
        retrieved = retriever(
            question=question,
            cache_dir=resolved_cache_dir,
            top_k=top_k,
        )
        return retrieved, resolved_cache_dir

    retrieved = retriever(
        captions=captions,
        question=question,
        top_k=top_k,
    )
    return retrieved, None


def run_answer_generation(
    runner_name: str,
    question: str,
    retrieved: list[dict],
    prompt: str,
    model: str,
    max_output_tokens: int,
    base_url: str | None = None,
) -> str:
    runner = get_runner(runner_name)

    if runner_name == "mock":
        return runner(question=question, retrieved=retrieved)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the generic mini-R-EQA retrieval and answer pipeline."
    )
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument("--question_id", type=str, default="q1")
    parser.add_argument(
        "--retriever",
        type=str,
        default="tfidf",
        choices=RETRIEVER_NAMES,
    )
    parser.add_argument("--runner", type=str, default="mock", choices=RUNNER_NAMES)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--prompt", type=str, default="mini_rag")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--max_output_tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError(f"top_k must be positive, got {args.top_k}")

    episode_dir = Path(args.episode_dir)
    captions = load_json(episode_dir / "captions.json")
    questions = load_json(episode_dir / "questions.json")

    question_item = find_question(questions, args.question_id)
    question = question_item["question"]
    gold_answer = question_item.get("answer")

    retrieved, cache_dir = run_retrieval(
        retriever_name=args.retriever,
        captions=captions,
        question=question,
        top_k=args.top_k,
        episode_dir=episode_dir,
        cache_dir=args.cache_dir,
    )

    print(f"Retriever: {args.retriever}")
    print(f"Cache dir: {cache_dir}")

    prompt = build_prompt(
        question=question,
        retrieved=retrieved,
        prompt_name=args.prompt,
    )
    answer = run_answer_generation(
        runner_name=args.runner,
        question=question,
        retrieved=retrieved,
        prompt=prompt,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        base_url=args.base_url,
    )

    print_result(
        question=question,
        gold_answer=gold_answer,
        retrieved=retrieved,
        prompt=prompt,
        answer=answer,
    )


if __name__ == "__main__":
    main()
