from __future__ import annotations

import argparse
from pathlib import Path

from mini_eqa.retrieval.cached_sbert import retrieve_topk as retrieve_cached_sbert_topk
from mini_eqa.retrieval.sbert import retrieve_topk as retrieve_sbert_topk
from mini_eqa.retrieval.tfidf import retrieve_topk as retrieve_tfidf_topk
from mini_eqa.runners.mock_runner import mock_answer
from mini_eqa.utils.io_utils import load_json
from mini_eqa.utils.prompt_utils import build_prompt


DEFAULT_CACHE_MODEL_DIR = "sentence-transformers_all-MiniLM-L6-v2"


def get_retriever(name: str):
    if name == "tfidf":
        return retrieve_tfidf_topk
    if name == "sbert":
        return retrieve_sbert_topk
    if name == "cached_sbert":
        return retrieve_cached_sbert_topk
    raise ValueError(f"Unsupported retriever: {name}")


def get_runner(name: str):
    if name == "mock":
        return mock_answer
    raise ValueError(f"Unsupported runner: {name}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the generic mini-R-EQA retrieval and mock-answer pipeline."
    )
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument("--question_id", type=str, default="q1")
    parser.add_argument(
        "--retriever",
        type=str,
        default="tfidf",
        choices=["tfidf", "sbert", "cached_sbert"],
    )
    parser.add_argument("--runner", type=str, default="mock", choices=["mock"])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--prompt", type=str, default="mini_rag")
    parser.add_argument("--cache_dir", type=str, default=None)
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

    retriever = get_retriever(args.retriever)
    runner = get_runner(args.runner)
    cache_dir = None

    if args.retriever == "cached_sbert":
        cache_dir = Path(args.cache_dir) if args.cache_dir is not None else (
            episode_dir / "embeddings" / DEFAULT_CACHE_MODEL_DIR
        )
        retrieved = retriever(
            question=question,
            cache_dir=cache_dir,
            top_k=args.top_k,
        )
    else:
        retrieved = retriever(
            captions=captions,
            question=question,
            top_k=args.top_k,
        )

    print(f"Retriever: {args.retriever}")
    print(f"Cache dir: {cache_dir}")

    prompt = build_prompt(
        question=question,
        retrieved=retrieved,
        prompt_name=args.prompt,
    )
    answer = runner(question=question, retrieved=retrieved)

    print_result(
        question=question,
        gold_answer=gold_answer,
        retrieved=retrieved,
        prompt=prompt,
        answer=answer,
    )


if __name__ == "__main__":
    main()
