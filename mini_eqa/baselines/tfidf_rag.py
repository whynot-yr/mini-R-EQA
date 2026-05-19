import argparse
from pathlib import Path

from mini_eqa.retrieval.tfidf import retrieve_topk
from mini_eqa.runners.mock_runner import mock_answer
from mini_eqa.utils.io_utils import load_json
from mini_eqa.utils.prompt_utils import build_prompt


def find_question(questions: list[dict], question_id: str) -> dict:
    for q in questions:
        if q["question_id"] == question_id:
            return q
    raise ValueError(f"Question id not found: {question_id}")


def print_result(question: str, gold_answer: str | None, retrieved: list[dict], prompt: str, answer: str) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument("--question_id", type=str, default="q1")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--prompt", type=str, default="mini_rag")
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)

    captions = load_json(episode_dir / "captions.json")
    questions = load_json(episode_dir / "questions.json")

    question_item = find_question(questions, args.question_id)

    question = question_item["question"]
    gold_answer = question_item.get("answer")

    retrieved = retrieve_topk(
        captions=captions,
        question=question,
        top_k=args.top_k,
    )

    prompt = build_prompt(
        question=question,
        retrieved=retrieved,
        prompt_name=args.prompt,
    )

    answer = mock_answer(
        question=question,
        retrieved=retrieved,
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