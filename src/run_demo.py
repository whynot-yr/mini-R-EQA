import json
import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_topk(captions, question, top_k=3):
    """
    captions: list of {"frame_id": str, "caption": str}
    question: str
    return: list of retrieved evidence
    """

    caption_texts = [item["caption"] for item in captions]

    # 把所有 caption 和 question 一起编码到 TF-IDF 空间
    texts = caption_texts + [question]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    caption_vectors = vectors[:-1]
    question_vector = vectors[-1]

    # 计算 question 和每个 caption 的余弦相似度
    scores = cosine_similarity(question_vector, caption_vectors)[0]

    # 从高到低排序，取 top-k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "frame_id": captions[idx]["frame_id"],
            "caption": captions[idx]["caption"],
            "score": float(scores[idx])
        })

    return results


def build_prompt(question, retrieved):
    evidence_text = "\n".join(
        [
            f"- {item['frame_id']}: {item['caption']}"
            for item in retrieved
        ]
    )

    prompt = f"""You are answering a question based on retrieved episodic memory.

Question:
{question}

Retrieved evidence:
{evidence_text}

Please answer the question using only the evidence above.
"""
    return prompt


def mock_answer(question, retrieved):
    """
    第一版先不用 LLM。
    这里先返回最相关 caption，让你确认 retrieval 有没有工作。
    """
    best = retrieved[0]
    return f"Mock answer based on top evidence: {best['caption']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", type=str, default="data/sample_episode")
    parser.add_argument("--question_id", type=str, default="q1")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    episode_dir = Path(args.episode_dir)

    captions = load_json(episode_dir / "captions.json")
    questions = load_json(episode_dir / "questions.json")

    question_item = None
    for q in questions:
        if q["question_id"] == args.question_id:
            question_item = q
            break

    if question_item is None:
        raise ValueError(f"Question id not found: {args.question_id}")

    question = question_item["question"]
    gold_answer = question_item.get("answer", None)

    retrieved = retrieve_topk(captions, question, top_k=args.top_k)
    prompt = build_prompt(question, retrieved)
    answer = mock_answer(question, retrieved)

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


if __name__ == "__main__":
    main()
