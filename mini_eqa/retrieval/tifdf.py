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
