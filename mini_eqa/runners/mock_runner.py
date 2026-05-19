def mock_answer(question, retrieved):
    """
    第一版先不用 LLM。
    这里先返回最相关 caption，让你确认 retrieval 有没有工作。
    """
    best = retrieved[0]
    return f"Mock answer based on top evidence: {best['caption']}"