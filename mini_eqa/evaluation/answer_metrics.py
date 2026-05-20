from __future__ import annotations

import re
import string
from collections import Counter


PUNCT_TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """
    Normalize text for simple answer matching.

    Steps:
    - lowercase
    - strip surrounding whitespace
    - remove simple ASCII punctuation
    - collapse repeated whitespace
    """
    normalized = text.lower().strip().translate(PUNCT_TRANSLATION_TABLE)
    return re.sub(r"\s+", " ", normalized).strip()


def exact_match(prediction: str, gold: str) -> float:
    """
    Return 1.0 when normalized prediction exactly matches normalized gold.
    """
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def contains_gold(prediction: str, gold: str) -> float:
    """
    Return 1.0 when normalized gold appears as a substring of normalized prediction.
    """
    normalized_prediction = normalize_text(prediction)
    normalized_gold = normalize_text(gold)

    if not normalized_gold:
        return 1.0 if not normalized_prediction else 0.0

    return 1.0 if normalized_gold in normalized_prediction else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """
    Compute token-level F1 using whitespace tokenization after normalization.

    This is a simple lexical overlap metric and is not semantic.
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
