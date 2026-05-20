from __future__ import annotations

from mini_eqa.retrieval.cached_sbert import retrieve_topk as retrieve_cached_sbert_topk
from mini_eqa.retrieval.sbert import retrieve_topk as retrieve_sbert_topk
from mini_eqa.retrieval.tfidf import retrieve_topk as retrieve_tfidf_topk
from mini_eqa.retrieval.uniform import retrieve_topk as retrieve_uniform_topk


RETRIEVER_NAMES = ["tfidf", "sbert", "cached_sbert", "uniform"]


def get_retriever(name: str):
    if name == "tfidf":
        return retrieve_tfidf_topk
    if name == "sbert":
        return retrieve_sbert_topk
    if name == "cached_sbert":
        return retrieve_cached_sbert_topk
    if name == "uniform":
        return retrieve_uniform_topk
    raise ValueError(f"Unsupported retriever: {name}")
