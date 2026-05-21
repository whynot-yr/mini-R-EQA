# R-EQA Alignment

This table tracks how the current mini-R-EQA codebase maps onto the original R-EQA/OpenEQA workflow.

| R-EQA original | mini-R-EQA | Status | Notes |
| --- | --- | --- | --- |
| `openeqa/baselines/captioning_qwen.py` | `mini_eqa/captioning/caption_frames.py` | partially aligned | Completed toy version with `filename_stub`; missing a real VLM backend. |
| `extract_emb.py` | `mini_eqa/preprocess/build_caption_embeddings.py` | completed toy version | Precomputes caption embeddings for cached dense retrieval. |
| `openeqa/baselines/llama_rag.py` | `mini_eqa/retrieval/cached_sbert.py` + `mini_eqa/baselines/rag.py` | partially aligned | Single-question RAG path is present, but model family and official output format differ. |
| `openeqa/baselines/llama_uniform_sampling.py` | `mini_eqa/retrieval/uniform.py` | completed toy version | Question-agnostic retrieval baseline is available. |
| `evaluate-predictions.py` | `mini_eqa/evaluation/evaluate_answers.py` currently | partially aligned | Current evaluator is a sanity check; missing official OpenEQA prediction exporter and LLM-Match compatibility. |

## Summary

- Completed toy version:
  - caption embedding preprocessing
  - uniform baseline
  - answer prediction and report saving
- Partially aligned:
  - frame captioning interface
  - cached dense RAG path
  - answer evaluation
- Missing:
  - real VLM captioning backend
  - OpenEQA official prediction exporter
  - official LLM-Match evaluation compatibility
