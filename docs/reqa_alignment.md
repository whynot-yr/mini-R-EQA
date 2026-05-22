# R-EQA Alignment

This table tracks how the current mini-R-EQA codebase maps onto the original R-EQA/OpenEQA workflow.

| R-EQA original | mini-R-EQA | Status | Notes |
| --- | --- | --- | --- |
| `openeqa/baselines/captioning_qwen.py` | `mini_eqa/captioning/caption_frames.py` + `mini_eqa/captioning/backends/qwen_vl.py` | partially aligned | Real Qwen-VL image captioning path is implemented, but server-scale benchmarking still depends on environment and data. |
| `extract_emb.py` | `mini_eqa/preprocess/build_caption_embeddings.py` | completed toy version | Precomputes caption embeddings for cached dense retrieval. |
| `openeqa/baselines/llama_rag.py` | `mini_eqa/retrieval/cached_sbert.py` + `mini_eqa/baselines/rag.py` | partially aligned | Single-question RAG path is present, but model family and official output format differ. |
| `openeqa/baselines/llama_uniform_sampling.py` | `mini_eqa/retrieval/uniform.py` | completed toy version | Question-agnostic retrieval baseline is available. |
| `evaluate-predictions.py` | `mini_eqa/exporters/openeqa_predictions.py` + `scripts/evaluate_with_openeqa.py` | partially aligned | Official evaluator invocation is wrapped, but final compatibility still depends on the real OpenEQA script and environment. |

## Summary

- Completed toy version:
  - caption embedding preprocessing
  - uniform baseline
  - answer prediction and report saving
- Partially aligned:
  - real captioning execution path
  - cached dense RAG path
  - official evaluation wrapper
- Missing:
  - full-benchmark execution validation
  - strict official numeric parity confirmation
