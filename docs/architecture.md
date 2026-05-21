# Architecture

mini-R-EQA is a small, modular prototype of a retrieval-augmented episodic question answering pipeline. The codebase keeps the major R-EQA stages separate so toy data, OpenEQA-style metadata, and future real episode data can share the same execution path.

## Pipeline

### 1. `data_adapters`

`mini_eqa/data_adapters/openeqa_adapter.py` converts OpenEQA-style QA metadata into the internal `questions.json` format used by the rest of the pipeline.

- Input: OpenEQA-style JSON question-answer metadata
- Output: `questions.json` plus adapter metadata
- Scope: metadata conversion only

### 2. `captioning`

`mini_eqa/captioning/caption_frames.py` converts episode frame images into `captions.json`.

- Input: episode frames on disk
- Output: `captions.json`
- Current backend: `filename_stub`

`filename_stub` is only a pipeline test backend. It does not inspect image content and is not a real captioner.

### 3. `preprocess`

`mini_eqa/preprocess/build_caption_embeddings.py` converts `captions.json` into:

- `caption_embeddings.npy`
- `caption_embedding_meta.json`

This stage preprocesses caption text so dense retrieval can reuse cached caption embeddings later.

### 4. `retrieval`

The retrieval layer ranks caption memory items for a question.

- `tfidf`: lexical retrieval baseline
- `sbert`: on-the-fly dense retrieval by encoding captions and the question at run time
- `cached_sbert`: cached caption embeddings plus question embedding at run time
- `uniform`: question-agnostic baseline

`cached_sbert` is not a new retrieval algorithm. It is the cached implementation of the same SBERT-style dense retrieval idea, with caption encoding moved into preprocessing.

### 5. `baselines/rag.py`

`mini_eqa/baselines/rag.py` is the single-question RAG pipeline.

It loads one question, runs retrieval, builds a prompt from retrieved evidence, calls a runner, and prints the answer.

### 6. `evaluation/run_predictions.py`

`mini_eqa/evaluation/run_predictions.py` is the batch prediction pipeline.

It loads all questions in an episode, runs retrieval and prompting per question, calls the selected runner, and saves a prediction report.

### 7. `runners`

The runner layer converts a prompt into a predicted answer.

- `mock`: deterministic placeholder behavior for testing
- `deepseek`: real LLM-backed answer generation path

### 8. `evaluation`

The evaluation layer currently covers:

- retrieval metrics
- answer metrics

Retrieval evaluation operates on toy data with `gold_frame_ids`. Answer evaluation is currently a sanity check over `gold_answer` and `predicted_answer`. It is not the final OpenEQA `LLM-Match` evaluation.

## Unified Experiment Entry

The standardized experiment entrypoint is:

```bash
python scripts/run_pipeline.py --config configs/debug_toy_cached_sbert.yaml
```

This wrapper reads YAML config values and reuses the existing batch prediction logic in `mini_eqa/evaluation/run_predictions.py`.
