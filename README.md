# Mini R-EQA

This repository is a minimal R-EQA-style prototype for understanding retrieval-augmented episodic memory question answering.

The project currently supports four retrieval modes:

- `tfidf`: lexical retrieval baseline
- `sbert`: dense retrieval with on-the-fly caption encoding
- `cached_sbert`: dense retrieval with precomputed caption embeddings
- `uniform`: question-agnostic uniform sampling baseline

## Current Status

- `v0.1-TF-IDF`: original one-file TF-IDF demo
- `v0.2-EVAL`: retrieval evaluation with Recall@K / Precision@K / MRR
- `v0.3-GENERIC-RAG`: generic retrieval + prompt + mock runner entrypoint
- `v0.4-SBERT`: dense retrieval baseline
- `v0.5-EMBEDDING-CACHE`: offline caption embedding cache for dense retrieval
- `v0.6-UNIFORM`: question-agnostic baseline plus lightweight retrieval registry
- `v0.7-REAL-LLM`: real LLM runner plus prediction saving
- `v0.9-OPENEQA-ADAPTER`: OpenEQA-style QA metadata adapter for small-subset experiments
- `v1.0-FRAME-CAPTIONING`: frames-to-captions pipeline with a lightweight stub backend

The project now supports both a mock answer path and a real LLM runner path for answer generation.

## Project Structure

```text
mini_eqa/
├── baselines/
│   ├── rag.py
│   └── tfidf_rag.py
├── captioning/
│   ├── backends.py
│   └── caption_frames.py
├── data_adapters/
│   └── openeqa_adapter.py
├── preprocess/
│   └── build_caption_embeddings.py
├── retrieval/
│   ├── cached_sbert.py
│   ├── registry.py
│   ├── sbert.py
│   ├── tfidf.py
│   └── uniform.py
├── runners/
│   ├── deepseek_runner.py
│   ├── mock_runner.py
│   └── registry.py
└── utils/
    ├── io_utils.py
    └── prompt_utils.py

data/
└── sample_episode/
    ├── captions.json
    ├── embeddings/
    └── questions.json

prompts/
└── mini_rag.txt

reports/
notes/
```

## Run

Run the original TF-IDF RAG baseline:

```
python -m mini_eqa.baselines.tfidf_rag --question_id q1 --top_k 3
python -m mini_eqa.baselines.tfidf_rag --question_id q2 --top_k 3
```

Run the generic pipeline with TF-IDF:

```
python -m mini_eqa.baselines.rag --retriever tfidf --runner mock --question_id q1 --top_k 3
```

Run the generic pipeline with on-the-fly SBERT:

```
python -m mini_eqa.baselines.rag --retriever sbert --runner mock --question_id q1 --top_k 3
```

Run the uniform baseline:

```
python -m mini_eqa.baselines.rag \
  --retriever uniform \
  --runner mock \
  --question_id q1 \
  --top_k 3
```

Build caption embedding cache:

```
python -m mini_eqa.preprocess.build_caption_embeddings \
  --episode_dir data/sample_episode \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --output_dir data/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2 \
  --overwrite
```

Run generic RAG with cached SBERT:

```
python -m mini_eqa.baselines.rag \
  --retriever cached_sbert \
  --runner mock \
  --question_id q1 \
  --top_k 3 \
  --cache_dir data/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2
```

Single question with deepseek runner:

```
python -m mini_eqa.baselines.rag \
  --retriever cached_sbert \
  --runner deepseek \
  --question_id q1 \
  --top_k 3 \
  --cache_dir data/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2 \
  --model gpt-4o-mini
```

Evaluate cached SBERT:

```
python -m mini_eqa.evaluation.evaluate_retrieval \
  --episode_dir data/sample_episode \
  --retriever cached_sbert \
  --top_k 3 \
  --cache_dir data/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2 \
  --output reports/eval_cached_sbert_v0.5_top3.json
```

Evaluate uniform retrieval:

```
python -m mini_eqa.evaluation.evaluate_retrieval \
  --episode_dir data/sample_episode \
  --retriever uniform \
  --top_k 3 \
  --output reports/eval_uniform_v0.6_top3.json
```

Batch predictions, first 2 questions:

```
python -m mini_eqa.evaluation.run_predictions \
  --episode_dir data/sample_episode \
  --retriever cached_sbert \
  --runner deepseek \
  --top_k 3 \
  --cache_dir data/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2 \
  --model gpt-4o-mini \
  --limit 2 \
  --output reports/predictions_cached_sbert_deepseek_v0.7.json
```

Generate predictions:

```
python -m mini_eqa.evaluation.run_predictions \
  --episode_dir data/sample_episode \
  --retriever cached_sbert \
  --runner mock \
  --top_k 3 \
  --cache_dir data/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2 \
  --limit 2 \
  --output reports/predictions_cached_sbert_mock_v0.8.json
```

Evaluate answers:

```
python -m mini_eqa.evaluation.evaluate_answers \
  --predictions reports/predictions_cached_sbert_mock_v0.8.json \
  --output reports/answer_eval_cached_sbert_mock_v0.8.json
```

Convert an OpenEQA-style QA file into mini-R-EQA `questions.json`:

```
python -m mini_eqa.data_adapters.openeqa_adapter \
  --qa_file path/to/open-eqa-v0.json \
  --output_dir data/openeqa_subset \
  --limit 10 \
  --overwrite
```

Caption episode frames into `captions.json`:

```
python -m mini_eqa.captioning.caption_frames \
  --frames_dir data/sample_frames \
  --output data/generated_episode/captions.json \
  --backend filename_stub \
  --overwrite
```

Prepare `questions.json` for the generated episode:

```text
Place your question file at data/generated_episode/questions.json
```

Build caption embeddings for the generated episode:

```
python -m mini_eqa.preprocess.build_caption_embeddings \
  --episode_dir data/generated_episode \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --overwrite
```

Run retrieval on the generated episode:

```
python -m mini_eqa.baselines.rag \
  --episode_dir data/generated_episode \
  --retriever cached_sbert \
  --runner mock \
  --question_id q1 \
  --top_k 3
```

Save example outputs:

```
python -m mini_eqa.baselines.tfidf_rag --question_id q1 --top_k 3 | tee reports/tfidf_v0.1_q1_reqa_style.txt
python -m mini_eqa.baselines.tfidf_rag --question_id q2 --top_k 3 | tee reports/tfidf_v0.1_q2_reqa_style.txt
```

Notes:

- `sbert` does dense retrieval by encoding captions and the question on the fly.
- `cached_sbert` uses pre-built caption embeddings and only encodes the question at inference time.
- `cached_sbert` is closer to the preprocessing + inference split used in real R-EQA systems.
- `uniform` ignores the question and acts as a simple baseline against question-aware retrieval methods.
- `deepseek` turns retrieved evidence plus the prompt into a natural-language predicted answer.
- `exact_match` is strict, `contains_gold` is useful for toy answers, and `token_f1` gives a simple lexical overlap score.
- `v0.9` only adapts QA metadata, while `v1.0` adds the frames-to-captions stage.
- `filename_stub` only validates the file pipeline. It is not a real captioning model.
- `qwen_vl` and `blip` are reserved backend names and currently raise a clear not-implemented error.
- If a dataset does not provide `gold_frame_ids`, retrieval evidence evaluation is not applicable.
- Answer generation and answer evaluation can still run as long as gold answers are present.

## Current Limitations

- LLM answer quality depends on both retrieval quality and the external model/API.
- TF-IDF mainly captures lexical overlap.
- Uniform retrieval does not use the question at all.
- Top-k retrieval may include irrelevant captions when useful evidence is limited.
- The current data is only a toy sample episode.
- The v1.0 captioner backend is only a filename-based placeholder.
- Real OpenEQA-style evaluation will eventually need semantic or LLM-based answer judging.

## Next Steps

- Compare retrieval variants on a larger toy dataset or real episodic data.
- Add answer-level evaluation comparing predicted answers with gold answers.
- Implement a real VLM captioning backend and run it on a small OpenEQA subset.
