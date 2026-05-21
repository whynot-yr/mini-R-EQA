# Mini R-EQA

This repository is a minimal R-EQA-style prototype for retrieval-augmented episodic question answering.

It currently supports:

- OpenEQA-style QA metadata adaptation
- frame captioning into `captions.json`
- caption embedding preprocessing
- retrieval with `tfidf`, `sbert`, `cached_sbert`, and `uniform`
- answer generation with `mock` and `deepseek`
- prediction report saving and lightweight evaluation

## Core Flow

```text
OpenEQA-style QA metadata -> questions.json
episode frames -> captions.json
captions.json -> caption embeddings
retrieval -> prompt -> runner -> prediction report -> evaluation
```

## Quickstart

Build caption embeddings:

```bash
python -m mini_eqa.preprocess.build_caption_embeddings \
  --episode_dir data/sample_episode \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --output_dir data/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2 \
  --overwrite
```

Run cached SBERT RAG:

```bash
python -m mini_eqa.baselines.rag \
  --retriever cached_sbert \
  --runner mock \
  --question_id q1 \
  --top_k 3 \
  --cache_dir data/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2
```

Run the batch pipeline from config:

```bash
python scripts/run_pipeline.py --config configs/debug_toy_cached_sbert.yaml
```

## R-EQA Faithful Debug Protocol

- `configs/reqa_faithful_debug.yaml` uses `cached_sbert` top-3.
- `configs/reqa_uniform_debug.yaml` uses `uniform` top-10.
- These are debug configs on toy data, not final reproduction.

```bash
python scripts/run_pipeline.py --config configs/reqa_faithful_debug.yaml
python scripts/run_pipeline.py --config configs/reqa_uniform_debug.yaml
```

## Docs

- Architecture and module roles:
  [docs/architecture.md](/home/whynot/ai_research/r_net/mini_circle/docs/architecture.md)
- Data schemas:
  [docs/data_schema.md](/home/whynot/ai_research/r_net/mini_circle/docs/data_schema.md)
- R-EQA alignment status:
  [docs/reqa_alignment.md](/home/whynot/ai_research/r_net/mini_circle/docs/reqa_alignment.md)
- Reproduction checklist:
  [docs/reproduction_checklist.md](/home/whynot/ai_research/r_net/mini_circle/docs/reproduction_checklist.md)
- Reusable experiment configs:
  [configs/](/home/whynot/ai_research/r_net/mini_circle/configs)

## Notes

- All previous CLI entrypoints remain available.
- `filename_stub` is only for captioning pipeline tests.
- `cached_sbert` is the cached implementation of SBERT retrieval, not a separate retrieval algorithm.
- If a dataset does not provide `gold_frame_ids`, retrieval evidence evaluation is not applicable.
- Current answer evaluation is a sanity check, not the final OpenEQA LLM-Match evaluation.
