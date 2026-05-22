# Mini R-EQA

This repository is a minimal R-EQA-style prototype for retrieval-augmented episodic question answering.

It currently supports:

- OpenEQA-style QA metadata adaptation
- frame captioning into `captions.json`
- caption embedding preprocessing
- retrieval with `tfidf`, `sbert`, `cached_sbert`, and `uniform`
- answer generation with `mock` and `deepseek`
- local OpenAI-compatible answer generation for Llama-style server inference
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

Export OpenEQA-compatible predictions:

```bash
python -m mini_eqa.exporters.openeqa_predictions \
  --predictions reports/predictions_reqa_faithful_debug.json \
  --output reports/openeqa_predictions_reqa_faithful_debug.json \
  --include_debug_fields
```

Prepare an OpenEQA small subset:

```bash
python scripts/prepare_openeqa_subset.py \
  --qa_file data/toy_openeqa/open_eqa_toy.json \
  --frames_root data/sample_frames \
  --output_root data/openeqa_prepared_toy \
  --limit 2 \
  --overwrite
```

Reserved real-captioning command for `qwen_vl`:

```bash
python -m mini_eqa.captioning.caption_frames \
  --frames_dir path/to/episode_frames \
  --output data/openeqa_prepared/episode_x/captions.json \
  --backend qwen_vl \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda \
  --overwrite
```

Local Llama / vLLM answer runner:

```bash
python -m mini_eqa.baselines.rag \
  --retriever cached_sbert \
  --runner openai_compatible \
  --question_id q1 \
  --top_k 3 \
  --cache_dir data/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2 \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --base_url http://localhost:8000/v1
```

Run partial reproduction on prepared episodes:

```bash
python scripts/run_partial_reproduction.py \
  --prepared_root data/openeqa_prepared_toy \
  --runner mock \
  --model gpt-4o-mini \
  --output_dir reports/partial_reproduction_toy \
  --limit_episodes 1 \
  --limit_questions 2
```

Official evaluation workflow:

```bash
python scripts/run_partial_reproduction.py \
  --prepared_root data/openeqa_prepared_toy \
  --runner mock \
  --model gpt-4o-mini \
  --output_dir reports/partial_reproduction_toy \
  --limit_episodes 1 \
  --limit_questions 2

python -m mini_eqa.exporters.openeqa_predictions \
  --predictions reports/predictions_reqa_faithful_debug.json \
  --output reports/openeqa_predictions_reqa_faithful_debug.json \
  --include_debug_fields

python scripts/evaluate_with_openeqa.py \
  --internal_predictions reports/predictions_reqa_faithful_debug.json \
  --exported_predictions reports/openeqa_predictions_reqa_faithful_debug_wrapper.json \
  --openeqa_eval_script /path/to/evaluate-predictions.py \
  --output reports/openeqa_eval_wrapper_log.json \
  --include_debug_fields
```

Validate and batch-process prepared episodes:

```bash
python scripts/validate_openeqa_data.py \
  --qa_file data/toy_openeqa/open_eqa_toy.json \
  --frames_root data/sample_frames \
  --limit 2 \
  --output reports/validation_toy.json

python scripts/caption_prepared_episodes.py \
  --prepared_root data/openeqa_prepared_toy \
  --backend filename_stub \
  --limit_episodes 1 \
  --dry_run

python scripts/build_embeddings_for_prepared.py \
  --prepared_root data/openeqa_prepared_toy \
  --limit_episodes 1 \
  --dry_run

python scripts/check_reproduction_status.py \
  --prepared_root data/openeqa_prepared_toy \
  --reports_root reports/partial_reproduction_toy \
  --output reports/status_toy.json
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
- OpenEQA-compatible export notes:
  [docs/openeqa_export.md](/home/whynot/ai_research/r_net/mini_circle/docs/openeqa_export.md)
- OpenEQA small-subset directory convention:
  [docs/openeqa_subset.md](/home/whynot/ai_research/r_net/mini_circle/docs/openeqa_subset.md)
- Real captioning notes:
  [docs/real_captioning.md](/home/whynot/ai_research/r_net/mini_circle/docs/real_captioning.md)
- Local Llama 4-bit runner:
  [docs/llama_4bit_runner.md](/home/whynot/ai_research/r_net/mini_circle/docs/llama_4bit_runner.md)
- Partial reproduction notes:
  [docs/partial_reproduction.md](/home/whynot/ai_research/r_net/mini_circle/docs/partial_reproduction.md)
- Server reproduction workflow:
  [docs/server_reproduction_workflow.md](/home/whynot/ai_research/r_net/mini_circle/docs/server_reproduction_workflow.md)
- Official evaluation notes:
  [docs/official_evaluation.md](/home/whynot/ai_research/r_net/mini_circle/docs/official_evaluation.md)
- Reusable experiment configs:
  [configs/](/home/whynot/ai_research/r_net/mini_circle/configs)

## Notes

- All previous CLI entrypoints remain available.
- `filename_stub` is only for captioning pipeline tests.
- `cached_sbert` is the cached implementation of SBERT retrieval, not a separate retrieval algorithm.
- If a dataset does not provide `gold_frame_ids`, retrieval evidence evaluation is not applicable.
- Current answer evaluation is a sanity check, not the final OpenEQA LLM-Match evaluation.
