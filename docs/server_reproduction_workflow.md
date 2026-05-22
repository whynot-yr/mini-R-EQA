# Server Reproduction Workflow

This document outlines the intended batch workflow for running mini-R-EQA reproduction experiments on a server.

## Steps

1. validate OpenEQA data
2. prepare subset
3. batch caption prepared episodes
4. batch build embeddings
5. check status
6. run partial reproduction

## Recommended Command Flow

### 1. Validate OpenEQA data

```bash
python scripts/validate_openeqa_data.py \
  --qa_file /path/to/open-eqa-v0.json \
  --frames_root /path/to/episode_histories \
  --output reports/validation_openeqa.json
```

### 2. Prepare subset

```bash
python scripts/prepare_openeqa_subset.py \
  --qa_file /path/to/open-eqa-v0.json \
  --frames_root /path/to/episode_histories \
  --output_root /data/mini_reqa_prepared/small
```

### 3. Batch caption prepared episodes

```bash
python scripts/caption_prepared_episodes.py \
  --prepared_root /data/mini_reqa_prepared/small \
  --backend qwen_vl \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda
```

### 4. Batch build embeddings

```bash
python scripts/build_embeddings_for_prepared.py \
  --prepared_root /data/mini_reqa_prepared/small \
  --model_name sentence-transformers/all-MiniLM-L6-v2
```

### 5. Check status

```bash
python scripts/check_reproduction_status.py \
  --prepared_root /data/mini_reqa_prepared/small \
  --reports_root reports/partial_small \
  --output reports/status_small.json
```

### 6. Run partial reproduction

```bash
python scripts/run_partial_reproduction.py \
  --prepared_root /data/mini_reqa_prepared/small \
  --runner mock \
  --model gpt-4o-mini \
  --output_dir reports/partial_small
```
