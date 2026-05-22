# Server Runbook

This checklist is intended for a 4 x RTX 3090 server setup.

## 1. Environment Setup

```bash
conda create -n mini_reqa python=3.10 -y
conda activate mini_reqa
pip install -r requirements.txt
pip install -r requirements-vlm.txt
pip install -r requirements-llama.txt
```

## 2. Local Lightweight Check

```bash
python -m py_compile $(find mini_eqa scripts -name "*.py")
python scripts/server_readiness_check.py
python scripts/run_pipeline.py --config configs/reqa_faithful_debug.yaml
```

## 3. Validate OpenEQA Data

```bash
python scripts/validate_openeqa_data.py \
  --qa_file /path/to/open-eqa-v0.json \
  --frames_root /path/to/episode_histories \
  --output reports/validation_openeqa.json
```

## 4. Prepare Small Subset

```bash
python scripts/prepare_openeqa_subset.py \
  --qa_file /path/to/open-eqa-v0.json \
  --frames_root /path/to/episode_histories \
  --output_root /data/mini_reqa_prepared/small \
  --limit 50
```

## 5. Qwen-VL Caption Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/caption_prepared_episodes.py \
  --prepared_root /data/mini_reqa_prepared/small \
  --backend qwen_vl \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --device cuda \
  --limit_episodes 1
```

Notes:

- Default single-card captioning can use `--device cuda`.
- If flash attention is available, you can try `--attn_implementation flash_attention_2`.
- If you want automatic placement, you can try `--device_map auto`.
- For 4 x 3090 captioning, prefer four independent processes with `CUDA_VISIBLE_DEVICES` pinning instead of `device_map auto`.

## 6. 4-GPU Captioning

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/caption_prepared_episodes.py --prepared_root /data/mini_reqa_prepared/full --backend qwen_vl --model_name Qwen/Qwen2.5-VL-7B-Instruct --device cuda --shard_id 0 --num_shards 4
CUDA_VISIBLE_DEVICES=1 python scripts/caption_prepared_episodes.py --prepared_root /data/mini_reqa_prepared/full --backend qwen_vl --model_name Qwen/Qwen2.5-VL-7B-Instruct --device cuda --shard_id 1 --num_shards 4
CUDA_VISIBLE_DEVICES=2 python scripts/caption_prepared_episodes.py --prepared_root /data/mini_reqa_prepared/full --backend qwen_vl --model_name Qwen/Qwen2.5-VL-7B-Instruct --device cuda --shard_id 2 --num_shards 4
CUDA_VISIBLE_DEVICES=3 python scripts/caption_prepared_episodes.py --prepared_root /data/mini_reqa_prepared/full --backend qwen_vl --model_name Qwen/Qwen2.5-VL-7B-Instruct --device cuda --shard_id 3 --num_shards 4
```

## 7. Build Embeddings

```bash
python scripts/build_embeddings_for_prepared.py \
  --prepared_root /data/mini_reqa_prepared/full \
  --model_name sentence-transformers/all-MiniLM-L6-v2
```

## 8. Check Status

```bash
python scripts/check_reproduction_status.py \
  --prepared_root /data/mini_reqa_prepared/full \
  --reports_root reports/partial_full \
  --output reports/status_full.json
```

## 9. Start Local Llama Server

Use a vLLM OpenAI-compatible server. The exact 4-bit flags may vary by installed vLLM version, so adjust them on the server as needed.

Example:

```bash
vllm serve <llama-3.1-70b-4bit-model> --host 0.0.0.0 --port 8000
```

## 10. Run Partial Reproduction

```bash
python scripts/run_partial_reproduction.py \
  --prepared_root /data/mini_reqa_prepared/full \
  --runner openai_compatible \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --base_url http://localhost:8000/v1 \
  --output_dir reports/partial_full
```

## 11. Export / Official Evaluation

```bash
python scripts/evaluate_with_openeqa.py \
  --internal_predictions reports/partial_full/predictions_reqa_episode_x.json \
  --exported_predictions reports/partial_full/openeqa_predictions_reqa_episode_x.json \
  --openeqa_eval_script /path/to/evaluate-predictions.py \
  --output reports/partial_full/openeqa_eval_episode_x.json \
  --dry_run
```

Then remove `--dry_run` to call the official evaluator.

## Important Notes

- Simple answer metrics are sanity checks only.
- Final R-EQA reproduction should use OpenEQA `evaluate-predictions.py`.
- DeepSeek is an API variant.
- Llama-3.1-70B 4-bit is a local answer-model variant unless all original R-EQA settings are exactly matched.
