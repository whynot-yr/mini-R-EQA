# Server Training Guide

## Overview

This guide explains how to run the full selector/scorer training pipeline on a server with GPU.

The pipeline consists of five steps:
1. Generate candidate reward dataset (requires DeepSeek API access)
2. Train scorer (MLPScorer, torch backend)
3. Train selector (MLPSelector, torch backend)
4. Dual-network fine-tuning (optional, fallback format)
5. Selector inference smoke test

## Requirements

### Environment

| Package | Notes |
|---------|-------|
| Python 3.10+ | |
| `torch >= 2.0` | Required for steps 2–5 |
| `sentence-transformers` | Required for SBERT ranking and question encoding |
| `pyyaml` | Required for `--config` YAML support |
| `numpy` | Required throughout |
| `DEEPSEEK_API_KEY` env var | Required for step 1 with `--runner deepseek` |

Install:
```bash
pip install torch sentence-transformers pyyaml numpy
```

### Data Layout

The pipeline expects these files to already exist before training:

```
data/openeqa_prepared/sample_episode/
├── captions.json                    # list of {frame_id, caption}
├── questions.json                   # list of {question_id, question, answer}
└── embeddings/
    └── sentence-transformers_all-MiniLM-L6-v2/
        ├── caption_embeddings.npy           # float32 [N, 384]
        └── caption_embedding_meta.json      # {model_name, items: [{frame_id, caption}]}
```

Do **not** run captioning or embedding generation from this guide — those steps are separate.

## Quick Start

From the project root, run the complete pipeline:

```bash
bash scripts/server_train_selector_scorer.sh
```

The script runs all 5 steps in order using configs from `configs/server/`.

## Step-by-Step

### Step 1: Generate Candidate Reward Dataset

```bash
python3 scripts/generate_candidate_reward_dataset.py \
    --config configs/server/generate_candidate_reward_dataset.yaml
```

**Outputs:**
- `outputs/candidate_reward_dataset.jsonl` — one row per question-candidate pair
- `outputs/candidate_reward_summary.json` — aggregate stats

**Notes:**
- Uses `--runner deepseek` (requires `DEEPSEEK_API_KEY`)
- For local smoke tests, run with `--runner mock --limit 5 --dry_run` instead
- If all rewards are zero, training will abort unless `allow_zero_variance_reward: true`

### Step 2: Train Scorer

```bash
python3 scripts/train_scorer.py --config configs/server/train_scorer.yaml
```

**Outputs:**
- `outputs/scorer_torch.pt` — torch checkpoint (loadable with `torch.load`)
- `outputs/scorer_train_metrics.json` — loss history

**Config key fields:**
```yaml
backend: torch
embedding_model: all-MiniLM-L6-v2   # SBERT model for question encoding
epochs: 20
learning_rate: 0.001
device: cuda
```

### Step 3: Train Selector

```bash
python3 scripts/train_selector.py --config configs/server/train_selector.yaml
```

**Outputs:**
- `outputs/selector_torch.pt` — torch checkpoint
- `outputs/selector_train_metrics.json`
- `outputs/selector_pseudo_label_summary.json`

### Step 4: Dual-Network Training (optional)

```bash
python3 scripts/train_dual_network.py --config configs/server/train_dual_network.yaml
```

> **Note:** The dual network trainer currently uses the fallback JSON checkpoint format.
> Point `scorer_checkpoint` and `selector_checkpoint` at the fallback outputs
> (`scorer_fallback.json`, `selector_fallback.json`), not the torch `.pt` files.

**Outputs:**
- `outputs/selector_dual.pt` (JSON format, despite the `.pt` extension)
- `outputs/dual_train_metrics.json`

### Step 5: Selector Inference Smoke Test

```bash
python3 scripts/selector_inference_smoke_test.py \
    --config configs/server/run_selector_inference.yaml
```

This smoke test currently loads the fallback-format selector produced by step 4
(`outputs/selector_dual.pt`, JSON content despite the `.pt` suffix), not the
torch-native selector checkpoint from step 3.

## Verifying Outputs

After training, verify:

1. Checkpoint files exist and are non-empty:
   ```bash
   ls -lh outputs/*.pt outputs/*.json
   ```

2. Scorer checkpoint is loadable:
   ```python
   import torch
   from mini_eqa.scorer.mlp_scorer import MLPScorer
   ckpt = torch.load("outputs/scorer_torch.pt", map_location="cpu")
   model = MLPScorer(ckpt["question_dim"], ckpt["candidate_dim"], ckpt["hidden_dim"])
   model.load_state_dict(ckpt["state_dict"])
   print("OK, params:", sum(p.numel() for p in model.parameters()))
   ```

3. Review loss curves in the metrics JSON:
   ```bash
   python3 -c "import json; m=json.load(open('outputs/scorer_train_metrics.json')); print(m['train_loss_history'])"
   ```

## Resuming or Rerunning

- Each step is independent. Re-run only the steps whose outputs are missing or stale.
- The dataset generation step is the most expensive (DeepSeek API calls). Cache `outputs/candidate_reward_dataset.jsonl` and skip step 1 if rerunning.
- Training steps (2–4) are cheap on GPU; rerunning them is fine.

## Copying Outputs Back Locally

After training, copy checkpoints and metrics back to your local machine:

```bash
scp server:/path/to/project/outputs/scorer_torch.pt ./reports/
scp server:/path/to/project/outputs/selector_torch.pt ./reports/
scp server:/path/to/project/outputs/scorer_train_metrics.json ./reports/
scp server:/path/to/project/outputs/selector_train_metrics.json ./reports/
scp server:/path/to/project/outputs/candidate_reward_summary.json ./reports/
```

Or with rsync:
```bash
rsync -avz server:/path/to/project/outputs/ ./reports/server_outputs/
```

## Configuration Reference

All server configs live under `configs/server/`. Key settings per file:

| Config | Key Settings |
|--------|-------------|
| `generate_candidate_reward_dataset.yaml` | `runner`, `embedding_cache_dir`, `embedding_model`, `top_k` |
| `train_scorer.yaml` | `backend`, `device`, `epochs`, `learning_rate`, `embedding_model` |
| `train_selector.yaml` | same as scorer |
| `train_dual_network.yaml` | `scorer_checkpoint`, `selector_checkpoint`, `auxiliary_weight` |
| `run_selector_inference.yaml` | `checkpoint`, `question_id`, `top_k` |

## Troubleshooting

| Error | Fix |
|-------|-----|
| `torch not installed` | `pip install torch` |
| `DEEPSEEK_API_KEY not set` | Export the env var or use `--runner mock` for smoke tests |
| `caption_embeddings.npy not found` | Run embedding generation first |
| `Reward variance is zero` | Use `--runner deepseek` for real data, or add `allow_zero_variance_reward: true` to the scorer config |
| `load_state_dict` fails | Ensure the checkpoint was saved with `backend=torch` (torch.save, not JSON) |
| `Selector inference expects a fallback-format JSON checkpoint` | Point the smoke test at `outputs/selector_dual.pt` or `selector_fallback.json`, not `selector_torch.pt` |
| CUDA OOM | Reduce `epochs` or process fewer examples with `--max_examples` |
