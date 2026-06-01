# Candidate Reward Dataset Integrity

## Required fields for multi-episode training

Every row in a candidate reward JSONL must have:

| Field | Requirement |
|-------|-------------|
| `episode_id` | Non-empty string. Required by `--prepared_root` training mode. |
| `frame_ids` | Non-empty list. Required by all training builders. |
| `predicted_answer` | Non-empty string. Required for reward computation. |
| `gold_answer` | Non-null string. Required for judge reward. |
| `reward` | Float. Top-level field used by all training scripts. |
| `is_hard_negative` | Bool. Must be consistent with `reward` and `retrieval_scores`. |

## episode_id

### Why it is required

The multi-episode training path (`--prepared_root`) resolves frame embeddings as:
```
prepared_root / episode_id / embeddings / <subdir> / caption_embeddings.npy
```
A missing or null `episode_id` causes an immediate `ValueError` with a clear message.

### How it is set

`generate_candidate_reward_dataset.py` always stamps `episode_id`:
- Default: `basename(episode_dir)` (e.g. `hm3d-v0__018-hm3d-dHwjuKfkRUR`)
- Override: `--episode_id <explicit_value>`

**Old datasets generated before this fix will have `episode_id=null`.** Use
`repair_candidate_reward_dataset.py` to backfill them.

## is_hard_negative consistency

### Definition

A candidate is a hard negative if:
1. Its mean retrieval score ≥ `hard_negative_min_score` (default 0.5)
2. Its final reward ≤ `hard_negative_max_reward` (default 0.2)
3. The judge label is not `"correct"`

### The inconsistency problem

When `judge_candidate_reward_dataset.py` upgrades rewards from local to judge scores,
`is_hard_negative` computed during generation (using the old local reward) becomes stale.

Example of the observed inconsistency:
```
reward=1.0, judge_label=correct, is_hard_negative=true   ← WRONG
```

**Always recompute `is_hard_negative` after judging.** The judge script does this
by default (`--recompute_hard_negatives` is on by default).

## Full training checklist

Before running `--prepared_root` training on a judged dataset:

### Step 1: Check the dataset
```bash
python3 scripts/check_candidate_reward_dataset.py \
  --input outputs/candidate_reward_dataset_judged.jsonl \
  --require_episode_id \
  --hard_negative_max_reward 0.2
```
This will fail nonzero if episode_id is missing or hard negatives are inconsistent.

### Step 2: Repair if needed
```bash
# Single file — infer episode_id from filename stem
python3 scripts/repair_candidate_reward_dataset.py \
  --input outputs/candidate_reward_dataset_local.jsonl \
  --output outputs/candidate_reward_dataset_local.fixed.jsonl

# Directory of per-episode files — episode_id inferred from each filename
python3 scripts/repair_candidate_reward_dataset.py \
  --input outputs/per_episode/ \
  --output outputs/candidate_reward_dataset_merged.jsonl
```

### Step 3: Re-judge with hard negative recomputation (if using post-hoc judge)
```bash
python3 scripts/judge_candidate_reward_dataset.py \
  --input outputs/candidate_reward_dataset_local.fixed.jsonl \
  --output outputs/candidate_reward_dataset_judged.jsonl \
  --reward_mode deepseek_judge \
  --recompute_hard_negatives \
  --skip_existing
```
The `--recompute_hard_negatives` flag is on by default — hard negatives are always
consistent with the final judge reward.

### Step 4: Validate again
```bash
python3 scripts/check_candidate_reward_dataset.py \
  --input outputs/candidate_reward_dataset_judged.jsonl \
  --require_episode_id \
  --hard_negative_max_reward 0.2
```
All checks must pass before multi-episode training.

### Step 5: Train
```bash
python3 scripts/train_scorer.py \
  --prepared_root /root/AI/mini-R-EQA/data/openeqa_prepared_hf \
  --dataset_path outputs/candidate_reward_dataset_judged.jsonl \
  --backend torch

python3 scripts/train_selector.py \
  --prepared_root /root/AI/mini-R-EQA/data/openeqa_prepared_hf \
  --dataset_path outputs/candidate_reward_dataset_judged.jsonl \
  --backend torch
```

## Common errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `prepared_root mode requires every JSONL row to contain a non-empty episode_id` | Old dataset without episode_id | Run `repair_candidate_reward_dataset.py` |
| `is_hard_negative=true but reward=1.0` | Stale hard negative after judge | Re-judge with `--recompute_hard_negatives` |
| `first JSONL row is missing episode_id; cannot auto-resolve embedding model` | No episode_id for SBERT model detection | Run repair, or pass `--embedding_model` explicitly |
