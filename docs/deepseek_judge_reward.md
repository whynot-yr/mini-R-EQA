# DeepSeek Semantic Judge Reward

## Overview

The lexical reward (exact_match / token_f1) is too sparse for EQA. Many correct spatial or object answers score 0 because their wording differs from the gold answer. The DeepSeek judge calls a language model to semantically evaluate the predicted answer.

## DeepSeek Generator vs DeepSeek Judge — Key Distinction

| Role | Purpose | API calls |
|------|---------|-----------|
| **Generator** (`--runner deepseek`) | Given selected frame captions, generate `predicted_answer`. | 1 per candidate |
| **Judge** (`--reward_mode deepseek_judge`) | Given question + gold + predicted, score semantic correctness. | 1 per candidate |

These are **two separate API calls**. Running both at once roughly doubles cost.

## Reward Modes

| Mode | Formula | When to use |
|------|---------|------------|
| `local` | `max(exact_match, token_f1)` | Default. Free. Good for debugging. |
| `deepseek_judge` | judge score (0.0 / 0.5 / 1.0) | Real training signal. Requires API key. |
| `hybrid` | `max(judge_score, exact_match)` | Combines both; judge can override exact misses. |

### Why `contains_gold` is NOT used as final reward

`contains_gold` awards 1.0 whenever the gold token appears as a substring of the prediction. This rewards hallucinations like "I don't see a sofa, maybe a sofa" (gold = "sofa"). The judge is instructed to explicitly reject such cases:

> "If the predicted answer says the answer cannot be determined, and the gold answer is specific, score 0.0."

`contains_gold` remains in the breakdown for diagnostics only.

## Judge Score Values

| Score | Label | Meaning |
|-------|-------|---------|
| 1.0 | correct | Semantically equivalent to gold answer |
| 0.5 | partial | Partially correct or contains correct with extras |
| 0.0 | incorrect | Wrong, contradicts, or refuses to answer |
| 0.0 | judge_parse_error | Judge response could not be parsed |
| 0.0 | judge_api_error | API call failed after all retries |

## Expected Cost

- 1 candidate row ≈ 1 judge API call
- Typical judge prompt ≈ 250 tokens in, 50 tokens out
- For 7 candidate types × N questions: 7N judge calls
- At current DeepSeek pricing (~$0.27/M input tokens, $1.10/M output tokens), 1000 questions ≈ $0.20–$0.40

## Recommended Command Sequence

### Step A: Generate answers with local reward (cheap, for debugging)

```bash
python3 scripts/generate_candidate_reward_dataset.py \
  --episode_dir /path/to/episode \
  --runner deepseek \
  --model deepseek-chat \
  --top_k 3 \
  --embedding_cache_dir /path/to/episode/embeddings/sentence-transformers_all-MiniLM-L6-v2 \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --reward_mode local \
  --output outputs/candidate_reward_dataset_local.jsonl \
  --summary_output outputs/candidate_reward_summary_local.json \
  --limit 5
```

### Step B: Upgrade rewards with DeepSeek judge (post-hoc, resumable)

```bash
python3 scripts/judge_candidate_reward_dataset.py \
  --input outputs/candidate_reward_dataset_local.jsonl \
  --output outputs/candidate_reward_dataset_judged.jsonl \
  --reward_mode deepseek_judge \
  --judge_model deepseek-chat \
  --skip_existing
```

Step B reads existing `predicted_answer` values — no new generation calls. The `--skip_existing` flag enables resuming if the run is interrupted.

### Step C: Validate

```bash
python3 scripts/check_candidate_reward_dataset.py \
  --input outputs/candidate_reward_dataset_judged.jsonl
```

Exits nonzero if reward variance is zero, fields are missing, or the file is empty.

### Step D: Train scorer / selector / dual network

```bash
python3 scripts/train_scorer.py \
  --backend torch \
  --dataset_path outputs/candidate_reward_dataset_judged.jsonl \
  --episode_dir /path/to/episode \
  --embedding_model all-MiniLM-L6-v2

python3 scripts/train_selector.py \
  --backend torch \
  --dataset_path outputs/candidate_reward_dataset_judged.jsonl \
  --episode_dir /path/to/episode \
  --embedding_model all-MiniLM-L6-v2

python3 scripts/train_dual_network.py \
  --backend torch \
  --dataset_path outputs/candidate_reward_dataset_judged.jsonl \
  --episode_dir /path/to/episode \
  --scorer_checkpoint outputs/scorer_torch.pt \
  --selector_checkpoint outputs/selector_torch.pt
```

## Resuming Interrupted Judging

`judge_candidate_reward_dataset.py` writes output **line by line** with `flush()` after each row. If the process is killed:

1. The output file contains all rows written so far.
2. Re-run with `--skip_existing` — already-judged rows (matched by `question_id + candidate_type + sorted(frame_ids)`) are copied through unchanged.
3. Only unprocessed rows make new API calls.

## Running a Small-Scale Judge Smoke Test

```bash
# Requires DEEPSEEK_API_KEY
python3 scripts/judge_candidate_reward_dataset.py \
  --input reports/candidate_reward_dataset.jsonl \
  --output /tmp/judged_smoke.jsonl \
  --reward_mode deepseek_judge \
  --limit 2
```

If the API key is not available, use `--reward_mode local` to test the pipeline without API calls.

## Retry Behavior

The judge uses exponential backoff:
- `max_retries=5` (configurable via `--judge_max_retries`)
- `initial_sleep=3.0s` (configurable via `--judge_retry_initial_sleep`)
- multiplier: 2× per attempt (max ~96s before final attempt)
- small jitter to avoid thundering herd

Retried errors: SSL errors, connection errors, timeouts, HTTP 429/500/502/503/504.
Not retried: HTTP 401 (bad API key), HTTP 403 (forbidden).

## API Key

Preferred:

```bash
export DEEPSEEK_API_KEY=...
```

For backward compatibility, the code also accepts:

```bash
export deepseek_API_KEY=...
```

Use `DEEPSEEK_API_KEY` for all new setups and server configs.

## Backward Compatibility

Old JSONL files with only `reward`, `reward_breakdown`, `predicted_answer` (no judge fields) continue to work with all training scripts. Training code reads only the top-level `reward` field.

New JSONL files add `judge_label`, `judge_rationale`, `judge_score` inside `reward_breakdown` and `reward_mode` in `debug`. These are ignored by the training pipeline.
