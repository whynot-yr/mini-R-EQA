# PR-7 Report: Fix Candidate Reward Dataset

## Summary

Replaced lexical overlap frame ranking with cached SBERT cosine similarity. Added rich schema fields (retrieval_scores, retrieval_ranks, is_hard_negative) to the candidate reward JSONL. Added summary output and zero-variance reward warnings.

## Files Changed

| File | Change |
|------|--------|
| `mini_eqa/inference/candidate_generation.py` | SBERT ranking via `_rank_cached_sbert`; lexical kept as fallback; `rank` field added per frame |
| `mini_eqa/schema/selector_scorer.py` | `CandidateRewardRecord` extended with `frame_ids`, `retrieval_scores`, `retrieval_ranks`, `is_hard_negative`, `top_k`, `episode_id`, `candidate_id`, `selected_items`, `debug` |
| `mini_eqa/data_loading/selector_scorer_loader.py` | Added `resolve_embedding_paths()` and `resolve_embedding_cache_dir()` helpers |
| `mini_eqa/data_loading/__init__.py` | Exported new helpers |
| `mini_eqa/training/scorer_dataset.py` | Backward compat: reads `frame_ids` or `candidate_frames` |
| `mini_eqa/training/selector_pseudo_labels.py` | Same backward compat |
| `scripts/generate_candidate_reward_dataset.py` | Full rewrite: SBERT args, summary, hard-negative detection, zero-variance warning |

## New CLI Options (generate_candidate_reward_dataset.py)

| Flag | Description |
|------|-------------|
| `--embedding_cache_dir` | Path to SBERT embedding cache dir. If omitted, auto-detected from episode_dir. |
| `--embedding_model` | Override SBERT model name (default: read from cache metadata). |
| `--summary_output` | JSON summary output path (default: `reports/candidate_reward_summary.json`). |
| `--hard_negative_min_score` | Min cosine score to qualify as hard negative (default: 0.5). |
| `--hard_negative_max_reward` | Max reward to qualify as hard negative (default: 0.2). |
| `--config` | YAML config file (same pattern as train_scorer.py). |
| `--runner {mock,deepseek}` | Runner selection (restricted from free-form string). |

## Hard Negative Logic

A candidate is marked `is_hard_negative = True` if:
- `embedding_cache_dir` is active (SBERT scores are in [−1, 1])  
- mean retrieval score ≥ `hard_negative_min_score` (high similarity)
- reward ≤ `hard_negative_max_reward` (low answer quality)

When using lexical fallback, `is_hard_negative` is always `False` (lexical scores are unbounded).

## Checks Passed

- `python3 -m compileall mini_eqa scripts` ✓
- `python3 scripts/generate_candidate_reward_dataset.py --help` ✓
- Dry-run smoke (2 questions, mock, auto-detected SBERT cache): 14 rows, all new fields present ✓
- Backward compat: old JSONL (`candidate_frames`) still works with training builders ✓
- New JSONL (`frame_ids`): tested with scorer and selector training builders ✓

## Schema Change

Old field `candidate_frames` renamed to `frame_ids`. Training builders (`scorer_dataset.py`, `selector_pseudo_labels.py`) accept both field names.
