# DeepSeek Judge Reward Audit

Date: 2026-06-01
Repo: `mini-R-EQA`
Scope:
- `mini_eqa/evaluation/llm_judge_reward.py`
- `mini_eqa/evaluation/reward_utils.py`
- `scripts/generate_candidate_reward_dataset.py`
- `scripts/judge_candidate_reward_dataset.py`
- `scripts/check_candidate_reward_dataset.py`
- `docs/deepseek_judge_reward.md`

## Verdict

`PASS WITH MINOR RISKS`

The DeepSeek semantic judge pipeline is structurally correct and training-compatible after a small set of blocking fixes:
- judge labels are now normalized to a valid set when the model returns malformed labels
- reward computation in judge modes no longer silently accepts a missing `judge_result` when a `gold_answer` exists
- dataset validation now reports judge parse-error count consistently
- docs now explicitly document the backward-compatible `deepseek_API_KEY` alias while preferring `DEEPSEEK_API_KEY`

No push was performed.

## Pass/Fail Table

| Area | Status | Notes |
|---|---|---|
| `judge_answer_with_deepseek()` uses `question`, `gold_answer`, `predicted_answer` | Pass | Prompt contains all three fields and is clearly framed as a semantic judge, not a generator |
| Judge path is distinct from answer generation | Pass | Generation uses runner path; judging uses `llm_judge_reward.py` and a different prompt/API call |
| Judge output is JSON-serializable with `score`, `label`, `rationale`, `raw_response` | Pass | Returned dict is plain JSON-safe data |
| Judge scores are clamped to `1.0 / 0.5 / 0.0` | Pass | Parsing clamps score bands before returning |
| Parse failures return safe `judge_parse_error` with preserved raw response | Pass | Fallback record preserves `raw_response` |
| Judge labels are validated/normalized | Pass after fix | Invalid labels now map from the clamped score instead of leaking arbitrary model text |
| `reward_mode=local` uses only `max(exact_match, token_f1)` | Pass | `contains_gold` is diagnostic only |
| `reward_mode=deepseek_judge` uses judge score as final reward | Pass | Final reward is the normalized judge score |
| `reward_mode=hybrid` uses `max(judge_score, exact_match)` | Pass | `contains_gold` is not used in final reward |
| `reward_breakdown` contains required fields | Pass | Includes exact/local/judge/final mode fields |
| `generate_candidate_reward_dataset.py` supports all reward modes and judge args | Pass | CLI and runtime wiring present |
| Judge mode runs after `predicted_answer` generation | Pass | Judging happens after `run_candidate_answer(...)` |
| Default reward mode remains local | Pass | Default is `local`, avoiding accidental double API cost |
| Generator warns clearly on judge mode | Pass | Warning printed to `stderr` |
| `judge_candidate_reward_dataset.py` reuses existing `predicted_answer` and does not regenerate | Pass | Post-hoc judge only |
| Post-hoc judge preserves original fields | Pass | Uses `updated = dict(row)` |
| Post-hoc judge writes incrementally and flushes each row | Pass | Output is line-by-line with `flush()` |
| Post-hoc judge supports `--skip_existing` with stable key | Pass | Key uses `question_id + candidate_type + sorted(frame_ids)` plus optional `candidate_id` |
| Dataset check catches missing file / empty file / missing core fields / zero variance | Pass | Exits nonzero on all required conditions unless zero variance is explicitly allowed |
| Dataset check reports required aggregate stats | Pass after fix | Judge parse error count is now always reported |
| Retry/backoff handles transient judge failures | Pass | Retries 429/5xx, connection, SSL, timeout with exponential backoff |
| 401/403 fail immediately | Pass | Fatal auth/permission paths are not retried |
| API key is read from `DEEPSEEK_API_KEY` | Pass | Preferred env var is used |
| `deepseek_API_KEY` alias is accepted and documented | Pass after fix | Alias remains backward-compatible; docs now mention it |
| Training scripts depend only on top-level `reward` | Pass | Scorer/selector/dual training all read top-level `reward` and frame id fields only |
| Old local-reward JSONL stays loadable | Pass | Training supports both `frame_ids` and legacy `candidate_frames` |
| Judged JSONL stays loadable | Pass | Additional `reward_breakdown` fields are ignored by training |
| Live DeepSeek API smoke test | Not run | No `DEEPSEEK_API_KEY` or `deepseek_API_KEY` was present locally |

## Blocking Issues Fixed

1. Invalid judge labels could leak through as arbitrary text.
   - Fix: normalized invalid labels to `correct`, `partial`, or `incorrect` based on the clamped score in `mini_eqa/evaluation/llm_judge_reward.py`.

2. Judge reward modes could silently produce a zeroed reward if `judge_result` was accidentally omitted despite a valid `gold_answer`.
   - Fix: `mini_eqa/evaluation/reward_utils.py` now raises a `ValueError` in `deepseek_judge` / `hybrid` mode when `gold_answer` exists but `judge_result` is missing.

3. Reward utils accepted arbitrary judge scores and labels from callers.
   - Fix: judge scores are now normalized again inside `compute_reward_breakdown(...)`, so the top-level reward remains in the allowed band even if a caller passes malformed judge data.

4. Dataset checker did not always print judge parse-error count.
   - Fix: `scripts/check_candidate_reward_dataset.py` now reports judge label distribution and parse-error count consistently, including zero.

## Non-Blocking Risks

- No live API smoke test was run in this environment because no DeepSeek API key was available.
- `judge_answer_with_deepseek()` uses a simple JSON extraction fallback (`{...}` block). This is adequate for current prompts but could still be brittle if future models emit heavily nested JSON or unusual formatting.
- `judge_api_error` is intentionally represented in wrapper scripts rather than returned directly from `llm_judge_reward.py`. This separation is reasonable, but it means API failures and parse failures are produced at different layers.

## Commands Run

Required checks:

```bash
python3 -m compileall mini_eqa scripts
python3 scripts/generate_candidate_reward_dataset.py --help
python3 scripts/judge_candidate_reward_dataset.py --help
python3 scripts/check_candidate_reward_dataset.py --help
```

Additional local verification:

```bash
python3 -c "import os; print('yes' if (os.getenv('DEEPSEEK_API_KEY') or os.getenv('deepseek_API_KEY')) else 'no')"
python3 -c "from mini_eqa.evaluation.llm_judge_reward import _parse_judge_response; print(_parse_judge_response('{\"score\": 0.8, \"label\": \"maybe\", \"rationale\": \"ok\"}')); print(_parse_judge_response('not json'))"
python3 -c "from mini_eqa.evaluation.reward_utils import compute_reward_breakdown; print(compute_reward_breakdown('gray','gray',reward_mode='local')); print(compute_reward_breakdown('gray','gray',reward_mode='hybrid', judge_result={'score':0.6,'label':'weird','rationale':'x'}))"
python3 -c "from mini_eqa.evaluation.reward_utils import compute_reward_breakdown; import sys; \
try: compute_reward_breakdown('gray','gray',reward_mode='deepseek_judge') \
except Exception as e: print(type(e).__name__, str(e))"
python3 scripts/judge_candidate_reward_dataset.py --input reports/candidate_reward_dataset.jsonl --output /tmp/deepseek_judge_reward_dryrun.jsonl --reward_mode local --limit 2 --dry_run
python3 scripts/check_candidate_reward_dataset.py --input reports/candidate_reward_dataset.jsonl --allow_zero_variance
```

## API Smoke Status

DeepSeek API smoke test was **not run**.

Reason:
- `DEEPSEEK_API_KEY` was not set
- `deepseek_API_KEY` was not set

## Recommended Server Commands

1. Generate candidate answers with local reward

```bash
export DEEPSEEK_API_KEY=...

python3 scripts/generate_candidate_reward_dataset.py \
  --episode_dir /path/to/episode \
  --runner deepseek \
  --model deepseek-chat \
  --top_k 3 \
  --embedding_cache_dir /path/to/episode/embeddings/sentence-transformers_all-MiniLM-L6-v2 \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --reward_mode local \
  --output outputs/candidate_reward_dataset_local.jsonl \
  --summary_output outputs/candidate_reward_summary_local.json
```

2. Post-hoc judge with DeepSeek

```bash
export DEEPSEEK_API_KEY=...

python3 scripts/judge_candidate_reward_dataset.py \
  --input outputs/candidate_reward_dataset_local.jsonl \
  --output outputs/candidate_reward_dataset_judged.jsonl \
  --reward_mode deepseek_judge \
  --judge_model deepseek-chat \
  --skip_existing
```

3. Check judged dataset

```bash
python3 scripts/check_candidate_reward_dataset.py \
  --input outputs/candidate_reward_dataset_judged.jsonl
```

4. Train scorer / selector / dual-network from judged dataset

```bash
python3 scripts/train_scorer.py \
  --backend torch \
  --dataset_path outputs/candidate_reward_dataset_judged.jsonl \
  --episode_dir /path/to/episode \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2

python3 scripts/train_selector.py \
  --backend torch \
  --dataset_path outputs/candidate_reward_dataset_judged.jsonl \
  --episode_dir /path/to/episode \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2

python3 scripts/train_dual_network.py \
  --backend torch \
  --dataset_path outputs/candidate_reward_dataset_judged.jsonl \
  --episode_dir /path/to/episode \
  --scorer_checkpoint outputs/scorer_torch.pt \
  --selector_checkpoint outputs/selector_torch.pt \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
```
