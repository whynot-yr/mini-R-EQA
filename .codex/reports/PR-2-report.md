# PR-2 Report

## Changed Files

- `README.md`
- `mini_eqa/evaluation/reward_utils.py`
- `mini_eqa/inference/candidate_generation.py`
- `mini_eqa/schema/__init__.py`
- `mini_eqa/schema/selector_scorer.py`
- `mini_eqa/utils/io_utils.py`
- `scripts/generate_candidate_reward_dataset.py`
- `reports/candidate_reward_dataset.jsonl`
- `.codex/tasks/PR-2.md`

## Implemented Features

- Added candidate-set generation for seven candidate types:
  - `reqa_top3`
  - `reqa_top6_random3`
  - `uniform3`
  - `random3`
  - `reqa_perturbed3`
  - `middle_rank_random3`
  - `low_rank_random3`
- Added a deterministic reward computation utility based on:
  - exact match
  - gold-string containment
  - token F1
- Defined candidate reward records in the selector-scorer schema.
- Added JSONL read/write helpers.
- Added an offline-capable dataset generation script with `mock` runner support.
- Generated `reports/candidate_reward_dataset.jsonl` from sample episode data as a dry-run artifact.

## Commands Run

- `python3 -m compileall mini_eqa scripts`
- `python3 scripts/generate_candidate_reward_dataset.py --help`
- `python3 scripts/generate_candidate_reward_dataset.py --episode_dir data/sample_episode --runner mock --output reports/candidate_reward_dataset.jsonl --limit 2 --dry_run`
- `find . -name 'candidate_reward_dataset.jsonl' | sort`
- `python3 - <<'PY' ... schema inspection for reports/candidate_reward_dataset.jsonl ... PY`

## Tests Passed/Failed

- Passed: `python3 -m compileall mini_eqa scripts`
- Passed: `python3 scripts/generate_candidate_reward_dataset.py --help`
- Passed: `python3 scripts/generate_candidate_reward_dataset.py --episode_dir data/sample_episode --runner mock --output reports/candidate_reward_dataset.jsonl --limit 2 --dry_run`
- Passed: schema inspection confirmed JSONL rows include:
  - `question_id`
  - `question`
  - `gold_answer`
  - `candidate_frames`
  - `candidate_type`
  - `predicted_answer`
  - `reward`
- Not run: `pytest`
  Reason: no repo test suite was present under `tests/` and PR-2 did not add pytest tests.

## Known Limitations

- The current R-EQA-style ranking path uses a deterministic lexical-overlap fallback so the dataset can be generated in the dependency-light local Python 3 environment.
- `DeepSeek` is supported only through the runner interface; PR-2 validation was intentionally performed with `mock`.
- Reward is a simple supervised proxy using the maximum of exact match, contains-gold, and token F1. It is stable and documented, but not a semantic judge.
- The dry-run artifact uses `--limit 2`, so the committed dataset file is a sample artifact rather than a full-scale dataset.

## Next PR Handoff

- PR-3 should consume `reports/candidate_reward_dataset.jsonl` as scorer supervision input.
- Reuse:
  - `mini_eqa/schema/selector_scorer.py`
  - `mini_eqa/evaluation/reward_utils.py`
  - `scripts/generate_candidate_reward_dataset.py`
- Keep PR-3 limited to scorer training only.

## Assumptions

- `AGENTS.md` does not exist in the repository root.
- The user-specified task filename `.codex/tasks/PR-2-candidate-reward-dataset.md` does not exist; this run mapped it to `.codex/tasks/PR-2.md`.
- The local Python 3 environment lacks `numpy`, `torch`, and related ML dependencies, so PR-2 used an offline-safe lexical candidate ranking path for validation.
- No training of any kind was introduced in PR-2.
