# PR-3 Report

## Changed Files

- `README.md`
- `mini_eqa/training/feature_utils.py`
- `mini_eqa/training/scorer_dataset.py`
- `mini_eqa/training/train_scorer.py`
- `scripts/train_scorer.py`
- `reports/scorer.pt`
- `reports/scorer_train_metrics.json`
- `.codex/tasks/PR-3.md`

## Implemented Features

- Added deterministic question embedding utilities based on hashed token features.
- Added candidate mean-pooling utilities for frame embeddings.
- Added scorer dataset loading that joins:
  - `reports/candidate_reward_dataset.jsonl`
  - `data/sample_episode/captions.json`
  - `data/sample_episode/embeddings/.../caption_embeddings.npy`
- Added a runnable scorer training path using scalar reward supervision and MSE loss semantics.
- Added output saving for:
  - `reports/scorer.pt`
  - `reports/scorer_train_metrics.json`
- Added a dry-run-friendly scorer training CLI.

## Commands Run

- `python3 -m compileall mini_eqa scripts`
- `python3 scripts/train_scorer.py --help`
- `python3 scripts/train_scorer.py --dataset_path reports/candidate_reward_dataset.jsonl --episode_dir data/sample_episode --output reports/scorer.pt --metrics_output reports/scorer_train_metrics.json --epochs 2 --max_examples 6 --dry_run`
- `find reports -maxdepth 1 -name 'scorer.pt' -o -name 'scorer_train_metrics.json' | sort`
- `python3 - <<'PY' ... checkpoint and metrics inspection ... PY`

## Tests Passed/Failed

- Passed: `python3 -m compileall mini_eqa scripts`
- Passed: `python3 scripts/train_scorer.py --help`
- Passed: scorer dry-run training command
- Passed: checkpoint inspection confirmed `reports/scorer.pt` exists with:
  - `framework`
  - `question_dim`
  - `candidate_dim`
  - `input_dim`
  - `weights`
  - `bias`
- Passed: metrics inspection confirmed `reports/scorer_train_metrics.json` records:
  - epoch count
  - learning rate
  - number of examples
  - `train_loss_history`
  - `final_train_loss`
- Not run: `pytest`
  Reason: no repo test suite was present under `tests/` and PR-3 did not add pytest tests.

## Known Limitations

- The local Python 3 environment does not provide `torch`, so PR-3 uses a pure-Python fallback linear scorer trainer rather than a torch-backed MLP training loop.
- `scorer.pt` is a JSON checkpoint saved with a `.pt` suffix for workflow continuity; it is not a torch binary checkpoint in this environment.
- The dry-run validation used `--max_examples 6`, so the committed checkpoint is a small-sample artifact.

## Next PR Handoff

- PR-4 should train only the selector and must not depend on scorer inference.
- Reuse:
  - `reports/candidate_reward_dataset.jsonl`
  - `mini_eqa/training/feature_utils.py`
  - `data/sample_episode` embeddings and captions
- Keep selector pseudo-label generation explicit and documented.

## Assumptions

- `AGENTS.md` does not exist in the repository root.
- The user-specified task filename `.codex/tasks/PR-3-train-scorer.md` does not exist; this run mapped it to `.codex/tasks/PR-3.md`.
- Only scorer training was allowed in this PR, so selector training and any dual-network logic were intentionally excluded.
