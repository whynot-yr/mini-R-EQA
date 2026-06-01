# PR-4 Report

## Changed Files

- `README.md`
- `mini_eqa/inference/selector_inference.py`
- `mini_eqa/training/selector_pseudo_labels.py`
- `mini_eqa/training/train_selector.py`
- `scripts/train_selector.py`
- `scripts/selector_inference_smoke_test.py`
- `reports/selector.pt`
- `reports/selector_train_metrics.json`
- `reports/selector_pseudo_label_summary.json`
- `.codex/tasks/PR-4.md`

## Implemented Features

- Added frame-level pseudo-label generation from `candidate_reward_dataset.jsonl`.
- Implemented the pseudo-label rule:
  - frames appearing in highest-reward candidate sets for a question are positive
  - frames appearing in lowest-reward candidate sets and not already positive are negative
- Added selector training examples built from:
  - hashed question embedding
  - frame embedding
- Added selector training with BCE-with-logits semantics.
- Added selector checkpoint and metrics outputs.
- Added selector top-k inference smoke test that loads `selector.pt` and returns ranked frames without any scorer dependency.

## Commands Run

- `python3 -m compileall mini_eqa scripts`
- `python3 scripts/train_selector.py --help`
- `python3 scripts/train_selector.py --dataset_path reports/candidate_reward_dataset.jsonl --episode_dir data/sample_episode --output reports/selector.pt --metrics_output reports/selector_train_metrics.json --summary_output reports/selector_pseudo_label_summary.json --epochs 2 --max_examples 8 --dry_run`
- `python3 scripts/selector_inference_smoke_test.py --checkpoint reports/selector.pt --episode_dir data/sample_episode --question_id q1 --top_k 3`
- `find reports -maxdepth 1 -name 'selector.pt' -o -name 'selector_train_metrics.json' -o -name 'selector_pseudo_label_summary.json' | sort`
- `python3 - <<'PY' ... selector artifact inspection ... PY`

## Tests Passed/Failed

- Passed: `python3 -m compileall mini_eqa scripts`
- Passed: `python3 scripts/train_selector.py --help`
- Passed: selector dry-run training command
- Passed: selector inference smoke test returning top-k selected frames
- Passed: artifact inspection confirmed:
  - `reports/selector.pt`
  - `reports/selector_train_metrics.json`
  - `reports/selector_pseudo_label_summary.json`
- Not run: `pytest`
  Reason: no repo test suite was present under `tests/` and PR-4 did not add pytest tests.

## Known Limitations

- The local Python 3 environment does not provide `torch`, so PR-4 uses a pure-Python fallback linear selector trainer rather than torch-backed MLP optimization.
- `selector.pt` is a JSON checkpoint saved with a `.pt` suffix for workflow continuity; it is not a torch binary checkpoint in this environment.
- The pseudo-label rule is intentionally simple and question-local. It does not yet incorporate scorer guidance or confidence weighting.
- The committed selector checkpoint is a small-sample dry-run artifact built with `--max_examples 8`.

## Next PR Handoff

- PR-5 can now load:
  - `reports/scorer.pt`
  - `reports/selector.pt`
- Reuse:
  - `mini_eqa/training/train_scorer.py`
  - `mini_eqa/training/train_selector.py`
  - `mini_eqa/inference/selector_inference.py`
- PR-5 may add scorer auxiliary guidance, but must avoid complex RL and GRPO.

## Assumptions

- `AGENTS.md` does not exist in the repository root.
- The user-specified task filename `.codex/tasks/PR-4-train-selector.md` does not exist; this run mapped it to `.codex/tasks/PR-4.md`.
- Scorer-dependent training was explicitly excluded from PR-4, so selector training was kept independent from scorer outputs.
