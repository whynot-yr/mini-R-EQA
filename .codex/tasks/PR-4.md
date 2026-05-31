# PR-4: Train Selector

Status: completed

## Goal

Train a simple selector model that scores frames per question using frame-level pseudo labels derived from the candidate reward dataset, without depending on the scorer during this stage.

## Scope

- Read `candidate_reward_dataset.jsonl`.
- Construct frame-level pseudo labels:
  - frames from high-reward candidates are positive
  - frames from low-reward candidates are negative
- Train `Selector(q, frame_i) -> frame_score_i`.
- Use `BCEWithLogitsLoss`.
- Save the checkpoint as `selector.pt`.
- Add selector inference or top-k selection smoke-test behavior.

## Forbidden Changes

- Do not depend on scorer outputs during this PR.
- Do not introduce dual-network training.
- Do not introduce complex RL.
- Do not replace the simple selector path with a more advanced training regime.

## Expected Files

- Training script under `scripts/`, for example:
  `scripts/train_selector.py`
- Supporting logic under `mini_eqa/training/`, for example:
  - `mini_eqa/training/train_selector.py`
  - `mini_eqa/training/selector_pseudo_labels.py`
- Selector model logic under `mini_eqa/selector/`
- Optional inference helper under `mini_eqa/inference/`
- Checkpoint output path for `selector.pt`
- Optional smoke-test script under `scripts/` or `tests/`
- Minimal docs update if selector training usage needs explanation

## Implementation Steps

1. Implement pseudo-label construction from `candidate_reward_dataset.jsonl`.
2. Define the rule that separates high-reward and low-reward candidates into positive and negative frame labels.
3. Build training examples for `Selector(q, frame_i) -> frame_score_i`.
4. Implement selector training using `BCEWithLogitsLoss`.
5. Save output checkpoint as `selector.pt`.
6. Add a selector inference or top-k selection smoke test using the trained checkpoint or a dry-run path.
7. Keep the selector path independent from scorer prediction in this PR.

## Acceptance Criteria

- The implementation can construct frame-level labels from `candidate_reward_dataset.jsonl`.
- `MLPSelector` can be trained.
- The training flow saves `selector.pt`.
- A selector inference or top-k selection smoke test exists.
- The implementation does not depend on scorer inference in this stage.
- `python -m compileall .` passes.

## Test Commands

- `python -m compileall .`
- `pytest`
  Run only if the repo contains tests or new tests are added.
- `python scripts/train_selector.py --help`
- A small dry-run training command, for example:
  `python scripts/train_selector.py ... --max_examples <small_n> --epochs 1`
- A smoke-test command for selector inference or top-k selection

## Report Requirements

Write `.codex/reports/PR-4-report.md` with:

- Changed Files
- Implemented Features
- Commands Run
- Tests Passed/Failed
- Known Limitations
- Next PR Handoff
- Assumptions

The report must explicitly document the pseudo-label rule and confirm that scorer-dependent training was not introduced in PR-4.
