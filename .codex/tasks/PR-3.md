# PR-3: Train Scorer

Status: completed

## Goal

Train a scorer model that predicts candidate-set reward from question embedding and mean-pooled candidate frame embeddings, using `candidate_reward_dataset.jsonl` as supervision and `MSELoss` as the training objective.

## Scope

- Read `candidate_reward_dataset.jsonl`.
- Build training examples from:
  - question embedding
  - mean-pooled candidate frame embeddings
- Train `Scorer(q, candidate_set) -> reward`.
- Use `MSELoss(predicted_reward, real_reward)`.
- Save the trained checkpoint as `scorer.pt`.
- Record training loss.
- Support a small-data dry-run.

## Forbidden Changes

- Do not train the selector.
- Do not introduce complex RL.
- Do not add dual-network joint training.
- Do not require online DeepSeek calls for training-time validation.

## Expected Files

- Training script under `scripts/`, for example:
  `scripts/train_scorer.py`
- Supporting training logic under `mini_eqa/training/`, for example:
  - `mini_eqa/training/train_scorer.py`
  - `mini_eqa/training/scorer_dataset.py`
- Scorer model logic under `mini_eqa/scorer/`
- Checkpoint output path for `scorer.pt`
- Optional training log output under `reports/` or a designated artifacts directory
- Minimal docs update if new training usage needs explanation

## Implementation Steps

1. Implement loading for `candidate_reward_dataset.jsonl`.
2. Convert each record into scorer training features:
   - question embedding
   - mean-pooled candidate frame embeddings
3. Build or finalize the scorer forward path for reward prediction.
4. Implement a training loop using `MSELoss`.
5. Log train loss in a visible, machine-readable or human-readable form.
6. Save checkpoint output as `scorer.pt`.
7. Add a small-data dry-run command that can complete quickly for validation.

## Acceptance Criteria

- The scorer training script is runnable.
- A small-data dry-run is supported.
- The training flow saves `scorer.pt`.
- Training loss is recorded.
- The implementation does not train the selector.
- `python -m compileall .` passes.

## Test Commands

- `python -m compileall .`
- `pytest`
  Run only if the repo contains tests or new tests are added.
- `python scripts/train_scorer.py --help`
- A small dry-run training command, for example:
  `python scripts/train_scorer.py ... --max_examples <small_n> --epochs 1`

## Report Requirements

Write `.codex/reports/PR-3-report.md` with:

- Changed Files
- Implemented Features
- Commands Run
- Tests Passed/Failed
- Known Limitations
- Next PR Handoff
- Assumptions

The report must explicitly confirm that only the scorer was trained in PR-3 and that selector training was not introduced.
