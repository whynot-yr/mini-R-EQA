# PR-5: Dual-Network Training

Status: completed

## Goal

Implement the first dual-network training stage where the scorer learns set reward and the selector learns to prefer frame sets associated with higher scorer value, using either a staged pipeline or a simple alternating training strategy without introducing complex RL or GRPO.

## Scope

- Load `scorer.pt` and `selector.pt`.
- Implement a dual-training script.
- Support either:
  - train scorer first, then train selector
  - simple alternating training
- Add selector loss components:
  - pseudo-label BCE loss
  - scorer predicted reward auxiliary loss
- Output selected frames.
- Record scorer reward auxiliary signal.
- Save a new selector checkpoint.
- Add clear limitation documentation for this first dual-network version.

## Forbidden Changes

- Do not introduce complex RL.
- Do not introduce GRPO.
- Do not replace the implementation with a full policy-optimization framework.
- Do not depend on an unbounded online training loop.

## Expected Files

- Dual-training script under `scripts/`, for example:
  `scripts/train_dual_network.py`
- Supporting logic under `mini_eqa/training/`, for example:
  - `mini_eqa/training/train_dual_network.py`
  - `mini_eqa/training/dual_objectives.py`
- Any selector-scoring bridge logic needed under `mini_eqa/inference/` or `mini_eqa/training/`
- Updated checkpoint output path for the new selector checkpoint
- Limitation documentation under `docs/`, for example:
  `docs/dual_network_limitations.md`
- Minimal README or workflow doc update if command usage changes

## Implementation Steps

1. Implement loading for existing `scorer.pt` and `selector.pt`.
2. Choose one valid first-version training mode:
   - staged: scorer first, selector second
   - simple alternating training
3. Implement selector optimization using:
   - pseudo-label BCE loss
   - scorer predicted reward auxiliary loss
4. Add the ability to output selected frames during validation or smoke testing.
5. Record the scorer auxiliary reward signal in logs or reports.
6. Save a new selector checkpoint.
7. Add explicit limitation documentation for what this dual-network approach does not yet cover.

## Acceptance Criteria

- A dual-training script exists.
- The implementation can load `scorer.pt` and `selector.pt`.
- The pipeline can output selected frames.
- The scorer reward auxiliary signal is recorded.
- A new selector checkpoint is saved.
- Limitation documentation is present and explicit.
- `python -m compileall .` passes.

## Test Commands

- `python -m compileall .`
- `pytest`
  Run only if the repo contains tests or new tests are added.
- `python scripts/train_dual_network.py --help`
- A small dry-run dual-training command, for example:
  `python scripts/train_dual_network.py ... --max_examples <small_n> --epochs 1`
- A smoke-test command that verifies selected-frame output

## Report Requirements

Write `.codex/reports/PR-5-report.md` with:

- Changed Files
- Implemented Features
- Commands Run
- Tests Passed/Failed
- Known Limitations
- Next PR Handoff
- Assumptions

The report must explicitly document the chosen training strategy, the role of the scorer auxiliary signal, and confirm that no complex RL or GRPO was introduced.
