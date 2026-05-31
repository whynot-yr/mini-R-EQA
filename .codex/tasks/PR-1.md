# PR-1: Framework Skeleton

Status: completed

## Goal

Create the initial selector-scorer framework skeleton for the EM-EQA / R-EQA iteration. Add clear package structure, importable skeleton classes, a unified schema layer, and basic data loading utilities for `captions.json` and `caption_embeddings.npy` without introducing training or online inference behavior.

## Scope

- Add new package directories for `selector`, `scorer`, `training`, and `inference`.
- Add importable skeleton implementations for `MLPSelector`, `MLPScorer`, and `MLPSupervisor`.
- Add a unified schema layer that can represent frame items, question items, candidate sets, and predictions.
- Add data loading utilities for `captions.json` and `caption_embeddings.npy`.
- Add at least one dry-run or smoke-test path for validating imports and basic loading behavior.
- Keep the implementation framework-only and minimal.

## Forbidden Changes

- Do not train any model.
- Do not call DeepSeek or any online runner.
- Do not generate `candidate_reward_dataset.jsonl`.
- Do not implement candidate reward generation logic.
- Do not implement complex dual-network training.
- Do not add full selector inference logic beyond minimal skeleton validation.

## Expected Files

- `mini_eqa/selector/__init__.py`
- `mini_eqa/selector/mlp_selector.py`
- `mini_eqa/scorer/__init__.py`
- `mini_eqa/scorer/mlp_scorer.py`
- `mini_eqa/scorer/mlp_supervisor.py`
- `mini_eqa/training/__init__.py`
- `mini_eqa/inference/__init__.py`
- `mini_eqa/schema/__init__.py`
- `mini_eqa/schema/selector_scorer.py`
- `mini_eqa/data_loading/__init__.py`
- `mini_eqa/data_loading/selector_scorer_loader.py`
- A small smoke-test or dry-run entrypoint under `scripts/` or `tests/`
- Minimal doc update in `README.md` or `docs/architecture.md` if package structure changes need explanation

## Implementation Steps

1. Add the new package directories and `__init__.py` files.
2. Define minimal, importable skeleton classes for `MLPSelector`, `MLPScorer`, and `MLPSupervisor`.
3. Define a unified schema for:
   - frame records
   - question records
   - candidate sets
   - prediction outputs
4. Implement a basic loader that reads:
   - `captions.json`
   - `caption_embeddings.npy`
5. Ensure loader outputs are compatible with the unified schema or clearly bridgeable to it.
6. Add a dry-run or smoke-test path that verifies imports and basic file loading.
7. Update lightweight documentation only if needed to explain the new structure.

## Acceptance Criteria

- New directory structure is clear and consistent with the existing `mini_eqa` package layout.
- `MLPSelector`, `MLPScorer`, and `MLPSupervisor` can be imported successfully.
- The schema can express:
  - frame
  - question
  - candidate set
  - prediction
- The data loader can read both `captions.json` and `caption_embeddings.npy`.
- At least one dry-run or smoke test exists and runs locally.
- `python -m compileall .` passes.

## Test Commands

- `python -m compileall .`
- `pytest`
  Run only if the repo contains tests or new tests are added.
- `python <dry_run_script>.py --help`
  or an equivalent smoke-test command for the new loader / skeleton path.

## Report Requirements

Write `.codex/reports/PR-1-report.md` with:

- Changed Files
- Implemented Features
- Commands Run
- Tests Passed/Failed
- Known Limitations
- Next PR Handoff
- Assumptions

The report must explicitly confirm that no training, no DeepSeek calls, and no candidate dataset generation were added in PR-1.
