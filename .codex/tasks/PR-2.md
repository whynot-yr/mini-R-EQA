# PR-2: Candidate Reward Dataset Generation

Status: completed

## Goal

Implement candidate reward dataset generation for each question by producing multiple candidate frame sets, running a selected answer runner on each set, computing a reward against the gold answer, and saving the results as stable JSONL records.

## Scope

- Generate 5 to 8 candidate sets per question.
- Support candidate generation strategies derived from existing retrieval outputs and random sampling.
- Support answer generation through either a DeepSeek runner or a mock runner.
- Compare predicted answers with gold answers to compute reward.
- Save outputs to `candidate_reward_dataset.jsonl`.
- Default to `top_k = 3` unless the task implementation clearly needs an override parameter.
- Include a dry-run example that can use `mock` without requiring online access.

## Forbidden Changes

- Do not train any model.
- Do not train the scorer.
- Do not train the selector.
- Do not introduce dual-network training behavior.
- Do not make DeepSeek mandatory for validation.

## Expected Files

- A dataset-generation script under `scripts/` such as:
  `scripts/generate_candidate_reward_dataset.py`
- Supporting logic under `mini_eqa/` such as:
  - `mini_eqa/inference/candidate_generation.py`
  - `mini_eqa/schema/selector_scorer.py`
  - `mini_eqa/utils/` helpers as needed
- Optional reward utility module, for example:
  `mini_eqa/evaluation/reward_utils.py`
- Optional small fixture or dry-run sample output path under `reports/` or `data/`
- Minimal docs update if command usage needs explanation

## Implementation Steps

1. Define a stable JSONL row schema for candidate reward records.
2. Implement candidate set generation for the following candidate types:
   - R-EQA top-3
   - R-EQA top-6 random sample of 3
   - uniform 3
   - random 3
   - R-EQA perturbed 3
   - middle-rank random 3
   - low-rank random 3
3. For each question, generate 5 to 8 candidate sets using the supported strategies.
4. Run either:
   - DeepSeek runner
   - mock runner
5. Compare predicted answer with gold answer and compute reward using a documented deterministic rule.
6. Save one JSONL record per candidate set to `candidate_reward_dataset.jsonl`.
7. Add a dry-run path using `mock` so the flow can be exercised offline.

## Acceptance Criteria

- The implementation supports `mock` runner and does not require DeepSeek for local validation.
- The JSONL schema is stable and documented in code or task-facing docs.
- Each JSONL row includes:
  - `question_id`
  - `question`
  - `gold_answer`
  - `candidate_frames`
  - `candidate_type`
  - `predicted_answer`
  - `reward`
- A dry-run example exists.
- No model training is added.
- `python -m compileall .` passes.

## Test Commands

- `python -m compileall .`
- `pytest`
  Run only if the repo contains tests or new tests are added.
- `python scripts/generate_candidate_reward_dataset.py --help`
- A dry-run command using `mock`, for example:
  `python scripts/generate_candidate_reward_dataset.py ... --runner mock --dry_run`

## Report Requirements

Write `.codex/reports/PR-2-report.md` with:

- Changed Files
- Implemented Features
- Commands Run
- Tests Passed/Failed
- Known Limitations
- Next PR Handoff
- Assumptions

The report must explicitly document the reward rule, the supported candidate types, and confirm that no model training was performed in PR-2.
