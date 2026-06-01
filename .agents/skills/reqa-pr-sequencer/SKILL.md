---
name: reqa-pr-sequencer
description: Execute the EM-EQA / R-EQA selector-scorer project iteration from repo-local PR task files in .codex/tasks, strictly one PR at a time from PR-1 through PR-5. Use when the user wants Codex to implement the next PR in sequence, prepare the plan for a specific PR, or continue the selector-scorer roadmap with required reports, tests, and commits.
---

# REQA PR Sequencer

Use this skill to execute the repo's PR-based iteration workflow from `.codex/tasks/`.

## Scope

- The source of truth for work items is `.codex/tasks/PR-1.md` through `.codex/tasks/PR-5.md`.
- Execute exactly one PR task per run.
- Do not start the next PR in the same run unless the user explicitly asks for it in a later turn.

## Required Start-Up Reads

Before starting any PR, read:

1. `AGENTS.md` if it exists
2. `README.md`
3. The current PR task file in `.codex/tasks/`
4. Only the code and docs directly relevant to that PR

If `AGENTS.md` does not exist, continue with the minimal reasonable implementation and record that assumption in the PR report.

## PR Selection Rules

- Prefer the lowest-numbered unfinished PR.
- If the user names a specific PR, execute only that PR.
- Treat missing checklists, missing detail, or partial ambiguity as non-blocking unless the request is unsafe or contradictory.
- If requirements are unclear, make the smallest reasonable implementation and record the assumptions in the report instead of stopping.

## Mandatory Pre-Implementation Output

Before editing any code for the selected PR, output all of the following:

1. `Implementation Checklist`
2. `Expected Files To Create/Modify`
3. `Acceptance Criteria`
4. `Test Plan`

Keep this output concrete and PR-specific.

## Execution Rules

- Modify only the files needed for the current PR.
- Do not mix work from multiple PRs.
- Preserve unrelated user changes.
- Keep the implementation minimal, incremental, and aligned with the current repository structure unless the PR explicitly requires a structural refactor.

## Explicit Prohibitions

- In `PR-1`, do not generate a candidate dataset.
- In `PR-1`, do not write training logic.
- In `PR-2`, do not train a model.
- In `PR-3` or `PR-4`, do not introduce complex RL.
- In `PR-5`, do not jump directly to complex GRPO/RL.

If a task file appears to ask for prohibited work at that stage, scale the implementation back to the nearest valid minimal slice and explain the adjustment in the report.

## Validation Requirements

After implementing the PR, run reasonable validation. Prefer:

- `python -m compileall .`
- `pytest` if the repo already contains tests
- Relevant script checks such as `--help`, `dry-run`, or small local smoke runs

If some validation is not possible in the current environment, state that clearly in the report.

## Reporting

After each completed PR, create `.codex/reports/PR-X-report.md`.

The report must include:

- `Changed Files`
- `Implemented Features`
- `Commands Run`
- `Tests Passed/Failed`
- `Known Limitations`
- `Next PR Handoff`

Also include a short `Assumptions` section whenever the task file or repo context was incomplete.

## Commit Rule

If the PR implementation is complete and tests pass, create a git commit with this exact format:

`PR-X: concise description`

Do not create the commit if validation failed.

## Suggested Run Shape

For each PR execution, follow this order:

1. Read required context
2. Announce the selected PR
3. Output the four mandatory planning sections
4. Implement the code changes
5. Run validation
6. Write `.codex/reports/PR-X-report.md`
7. Commit if validation passed

## When Not To Use

- Do not use this skill for broad multi-PR roadmap planning in a single execution.
- Do not use this skill when the user only wants architecture analysis without implementation.
