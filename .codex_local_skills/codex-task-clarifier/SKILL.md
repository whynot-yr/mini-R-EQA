---
name: codex-task-clarifier
description: Turn rough change requests into a clear, executable Codex task prompt with scope, constraints, acceptance criteria, file targets, and validation steps. Use when the user gives scattered requirements, mixed goals, or partial ideas and wants them rewritten into a high-quality prompt for Codex.
---

# Codex Task Clarifier

Use this skill when the user's request is ambiguous, scattered, or too broad and needs to be converted into a practical execution prompt.

## Goal

Produce a prompt that a coding agent can execute with minimal follow-up.

## Workflow

1. Extract the raw intent.
   - Separate requested changes from background context.
   - Identify explicit constraints, implied constraints, and missing decisions.

2. Normalize the task.
   - Convert vague wording into concrete deliverables.
   - Split "analyze", "change", "verify", and "document" into separate expectations.
   - Remove duplicate or conflicting requirements unless they are important to preserve as questions.

3. Anchor the prompt in the codebase.
   - Mention likely files, modules, scripts, configs, or docs if they are known.
   - If exact files are unknown, name the subsystem instead of guessing.

4. Add execution constraints.
   - Preserve user preferences about tools, style, compatibility, or risk.
   - Include non-goals when the user is likely to over-scope the task.

5. Add a definition of done.
   - Expected behavior change
   - Required tests/checks
   - Required docs/config updates
   - Output artifacts if any

## Output Format

Return:

- `Task`: one short paragraph
- `Context`: only the facts Codex needs
- `Constraints`: flat bullet list
- `Acceptance Criteria`: flat bullet list
- `Suggested Validation`: flat bullet list

## Rewrite Rules

- Keep the final prompt concise and executable.
- Prefer imperative wording: "Update", "Add", "Refactor", "Verify".
- Avoid generic filler like "please help".
- If key ambiguities remain, list only the blocking questions.
- If the user explicitly wants a ready-to-paste prompt, output only the prompt body.
