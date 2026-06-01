---
name: project-architecture-analyzer
description: Analyze the current repository architecture, module boundaries, entrypoints, data flow, configs, and execution path. Use when the user asks to understand the current project structure, summarize architecture, trace how a feature moves through the codebase, identify extension points, or map docs/files/modules before making changes.
---

# Project Architecture Analyzer

Use this skill when the task is to explain or inspect the current codebase structure, not to design a new architecture from scratch.

## Workflow

1. Start from repository shape.
   - Use `rg --files` to map top-level areas.
   - Read `README.md` and architecture docs first if they exist.
   - Identify whether the repo is library-first, app-first, script-first, or research-prototype-first.

2. Identify execution entrypoints.
   - Find CLI scripts, package `__main__` files, runners, server starts, and config-driven wrappers.
   - Distinguish "real entrypoint" from helper scripts.

3. Map the main flow.
   - Trace the path from input data to final output.
   - Name the stages in order and the files that own each stage.
   - Call out where state is read, transformed, cached, and written.

4. Explain module boundaries.
   - Separate orchestration layers from core logic.
   - Separate adapters, preprocessing, model/runners, evaluation, and utilities.
   - Note registry patterns, plugin points, config surfaces, and hard-coded assumptions.

5. Summarize the architecture for the user.
   - Give a short top-down picture first.
   - Then list the key directories/files and their responsibilities.
   - Include concrete file references.

## Output Shape

Prefer this structure unless the user asks otherwise:

1. Repository type and primary purpose
2. Main execution flow
3. Key modules and responsibilities
4. Important entrypoints/configs
5. Extension points, risks, or coupling hotspots

## Quality Bar

- Do not invent flows; verify with files.
- Prefer exact file paths over vague names.
- Distinguish documented architecture from actual code if they differ.
- When the repo has existing architecture docs, reconcile them with current implementation.
- If the user wants changes afterward, use the architecture map to define the smallest safe edit surface.
