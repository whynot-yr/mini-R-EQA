# Claude Code Global Project Rules

1. For multi-stage tasks, do not implement everything in one uncontrolled edit.
2. Read the current task spec from `tasks/current_task.md` or a user-provided task file.
3. For each stage:
   - inspect repo
   - plan
   - implement
   - run checks
   - review diff
   - invoke auditor agent
   - fix issues
   - write report
   - commit if checks pass
4. Never push unless explicitly asked.
5. Never run expensive training unless explicitly asked.
6. Never call paid APIs unless explicitly asked.
7. Stop only for destructive operations, credentials, paid API calls, or architecture ambiguity.
