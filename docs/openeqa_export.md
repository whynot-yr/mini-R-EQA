# OpenEQA Export

mini-R-EQA currently produces an internal prediction report that is optimized for debugging and iteration inside this repository.

## Internal Report vs. OpenEQA-Compatible Export

The internal prediction report keeps:

- retrieval settings
- retrieved evidence
- prompt text
- predicted answer
- optional extra metadata copied from `questions.json`

The OpenEQA-compatible export is a compatibility layer. It converts the internal report into a conservative JSON list format that is easier to adapt to `evaluate-predictions.py`.

Each exported item currently contains:

- `question_id`
- `question`
- `answer`
- `prediction`
- `episode_history`
- `scene_id`

## Current Scope

- This exporter does not modify the internal prediction report.
- This exporter does not implement GPT-4 LLM-Match.
- This exporter does not claim final official submission compatibility yet.

## Debug Fields

`--include_debug_fields` can add:

- `retrieved`
- `prompt`
- `retriever`
- `runner`
- `model`

These fields are useful for dry-run inspection, but they should not be treated as the final official submission format.

## Next Step

The next validation step is a dry-run against the real OpenEQA `evaluate-predictions.py` script to confirm exact field and file expectations.
