# Official Evaluation

mini-R-EQA currently supports two layers of evaluation output:

## 1. Internal Prediction Report

The internal report keeps:

- retriever metadata
- runner metadata
- prompt text
- retrieved evidence
- predicted answer

This format is optimized for debugging and partial reproduction inside this repository.

## 2. OpenEQA-Compatible Export

The exporter converts the internal prediction report into a conservative OpenEQA-compatible JSON list.

This is a compatibility layer, not a reimplementation of the official evaluator.

## Important Distinction

- `evaluate_answers.py` provides simple answer sanity metrics.
- Those metrics are not final OpenEQA `LLM-Match`.
- Final R-EQA-style reproduction should use OpenEQA `evaluate-predictions.py` / `LLM-Match`.

## Model Setting Note

If you use DeepSeek as the answer model, that should be treated as an R-EQA pipeline variant.

It is not the same as the original LLaMA 3.1 70B answer-model setting.
