# DeepSeek LLM-Match Evaluation

This workflow adds a DeepSeek-based OpenEQA-style judge for exported results.

## Important Scope

- This is not the official OpenEQA GPT-4 `LLM-Match` score.
- The official OpenEQA score still requires the official OpenEQA evaluator and its GPT-4-based judging setup.
- This repo workflow is intended for aligned internal comparison across variants such as `reqa` vs. `uniform`.

## Inputs

### Dataset file

Expected dataset items:

```json
{
  "question_id": "...",
  "question": "...",
  "answer": "...",
  "extra_answers": ["..."]
}
```

- `answer` is the gold answer.
- `extra_answers` is optional and is treated as additional valid references.

### Official-style results file

Expected result items:

```json
{
  "question_id": "...",
  "answer": "model prediction"
}
```

- In this file, `answer` must be the model prediction.
- Gold answers are read from the dataset file, not from the results file.

## Export Step

Use the dedicated exporter to convert per-episode prediction reports into an official-style results list:

```bash
python scripts/export_official_results.py \
  --pred_dir /mnt/storage/mini_reqa_outputs/hf_full_deepseek_chat \
  --method reqa \
  --dataset data/open_eqa_hf/qa/open-eqa-hf.json \
  --output /mnt/storage/mini_reqa_outputs/official_eval_inputs/reqa_deepseek_chat_results.json
```

Supported methods:

- `reqa`
- `uniform`

The exporter validates exact `question_id` coverage against the dataset:

- every dataset `question_id` must appear exactly once
- duplicates are rejected
- extra IDs are rejected
- output order follows dataset order

## Evaluation Step

Run the DeepSeek judge:

```bash
python scripts/evaluate_with_deepseek_llm_match.py \
  --dataset data/open_eqa_hf/qa/open-eqa-hf.json \
  --results /mnt/storage/mini_reqa_outputs/official_eval_inputs/reqa_deepseek_chat_results.json \
  --output /mnt/storage/mini_reqa_outputs/deepseek_llm_match/reqa_deepseek_llm_match.json \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --resume
```

The judge uses a 1-5 scale:

- `5`: fully correct or semantically equivalent
- `4`: mostly correct with minor omission
- `3`: partially correct
- `2`: mostly incorrect but related
- `1`: incorrect, irrelevant, contradictory, or no answer

Scores are mapped to `0-100` with:

```text
100 * (score - 1) / 4
```

## Resume Behavior

`--resume` allows long API runs to continue from an existing output JSON.

- completed question IDs are skipped
- failed or dry-run entries are not treated as completed
- the output file is updated incrementally after each processed item

This is intended to make long judge runs robust to interruptions or transient API failures.

## Output Structure

The saved JSON contains:

- `metadata`
- `summary`
- `per_question`

Each `per_question` item includes:

- `question_id`
- `question`
- `gold_answer`
- `extra_answers`
- `prediction`
- `raw_judge_response`
- `parsed_score_1_to_5`
- `mapped_score_0_to_100`
- `status`
- `error`

## Commands

### Export REQA

```bash
python scripts/export_official_results.py \
  --pred_dir /mnt/storage/mini_reqa_outputs/hf_full_deepseek_chat \
  --method reqa \
  --dataset data/open_eqa_hf/qa/open-eqa-hf.json \
  --output /mnt/storage/mini_reqa_outputs/official_eval_inputs/reqa_deepseek_chat_results.json
```

### Export Uniform

```bash
python scripts/export_official_results.py \
  --pred_dir /mnt/storage/mini_reqa_outputs/hf_full_deepseek_chat \
  --method uniform \
  --dataset data/open_eqa_hf/qa/open-eqa-hf.json \
  --output /mnt/storage/mini_reqa_outputs/official_eval_inputs/uniform_deepseek_chat_results.json
```

### Dry-run DeepSeek LLM-Match

```bash
python scripts/evaluate_with_deepseek_llm_match.py \
  --dataset data/open_eqa_hf/qa/open-eqa-hf.json \
  --results /mnt/storage/mini_reqa_outputs/official_eval_inputs/reqa_deepseek_chat_results.json \
  --output /mnt/storage/mini_reqa_outputs/deepseek_llm_match/reqa_deepseek_llm_match.json \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --limit 5 \
  --dry_run \
  --verbose
```

### Actual REQA evaluation

```bash
python scripts/evaluate_with_deepseek_llm_match.py \
  --dataset data/open_eqa_hf/qa/open-eqa-hf.json \
  --results /mnt/storage/mini_reqa_outputs/official_eval_inputs/reqa_deepseek_chat_results.json \
  --output /mnt/storage/mini_reqa_outputs/deepseek_llm_match/reqa_deepseek_llm_match.json \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --resume
```

### Actual Uniform evaluation

```bash
python scripts/evaluate_with_deepseek_llm_match.py \
  --dataset data/open_eqa_hf/qa/open-eqa-hf.json \
  --results /mnt/storage/mini_reqa_outputs/official_eval_inputs/uniform_deepseek_chat_results.json \
  --output /mnt/storage/mini_reqa_outputs/deepseek_llm_match/uniform_deepseek_llm_match.json \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --resume
```
