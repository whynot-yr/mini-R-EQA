# Real Captioning

`filename_stub` only tests file flow. It does not inspect image content and should not be used for real experiments.

`qwen_vl` is intended for real frame captioning and is the bridge toward the original R-EQA caption generation stage.

## Practical Constraint

Real captioning is the first stage in this repository that is likely to require a rented GPU and extra dependencies such as `torch`, `transformers`, `qwen-vl-utils`, and `pillow`.

This repository does not auto-install those dependencies and does not auto-download models.

## Recommended Small-Subset Workflow

1. prepare OpenEQA subset
2. run `caption_frames` with `qwen_vl` on one episode
3. build caption embeddings
4. run R-EQA top-3 and uniform top-10
5. evaluate

For smoke tests, start with `--limit 2`.

## Current State

- `filename_stub`: working file-flow backend
- `qwen_vl`: real single-image captioning backend
- recommended first test: one episode with `--limit 2`
