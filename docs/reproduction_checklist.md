# R-EQA Reproduction Checklist

## Protocol

- [ ] R-EQA top-k = 3
- [ ] Uniform top-k = 10
- [ ] Caption model fixed
- [ ] Embedding model fixed
- [ ] Answer model fixed
- [ ] Evaluation protocol fixed

## Data

- [ ] OpenEQA QA metadata loaded
- [ ] Episode history frames resolved
- [ ] captions.json generated per episode
- [ ] caption embeddings cached per episode

## Inference

- [ ] R-EQA cached_sbert top-3
- [ ] Uniform k=10
- [ ] Blind LLM baseline, future optional

## Output

- [ ] Internal prediction report
- [ ] OpenEQA-compatible prediction export
- [ ] Evaluation report

## Current Status

- toy pipeline complete
- filename_stub for smoke test, qwen_vl for real captioning
- official OpenEQA export and wrapper exist
- simple answer metrics only
- final OpenEQA LLM-Match still depends on the real official evaluator
