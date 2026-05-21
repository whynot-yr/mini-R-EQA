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
- filename_stub only
- no real VLM captioning yet
- no official OpenEQA prediction exporter yet
- simple answer metrics only
