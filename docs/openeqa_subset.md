# OpenEQA Small Subset

This document defines the directory convention for preparing a small OpenEQA subset inside mini-R-EQA.

## Directory Convention

```text
data/openeqa_prepared/
  episode_xxx/
    questions.json
    episode_meta.json
    captions.json
    embeddings/
```

`questions.json` and `episode_meta.json` are created during subset preparation.

`captions.json` and `embeddings/` are generated later.

## Preparation Flow

1. prepare subset
2. caption frames
3. build embeddings
4. run predictions
5. export predictions
6. evaluate

## Real Data Placement

- `qa_file` should point to an OpenEQA-style QA metadata JSON file.
- `frames_root` should point to the root directory that contains episode-history folders.
- The helper resolves frame paths as `frames_root / episode_history`.

## Notes

- The subset preparation script does not copy frame images.
- The subset preparation script does not generate captions.
- If a frame directory cannot be resolved, the script warns by default and records `frames_dir: null`.
- Use `--strict_frames` when you want missing frame directories to fail immediately.
