# Data Schema

This document defines the core file formats used by mini-R-EQA.

## `captions.json`

```json
[
  {
    "frame_id": "frame_001",
    "image_path": "path/to/frame_001.png",
    "caption": "..."
  }
]
```

Notes:

- `frame_id` must be stable because retrieved evidence refers to it downstream.
- `image_path` may be absent in older toy captions, but real frame-captioning outputs should include it.
- `caption` is the text memory used by retrieval and prompting.

## `questions.json`

```json
[
  {
    "question_id": "q1",
    "question": "...",
    "answer": "...",
    "episode_history": "...",
    "scene_id": "...",
    "gold_frame_ids": ["frame_001"]
  }
]
```

Notes:

- `question_id` should be stable across prediction and evaluation outputs.
- `gold_frame_ids` is optional.
- Toy data can use `gold_frame_ids` for retrieval evaluation.
- Real OpenEQA-style QA data usually does not provide frame-level evidence labels, so retrieval evaluation is not always applicable.

## Prediction Report

```json
{
  "episode_dir": "...",
  "retriever": "...",
  "runner": "...",
  "top_k": 3,
  "model": "...",
  "predictions": [
    {
      "question_id": "...",
      "question": "...",
      "gold_answer": "...",
      "retrieved": [],
      "prompt": "...",
      "predicted_answer": "..."
    }
  ]
}
```

Notes:

- Each prediction item keeps the retrieved evidence and full prompt for debugging.
- `gold_frame_ids` may appear in prediction items when the source question includes them.
- Answer evaluation only requires `gold_answer` and `predicted_answer`.
