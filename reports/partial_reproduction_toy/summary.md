# Partial Reproduction Summary

Simple answer metrics are sanity checks, not official OpenEQA LLM-Match.

| episode | method | num_questions | exact_match | contains_gold | token_f1 | predictions_path |
|---|---|---:|---:|---:|---:|---|
| sample_episode | reqa | 0 | n/a | n/a | n/a | Missing captions.json. Run captioning first: python -m mini_eqa.captioning.caption_frames --frames_dir path/to/episode_frames --output data/openeqa_prepared_toy/sample_episode/captions.json --backend filename_stub --overwrite |
| sample_episode | uniform | 0 | n/a | n/a | n/a | Missing captions.json. Run captioning first: python -m mini_eqa.captioning.caption_frames --frames_dir path/to/episode_frames --output data/openeqa_prepared_toy/sample_episode/captions.json --backend filename_stub --overwrite |
