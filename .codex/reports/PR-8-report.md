# PR-8 Report: Prepare Server Training

## Summary

Added server-oriented YAML configs, a training shell script, and documentation for running the full selector/scorer pipeline on a GPU server. No new model architecture. No local expensive training.

## Files Added

| File | Description |
|------|-------------|
| `configs/server/generate_candidate_reward_dataset.yaml` | Config for dataset generation (deepseek runner, SBERT ranking) |
| `configs/server/train_scorer.yaml` | Config for scorer training (torch backend, cuda) |
| `configs/server/train_selector.yaml` | Config for selector training (torch backend, cuda) |
| `configs/server/train_dual_network.yaml` | Config for dual-network training (fallback format) |
| `configs/server/run_selector_inference.yaml` | Config for selector inference smoke test |
| `scripts/server_train_selector_scorer.sh` | End-to-end training pipeline script |
| `docs/server_training_guide.md` | Full documentation for server training |

## Files Modified

| File | Change |
|------|--------|
| `scripts/train_dual_network.py` | Added --config YAML support |
| `scripts/selector_inference_smoke_test.py` | Added --config YAML support; required args validated at runtime |

## Shell Script Safety Checks

1. **torch availability** — fails immediately if torch not installed
2. **Embedding files** — checks `caption_embeddings.npy` and `caption_embedding_meta.json` exist
3. **Reward variance** — fails after dataset generation if reward_std == 0 (unless `allow_zero_variance_reward: true` in scorer config)
4. **Dataset existence** — verifies JSONL exists before training
5. **Fallback pre-step** — auto-generates fallback JSON checkpoints before dual-network training if they don't exist

## Checks Passed

- `python3 -m compileall mini_eqa scripts` ✓
- `bash -n scripts/server_train_selector_scorer.sh` (syntax check) ✓
- `python3 scripts/generate_candidate_reward_dataset.py --config configs/server/generate_candidate_reward_dataset.yaml --runner mock --limit 1 --dry_run` ✓
- `python3 scripts/train_scorer.py --help` ✓
- `python3 scripts/train_selector.py --help` ✓
- `python3 scripts/train_dual_network.py --help` ✓

## Known Limitations

- `train_dual_network.py` uses the old fallback JSON checkpoint format. The shell script auto-generates the fallback checkpoints before calling it.
- `run_selector_inference.yaml` points at `outputs/selector_torch.pt` (step 3), not the dual-network output. Smoke tests the primary torch-trained checkpoint.
- DeepSeek API calls are not tested locally (no API key). The runner is `deepseek` in the server config but can be overridden with `--runner mock` for smoke tests.
