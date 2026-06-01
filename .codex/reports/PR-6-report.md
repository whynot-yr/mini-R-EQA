# PR-6 Report: Fix Selector/Scorer Training Paths

## Summary

Connected `MLPScorer` and `MLPSelector` PyTorch models to the real training pipeline. The previous code trained only a pure-Python linear model and saved JSON checkpoints with `.pt` suffix. This PR adds a proper `torch` backend with `MSELoss`/`BCEWithLogitsLoss` and `torch.save` checkpoints.

## Files Changed

| File | Change |
|------|--------|
| `mini_eqa/training/scorer_dataset.py` | Optional SBERT question encoding; fix numpy float32→Python float conversion bug |
| `mini_eqa/training/selector_pseudo_labels.py` | Same + cosine similarity appended to frame features in sbert mode |
| `mini_eqa/training/train_scorer.py` | Added `train_torch_scorer()` using MLPScorer + MSELoss + Adam |
| `mini_eqa/training/train_selector.py` | Added `train_torch_selector()` using MLPSelector + BCEWithLogitsLoss + Adam |
| `scripts/train_scorer.py` | Added `--backend {torch,fallback}`, `--config`, `--device`, `--hidden_dim`, `--embedding_model` |
| `scripts/train_selector.py` | Same |
| `docs/server_training_ready.md` | New documentation |

## Bugs Fixed

- **numpy float32 serialization**: `list(bundle.caption_embeddings[...])` produced numpy float32 values not JSON-serializable. Changed to `[float(v) for v in ...]`.
- **tolist() in state_dict**: Changed `v.cpu().tolist()` to `v.cpu()` so torch checkpoint contains real tensors loadable by `model.load_state_dict()`.
- **Silent empty embedding**: Changed `question_emb_map.get(question_id, [])` to direct dict access to fail loudly on missing question_id.

## Checks Passed

- `python3 -m compileall mini_eqa scripts` ✓
- `python3 scripts/train_scorer.py --help` ✓
- `python3 scripts/train_selector.py --help` ✓
- Fallback smoke (dry_run, 3 examples): scorer and selector both complete ✓
- Torch smoke (dry_run, 5 examples, 2 epochs): scorer and selector both complete ✓
- Torch checkpoint round-trip: `torch.load` + `model.load_state_dict()` verified for both models ✓

## Checkpoint Naming

| Backend | Scorer | Selector |
|---------|--------|----------|
| torch   | `scorer_torch.pt` | `selector_torch.pt` |
| fallback | `scorer_fallback.json` | `selector_fallback.json` |

## Known Limitations

- `train_dual_network.py` (PR-5) still uses the old JSON fallback format. It will fail if given a torch checkpoint. Addressed in PR-8.
- CUDA not available on local machine (old driver). All local tests run on CPU.
