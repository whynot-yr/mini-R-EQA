# PR-9 Report: Torch-Native Dual-Network Training

## Summary

Implemented a differentiable dual-network training path. The scorer is loaded from a torch checkpoint, frozen, and used as a surrogate reward model. The selector is fine-tuned with a combined BCE pseudo-label loss and a soft-selection scorer auxiliary loss.

## Files Changed

| File | Change |
|------|--------|
| `mini_eqa/training/train_dual_network.py` | Added `_load_torch_checkpoint()`, `train_torch_dual_network()` with frozen scorer + soft selection |
| `mini_eqa/inference/selector_inference.py` | Added `is_torch_checkpoint()`, `select_top_k_frames_torch()` for torch checkpoint inference |
| `scripts/train_dual_network.py` | Full rewrite: `--backend {torch,fallback}`, `--lambda_aux`, `--temperature`, `--embedding_model`, `--device`; fallback path unchanged |
| `scripts/selector_inference_smoke_test.py` | Auto-detect torch vs JSON via `is_torch_checkpoint()`; route to appropriate inference function |
| `docs/torch_dual_network_training.md` | New documentation |

## Training Objective

```
loss = BCE(selector_logits, pseudo_labels)
     + lambda_aux × (−mean_scorer_utility)

scorer_utility = scorer(q_emb, Σ_i softmax(logits/T)_i × frame_emb_i)
```

- Scorer parameters: `requires_grad_(False)` → excluded from Adam optimizer
- Gradients from aux_loss flow through: utility → pooled → weights → logits → selector
- Hard top-k is NOT used in the backward pass

## Checks Passed

- `python3 -m compileall mini_eqa scripts` ✓
- `python3 scripts/train_dual_network.py --help` ✓
- Torch backend end-to-end (scorer q_dim=384, selector frame_dim=385): 2 questions, 2 epochs, aux_loss negative (utility being maximized) ✓
- Fallback backend end-to-end (JSON checkpoints): unchanged behavior ✓
- `selector_inference_smoke_test.py`: torch checkpoint → `select_top_k_frames_torch` ✓
- `selector_inference_smoke_test.py`: JSON checkpoint → `select_top_k_frames` ✓
- Hard failure for JSON checkpoint with `--backend torch` (framework check) ✓
- Validation: q_dim mismatch between scorer and selector → clear error ✓
- Validation: example frame_dim mismatch with checkpoint → clear error ✓

## Server Command

```bash
python3 scripts/train_dual_network.py \
    --backend torch \
    --dataset_path outputs/candidate_reward_dataset.jsonl \
    --episode_dir /root/AI/mini-R-EQA/data/openeqa_prepared_hf/hm3d-v0__018 \
    --embedding_subdir sentence-transformers_all-MiniLM-L6-v2 \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --scorer_checkpoint outputs/scorer/scorer_torch.pt \
    --selector_checkpoint outputs/selector/selector_torch.pt \
    --device cuda \
    --output outputs/dual/selector_dual_torch.pt \
    --metrics_output outputs/dual/dual_train_metrics.json \
    --epochs 10 \
    --lambda_aux 0.1 \
    --temperature 1.0
```

## Known Limitations

- This is not RL/GRPO: no policy gradient, no rollouts, no KL penalty
- Scorer is used as a fixed frozen reward model, not retrained
- `select_top_k_frames_torch` encodes questions at inference time (not cached) — adds SBERT latency per query
