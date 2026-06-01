# Torch Dual-Network Training

## Overview

The dual-network training fine-tunes the selector while using a frozen scorer to provide auxiliary gradient signal. This is the first stage of differentiable frame selection; future work can upgrade it to RL/GRPO.

## Architecture

```
          ┌──────────────────────────────────────────────────┐
          │           Training Loop (per question)           │
          │                                                  │
          │  q_emb ──►┐                                      │
          │            ├──► MLPSelector ──► logits [N]       │
          │  frame_embs┘          │                          │
          │                       │ BCEWithLogitsLoss        │
          │                       │ (vs pseudo labels)       │
          │                       │                          │
          │                       ▼                          │
          │              softmax(logits / T)                 │
          │                       │                          │
          │                       │ soft weights [N]         │
          │                       ▼                          │
          │              weighted pooled frame emb           │
          │                       │                          │
          │  q_emb ──►┐           │                          │
          │            ├──► MLPScorer (FROZEN) ──► utility   │
          │  pooled ───┘                           │         │
          │                            -mean() = aux_loss   │
          │                                                  │
          │  total_loss = BCE + λ_aux * aux_loss             │
          │  ∂total_loss / ∂selector_params only             │
          └──────────────────────────────────────────────────┘
```

## Key Design Decisions

### Scorer is frozen

The scorer is loaded from `scorer_torch.pt` and its parameters are excluded from the optimizer:
```python
for param in scorer.parameters():
    param.requires_grad_(False)
```

The scorer's computation graph is still active so gradients from the auxiliary loss flow through the pooled embedding back into the selector.

### Soft selection (differentiable, not hard top-k)

During training, all N frames in a question's candidate set are scored simultaneously:
```
weights = softmax(selector_logits / temperature)
pooled  = Σ_i(weights_i × frame_emb_i)
```

This keeps the training graph fully differentiable. Hard top-k is non-differentiable and cannot be used inside the backward pass.

### Hard top-k during inference only

At inference time, `select_top_k_frames_torch` ranks frames by selector logit and returns the top-k:
```python
ranked.sort(key=lambda item: -item["score"])
return ranked[:top_k]
```

### Frame embedding stripping for scorer

The selector may have `frame_dim = question_dim + 1` (384 SBERT + 1 cosine_sim = 385) while the scorer expects `candidate_dim = 384`. The first `scorer_candidate_dim` dimensions are extracted from the selector's frame embeddings before passing to the scorer:
```python
scorer_frame_tensor = frame_tensor[:, :scorer_cand_dim]
pooled = (weights.unsqueeze(1) * scorer_frame_tensor).sum(0)
```

### This is not RL/GRPO

- No policy gradient or REINFORCE
- No reward rollouts
- No KL penalty
- The scorer is a learned reward model, not an environment

The differentiable approach here is a surrogate for GRPO: it approximates "select frames that maximize predicted answer quality" using a smooth, differentiable objective.

## Usage

### Command line

```bash
python3 scripts/train_dual_network.py \
    --backend torch \
    --dataset_path outputs/candidate_reward_dataset.jsonl \
    --episode_dir /path/to/episode \
    --embedding_subdir sentence-transformers_all-MiniLM-L6-v2 \
    --scorer_checkpoint outputs/scorer/scorer_torch.pt \
    --selector_checkpoint outputs/selector/selector_torch.pt \
    --device cuda \
    --output outputs/dual/selector_dual_torch.pt \
    --metrics_output outputs/dual/dual_train_metrics.json \
    --epochs 10 \
    --lambda_aux 0.1 \
    --temperature 1.0 \
    --smoke_question_id q1
```

### Config file

```yaml
# configs/server/train_dual_network.yaml
backend: torch
dataset_path: outputs/candidate_reward_dataset.jsonl
episode_dir: data/openeqa_prepared/sample_episode
embedding_subdir: sentence-transformers_all-MiniLM-L6-v2
scorer_checkpoint: outputs/scorer/scorer_torch.pt
selector_checkpoint: outputs/selector/selector_torch.pt
output: outputs/dual/selector_dual_torch.pt
metrics_output: outputs/dual/dual_train_metrics.json
device: cuda
epochs: 10
learning_rate: 0.001
lambda_aux: 0.1
temperature: 1.0
```

```bash
python3 scripts/train_dual_network.py --config configs/server/train_dual_network.yaml
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_aux` | 0.1 | Weight of scorer auxiliary loss vs BCE |
| `temperature` | 1.0 | Softmax temperature for soft selection. Lower = sharper weights. |
| `epochs` | 10 | Training epochs |
| `learning_rate` | 1e-3 | Adam learning rate for selector |

**Tuning guidance:**
- Increase `lambda_aux` if the selector ignores scorer signal (aux_loss not decreasing)
- Decrease `lambda_aux` if BCE loss stops decreasing (pseudo-label accuracy regresses)
- Lower `temperature` (e.g. 0.5) makes selection sharper, closer to hard top-k
- Higher `temperature` (e.g. 2.0) spreads weight more evenly (smoother gradients)

## Checkpoint Format

The output `selector_dual_torch.pt` is a `torch.save` dict:

```python
{
    "framework": "torch",
    "question_dim": 384,
    "frame_dim": 385,
    "hidden_dim": 256,
    "state_dict": {...},   # MLPSelector weights
    "dual_training": {
        "strategy": "soft_selection_with_frozen_scorer",
        "lambda_aux": 0.1,
        "temperature": 1.0,
        "scorer_question_dim": 384,
        "scorer_candidate_dim": 384,
    }
}
```

Load for inference:
```python
import torch
from mini_eqa.selector.mlp_selector import MLPSelector

ckpt = torch.load("selector_dual_torch.pt", weights_only=False, map_location="cpu")
model = MLPSelector(ckpt["question_dim"], ckpt["frame_dim"], ckpt["hidden_dim"])
model.load_state_dict(ckpt["state_dict"])
model.eval()
```

Or use the provided helper:
```python
from mini_eqa.inference.selector_inference import select_top_k_frames_torch

selected = select_top_k_frames_torch(
    checkpoint_path="outputs/dual/selector_dual_torch.pt",
    episode_dir="data/sample_episode",
    question_id="q1",
    top_k=3,
)
```

## Fallback Backend

`--backend fallback` uses the old pure-Python linear dual trainer (JSON checkpoints). This is kept for backward compatibility with PR-5 artifacts and does not use MLPSelector/MLPScorer.

## Prerequisite Checks

If `--backend torch` but either checkpoint is in JSON format (framework ≠ "torch"), the script fails loudly:
```
ValueError: Expected a torch-format checkpoint at ..., got framework='python_fallback_linear'.
Use --backend fallback for JSON checkpoints, or train with --backend torch first.
```
