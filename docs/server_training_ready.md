# Server Training Ready: Scorer and Selector

## Overview

The scorer and selector now support a real PyTorch MLP training backend in addition to the original pure-Python fallback.

| Backend    | Model           | Loss               | Checkpoint format |
|------------|-----------------|--------------------|--------------------|
| `torch`    | `MLPScorer` / `MLPSelector` | MSELoss / BCEWithLogitsLoss | `torch.save` `.pt` |
| `fallback` | Linear (pure Python) | MSE / BCE (manual) | JSON `.json` |

## Torch Backend

### Requirements

- PyTorch installed (`pip install torch`)
- `sentence-transformers` installed for SBERT question encoding (`pip install sentence-transformers`)

### Inputs

**Scorer**

| Feature | Dim | Source |
|---------|-----|--------|
| Question embedding | 384 | SBERT `all-MiniLM-L6-v2` (encoded at training time) |
| Candidate frame embedding | 384 | Mean-pooled from cached frame embeddings |

**Selector**

| Feature | Dim | Source |
|---------|-----|--------|
| Question embedding | 384 | SBERT `all-MiniLM-L6-v2` (encoded at training time) |
| Frame embedding + cosine sim | 385 | Cached frame embedding + dot(q, f) |

### Outputs

| File | Description |
|------|-------------|
| `scorer_torch.pt` | `torch.save` dict: `{framework, question_dim, candidate_dim, hidden_dim, state_dict}` |
| `selector_torch.pt` | `torch.save` dict: `{framework, question_dim, frame_dim, hidden_dim, state_dict}` |
| `scorer_train_metrics.json` | Loss history and config |
| `selector_train_metrics.json` | Loss history and config |

### Loading a checkpoint

```python
import torch
from mini_eqa.scorer.mlp_scorer import MLPScorer

ckpt = torch.load("reports/scorer_torch.pt", map_location="cpu")
model = MLPScorer(
    question_dim=ckpt["question_dim"],
    candidate_dim=ckpt["candidate_dim"],
    hidden_dim=ckpt["hidden_dim"],
)
model.load_state_dict({k: torch.tensor(v) for k, v in ckpt["state_dict"].items()})
model.eval()
```

## Fallback Backend

The fallback trains a single linear layer using pure Python (no torch dependency).

| File | Description |
|------|-------------|
| `scorer_fallback.json` | JSON dict with `weights`, `bias`, dimension metadata |
| `selector_fallback.json` | JSON dict with `weights`, `bias`, dimension metadata |

Fallback is intended for CI smoke tests only. Do not use for real training.

## CLI Commands

### Scorer

```bash
# Torch backend (real training)
python3 scripts/train_scorer.py \
    --backend torch \
    --dataset_path reports/candidate_reward_dataset.jsonl \
    --episode_dir data/sample_episode \
    --embedding_model all-MiniLM-L6-v2 \
    --epochs 10 \
    --learning_rate 1e-3 \
    --device cpu

# Fallback (smoke test only)
python3 scripts/train_scorer.py --backend fallback
```

### Selector

```bash
# Torch backend (real training)
python3 scripts/train_selector.py \
    --backend torch \
    --dataset_path reports/candidate_reward_dataset.jsonl \
    --episode_dir data/sample_episode \
    --embedding_model all-MiniLM-L6-v2 \
    --epochs 10 \
    --learning_rate 1e-3 \
    --device cpu

# Fallback (smoke test only)
python3 scripts/train_selector.py --backend fallback
```

### Config file

Both scripts accept `--config path/to/config.yaml`. YAML values set the defaults; CLI flags override them.

```yaml
backend: torch
dataset_path: data/openeqa_prepared/candidate_reward_dataset.jsonl
episode_dir: data/openeqa_prepared
embedding_model: all-MiniLM-L6-v2
epochs: 20
learning_rate: 0.001
device: cuda
hidden_dim: 256
```

## Backend Behavior Rules

- `backend=torch`: If torch is not installed, the script exits immediately with a clear error.
- `backend=torch`: The script resolves the caption embedding metadata model and uses that SBERT family for question embeddings; it fails if the cache metadata is missing or the requested model family mismatches.
- `backend=fallback`: Allowed only for smoke tests. Does not use `MLPScorer` / `MLPSelector`.
- Checkpoints from the two backends are **not interchangeable** — do not load a fallback JSON with `torch.load`.
- `scripts/selector_inference_smoke_test.py` currently supports fallback-format JSON checkpoints only; use the dual-network output or a fallback selector checkpoint for smoke inference.
