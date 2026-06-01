#!/usr/bin/env bash
# Server training pipeline for selector/scorer.
# Assumes captions and caption embeddings have already been built.
# Run from the project root directory.
#
# Usage:
#   bash scripts/server_train_selector_scorer.sh
#   bash scripts/server_train_selector_scorer.sh <candidate_config> <scorer_config> <selector_config> <dual_config> <inference_config>
#
# Example overriding all configs:
#   bash scripts/server_train_selector_scorer.sh \
#       configs/server/generate_candidate_reward_dataset.yaml \
#       configs/server/train_scorer.yaml \
#       configs/server/train_selector.yaml \
#       configs/server/train_dual_network.yaml \
#       configs/server/run_selector_inference.yaml

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CANDIDATE_CONFIG="${1:-configs/server/generate_candidate_reward_dataset.yaml}"
SCORER_CONFIG="${2:-configs/server/train_scorer.yaml}"
SELECTOR_CONFIG="${3:-configs/server/train_selector.yaml}"
DUAL_CONFIG="${4:-configs/server/train_dual_network.yaml}"
INFERENCE_CONFIG="${5:-configs/server/run_selector_inference.yaml}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

# ── Safety checks ─────────────────────────────────────────────────────────────

log "Checking safety preconditions..."

# 1. torch must be importable
python3 -c "import torch" 2>/dev/null || fail "PyTorch is not installed. Install torch before running server training."
log "torch OK ($(python3 -c 'import torch; print(torch.__version__)'))"

# 2. Extract key config values from the scorer config
EPISODE_DIR=$(python3 -c "
import yaml, sys
with open('$SCORER_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('episode_dir', ''))
")
EMBEDDING_DIR=$(python3 -c "
import yaml, sys
with open('$CANDIDATE_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('embedding_cache_dir', ''))
")
DATASET_PATH=$(python3 -c "
import yaml, sys
with open('$SCORER_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('dataset_path', ''))
")

# 3. Check embedding files exist
if [ -n "$EMBEDDING_DIR" ]; then
    [ -f "$EMBEDDING_DIR/caption_embeddings.npy" ] || fail "caption_embeddings.npy not found in $EMBEDDING_DIR"
    [ -f "$EMBEDDING_DIR/caption_embedding_meta.json" ] || fail "caption_embedding_meta.json not found in $EMBEDDING_DIR"
    log "Embedding files OK in $EMBEDDING_DIR"
else
    log "WARNING: embedding_cache_dir not set in $CANDIDATE_CONFIG — will use lexical fallback"
fi

# ── Print config summary ──────────────────────────────────────────────────────

log "=== Server Training Pipeline ==="
log "Candidate config: $CANDIDATE_CONFIG"
log "Scorer config:    $SCORER_CONFIG"
log "Selector config:  $SELECTOR_CONFIG"
log "Dual config:      $DUAL_CONFIG"
log "Inference config: $INFERENCE_CONFIG"
log "Episode dir:      $EPISODE_DIR"
log "Dataset path:     $DATASET_PATH"
log "================================"

# ── Step 1: Generate candidate reward dataset ─────────────────────────────────

log "Step 1/5: Generating candidate reward dataset..."
python3 scripts/generate_candidate_reward_dataset.py --config "$CANDIDATE_CONFIG"

# 4. Check reward variance after generation
REWARD_VARIANCE=$(python3 -c "
import json, yaml
with open('$CANDIDATE_CONFIG') as f:
    cfg = yaml.safe_load(f)
summary_file = cfg.get('summary_output', 'outputs/candidate_reward_summary.json')
try:
    with open(summary_file) as f:
        s = json.load(f)
    print(s.get('reward_std', 1.0))
except Exception:
    print(1.0)
")

if python3 -c "import sys; sys.exit(0 if float('$REWARD_VARIANCE') > 0 else 1)" 2>/dev/null; then
    log "Reward variance OK (std=$REWARD_VARIANCE)"
else
    log "WARNING: reward variance is zero (std=0). All rewards are identical."
    log "         This will cause degenerate pseudo labels. Use --runner deepseek for real data."
    # Check if allow_zero_variance_reward is set
    ALLOW=$(python3 -c "
import yaml
with open('$SCORER_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(str(cfg.get('allow_zero_variance_reward', False)).lower())
")
    if [ "$ALLOW" != "true" ]; then
        fail "Reward variance is zero and allow_zero_variance_reward is not set. Aborting training."
    fi
fi

# 5. Verify dataset exists before training
[ -f "$DATASET_PATH" ] || fail "Candidate reward dataset not found at $DATASET_PATH. Cannot train."
log "Dataset found at $DATASET_PATH"

# ── Step 2: Train scorer ──────────────────────────────────────────────────────

log "Step 2/5: Training scorer (torch backend)..."
python3 scripts/train_scorer.py --config "$SCORER_CONFIG"

# ── Step 3: Train selector ────────────────────────────────────────────────────

log "Step 3/5: Training selector (torch backend)..."
python3 scripts/train_selector.py --config "$SELECTOR_CONFIG"

# ── Step 4: Dual-network training ────────────────────────────────────────────
# train_dual_network.py requires fallback JSON checkpoints (not torch .pt).
# Generate them here if they don't already exist.

SCORER_FALLBACK=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$DUAL_CONFIG')); print(cfg.get('scorer_checkpoint','outputs/scorer_fallback.json'))")
SELECTOR_FALLBACK=$(python3 -c "import yaml; cfg=yaml.safe_load(open('$DUAL_CONFIG')); print(cfg.get('selector_checkpoint','outputs/selector_fallback.json'))")

if [ ! -f "$SCORER_FALLBACK" ]; then
    log "Generating fallback scorer checkpoint for dual-network training..."
    python3 scripts/train_scorer.py \
        --config "$SCORER_CONFIG" \
        --backend fallback \
        --output "$SCORER_FALLBACK" \
        --epochs 5
fi
if [ ! -f "$SELECTOR_FALLBACK" ]; then
    log "Generating fallback selector checkpoint for dual-network training..."
    python3 scripts/train_selector.py \
        --config "$SELECTOR_CONFIG" \
        --backend fallback \
        --output "$SELECTOR_FALLBACK" \
        --epochs 5
fi

log "Step 4/5: Dual-network training (fallback format)..."
python3 scripts/train_dual_network.py --config "$DUAL_CONFIG"

# ── Step 5: Selector inference smoke test ─────────────────────────────────────

log "Step 5/5: Selector inference smoke test..."
python3 scripts/selector_inference_smoke_test.py --config "$INFERENCE_CONFIG"

log "=== Pipeline complete ==="
log "Outputs:"
log "  Scorer checkpoint:     $(python3 -c "import yaml; cfg=yaml.safe_load(open('$SCORER_CONFIG')); print(cfg.get('output','?'))")"
log "  Selector checkpoint:   $(python3 -c "import yaml; cfg=yaml.safe_load(open('$SELECTOR_CONFIG')); print(cfg.get('output','?'))")"
log "  Dual checkpoint:       $(python3 -c "import yaml; cfg=yaml.safe_load(open('$DUAL_CONFIG')); print(cfg.get('output','?'))")"
log "  Scorer metrics:        $(python3 -c "import yaml; cfg=yaml.safe_load(open('$SCORER_CONFIG')); print(cfg.get('metrics_output','?'))")"
log "  Selector metrics:      $(python3 -c "import yaml; cfg=yaml.safe_load(open('$SELECTOR_CONFIG')); print(cfg.get('metrics_output','?'))")"
log "Copy these back to your local machine with scp or rsync."
