# Dual Network Limitations

This repository's first dual-network training stage is intentionally conservative.

## Current Strategy

- The scorer is treated as a pre-trained frozen checkpoint.
- The selector is fine-tuned with:
  - pseudo-label BCE loss
  - scorer predicted reward auxiliary signal

## What This Does Not Include

- No GRPO
- No PPO
- No policy-gradient RL
- No trajectory-level rollout optimization
- No actor-critic framework
- No joint end-to-end differentiable selector-scorer optimization

## Why

The goal of this first stage is to validate the basic selector-scorer handoff with a simple and debuggable training path before introducing more complex optimization schemes.

## Practical Consequences

- Auxiliary reward is only a heuristic guidance signal.
- The scorer remains fixed during the staged dual-training run.
- The selector checkpoint produced here should be treated as a first-pass artifact, not as a final optimized policy.
