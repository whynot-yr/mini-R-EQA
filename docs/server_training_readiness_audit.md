# Server Training Readiness Audit

Date: 2026-06-01
Branch: `selector-scorer-pr6-pr8`

Reviewed commits:
- `4db0869` PR-6: Connect MLPScorer/MLPSelector to real torch training path
- `a52868a` PR-7: SBERT candidate ranking, extended reward schema, summary output
- `231f0e4` PR-8: Server training configs, pipeline script, and documentation

## Verdict

`PASS WITH CAVEATS`

It is safe to push and run small-scale server training after the compatibility fixes in this audit:
- torch training now fails loudly if PyTorch is missing
- torch training no longer silently falls back to hashed question embeddings
- dual-network auxiliary scoring now accepts both old `candidate_frames` and new `frame_ids` dataset rows
- the server smoke-test config now points at the fallback-format dual selector checkpoint it can actually load

Remaining caveat:
- `train_dual_network.py` is still fallback-only and does not perform full torch joint training

## Pass/Fail Table

| Area | Status | Evidence | Notes |
|---|---|---|---|
| Torch scorer path uses `MLPScorer` | Pass | `mini_eqa/training/train_scorer.py:61-133` | Imports `MLPScorer`, uses `Adam`, `nn.MSELoss`, `torch.tensor`, `loss.backward()`, `optimizer.step()` |
| Torch selector path uses `MLPSelector` | Pass | `mini_eqa/training/train_selector.py:75-147` | Imports `MLPSelector`, uses `Adam`, `nn.BCEWithLogitsLoss`, `torch.tensor`, `loss.backward()`, `optimizer.step()` |
| `backend=torch` fails loudly if torch is missing | Pass | `scripts/train_scorer.py:90-99`, `scripts/train_selector.py:96-105` | Verified locally: both CLIs exit with a clear error before training |
| Torch checkpoints saved with `torch.save` | Pass | `mini_eqa/training/train_scorer.py:136-151`, `mini_eqa/training/train_selector.py:150-165` | Fallback path uses JSON save only |
| Fallback checkpoints are not defaulted to `.pt` | Pass | `scripts/train_scorer.py:155-159`, `scripts/train_selector.py:161-165` | Default fallback names end with `.json` |
| Torch training avoids hashed question embeddings | Pass after fix | `scripts/train_scorer.py:161-167`, `scripts/train_selector.py:167-173` | Torch mode now resolves model from caption embedding metadata and passes SBERT model into dataset builders |
| Question embeddings use same SBERT family as caption embeddings | Pass after fix | `scripts/train_scorer.py:106-146`, `scripts/train_selector.py:112-152` | Canonical family match enforced against `caption_embedding_meta.json`; metadata model string is used for training |
| Embedding dimensions are consistent | Pass | `mini_eqa/training/scorer_dataset.py:50-79`, `mini_eqa/training/selector_pseudo_labels.py:67-141` | Scorer uses question dim 384 + candidate dim 384; selector uses question dim 384 + frame dim 385 when cosine feature is appended |
| Candidate generation uses cached SBERT for real ranking | Pass | `mini_eqa/inference/candidate_generation.py:43-99`, `:166-221` | Cached embeddings plus cosine similarity are used whenever `cache_dir` is provided |
| Lexical fallback is not used on the server path | Pass | `configs/server/generate_candidate_reward_dataset.yaml`, `scripts/server_train_selector_scorer.sh:60-67` | Server config sets `embedding_cache_dir`; script checks for embedding files |
| Lexical fallback only happens when explicitly requested | Partial | `scripts/generate_candidate_reward_dataset.py:289-304` | Outside server configs, fallback still occurs when no cache is supplied and no embeddings dir is found; acceptable for smoke/debug only |
| `retrieval_scores` and `retrieval_ranks` are saved | Pass | `scripts/generate_candidate_reward_dataset.py:177-201` | Stored on every generated row |
| Candidate reward dataset supports `--runner deepseek` | Pass | `scripts/generate_candidate_reward_dataset.py:54`, `:279-284` | CLI and warning path verified from source and `--help` |
| `--runner mock` is treated as debug only | Partial | `docs/server_training_guide.md:71-74` | Docs treat `mock` as smoke/debug; generator itself only warns on zero variance and does not hard-block `mock` |
| Zero reward variance is detected and can abort server training | Pass with caveat | `scripts/generate_candidate_reward_dataset.py:125-135`, `scripts/server_train_selector_scorer.sh:86-115` | Generator warns; server wrapper enforces fail unless `allow_zero_variance_reward` is set |
| JSONL schema is backward-compatible with training | Pass after fix | `mini_eqa/training/scorer_dataset.py:58-71`, `mini_eqa/training/selector_pseudo_labels.py:86-98`, `mini_eqa/inference/scorer_inference.py:14-16`, `:88-105` | Training and dual aux path now accept both `frame_ids` and `candidate_frames` |
| Server configs under `configs/server/` are coherent | Pass after fix | `configs/server/*.yaml` | Step 5 config now points to the dual JSON-format checkpoint |
| `scripts/server_train_selector_scorer.sh` checks torch, embeddings, dataset, and reward variance | Pass | `scripts/server_train_selector_scorer.sh:36-67`, `:86-119` | Verified from source |
| Server script does not call captioning or rebuild embeddings | Pass | `scripts/server_train_selector_scorer.sh:81-161` | It assumes captions/embeddings already exist |
| Server script can run step-by-step on server | Pass with caveat | `scripts/server_train_selector_scorer.sh` plus configs | Requires server env with `torch`, `numpy`, `sentence-transformers`, `pyyaml`, and `DEEPSEEK_API_KEY` |
| Dual-network remains fallback JSON format | Pass | `configs/server/train_dual_network.yaml`, `mini_eqa/training/train_dual_network.py` | This is still a known limitation, not a torch joint-training implementation |
| Dual-network checkpoint incompatibility is guarded | Pass after fix | `configs/server/run_selector_inference.yaml`, `mini_eqa/inference/selector_inference.py:20-27` | Smoke test now uses `outputs/selector_dual.pt`; torch `.pt` selector checkpoints get a clear error |

## Blocking Issues

None remaining for small-scale server training after the fixes above.

Previously blocking issues fixed in this audit:
- `mini_eqa/inference/scorer_inference.py` only read `candidate_frames` in the dual-network auxiliary path, while the current generator writes `frame_ids`
- `configs/server/run_selector_inference.yaml` pointed the smoke test at `outputs/selector_torch.pt`, but `selector_inference_smoke_test.py` only supports fallback-format JSON checkpoints
- `backend=torch` could still silently use `hashed_text_embedding` if `--embedding_model` was omitted

## Non-Blocking Issues

- `scripts/generate_candidate_reward_dataset.py` warns on zero variance but does not itself enforce failure. The fail policy currently lives in `scripts/server_train_selector_scorer.sh`.
- `--runner mock` is documented as debug/smoke-only, but the generator does not hard-block it. The server configs already use `deepseek`, so this does not block server training.
- `scripts/selector_inference_smoke_test.py` still does not load torch-native selector checkpoints. That is acceptable because step 5 is now explicitly a fallback-format smoke test.

## Exact Commands Run

Requested checks:

```bash
git status --short
python3 -m compileall mini_eqa scripts
python3 scripts/generate_candidate_reward_dataset.py --help
python3 scripts/train_scorer.py --help
python3 scripts/train_selector.py --help
python3 scripts/train_dual_network.py --help
python3 scripts/selector_inference_smoke_test.py --help
```

Additional local verification:

```bash
python3 -c "import torch; print(torch.__version__)"
python3 scripts/train_scorer.py --backend torch --dataset_path reports/candidate_reward_dataset.jsonl --episode_dir data/sample_episode --embedding_subdir sentence-transformers_all-MiniLM-L6-v2 --output /tmp/scorer_torch.pt --metrics_output /tmp/scorer_torch_metrics.json
python3 scripts/train_selector.py --backend torch --dataset_path reports/candidate_reward_dataset.jsonl --episode_dir data/sample_episode --embedding_subdir sentence-transformers_all-MiniLM-L6-v2 --output /tmp/selector_torch.pt --metrics_output /tmp/selector_torch_metrics.json --summary_output /tmp/selector_summary.json
python3 -c "import json, random, struct; from pathlib import Path; src=Path('data/sample_episode'); dst=Path('/tmp/mini_circle_audit_episode'); embdir=dst/'embeddings'/'sentence-transformers_all-MiniLM-L6-v2'; embdir.mkdir(parents=True, exist_ok=True); captions=json.load(open(src/'captions.json')); questions=json.load(open(src/'questions.json')); json.dump(captions, open(dst/'captions.json','w'), ensure_ascii=False, indent=2); json.dump(questions, open(dst/'questions.json','w'), ensure_ascii=False, indent=2); rows=len(captions); cols=384; rng=random.Random(0); values=[]; \
for _ in range(rows): \
 row=[rng.uniform(-1.0,1.0) for __ in range(cols)]; \
 norm=sum(v*v for v in row) ** 0.5 or 1.0; \
 values.extend([v/norm for v in row]); \
header=str({'descr':'<f4','fortran_order':False,'shape':(rows, cols)}); preamble_len=10; pad=(16 - ((preamble_len + len(header) + 1) % 16)) % 16; header_text=header + (' ' * pad) + '\n'; \
with open(embdir/'caption_embeddings.npy','wb') as f: \
 f.write(b'\x93NUMPY'); f.write(bytes([1,0])); f.write(struct.pack('<H', len(header_text))); f.write(header_text.encode('latin1')); f.write(struct.pack('<%sf' % (rows*cols), *values)); \
meta={'model_name':'sentence-transformers/all-MiniLM-L6-v2','normalize_embeddings':True,'num_captions':len(captions),'embedding_dim':384,'captions_path':str(dst/'captions.json'),'items':[{'index':i,'frame_id':item['frame_id'],'caption':item['caption']} for i,item in enumerate(captions)]}; json.dump(meta, open(embdir/'caption_embedding_meta.json','w'), ensure_ascii=False, indent=2); print(dst)"
python3 scripts/train_scorer.py --backend fallback --dataset_path reports/candidate_reward_dataset.jsonl --episode_dir /tmp/mini_circle_audit_episode --output /tmp/mini_circle_audit/scorer_fallback.json --metrics_output /tmp/mini_circle_audit/scorer_metrics.json --epochs 1 --max_examples 4
python3 scripts/train_selector.py --backend fallback --dataset_path reports/candidate_reward_dataset.jsonl --episode_dir /tmp/mini_circle_audit_episode --output /tmp/mini_circle_audit/selector_fallback.json --metrics_output /tmp/mini_circle_audit/selector_metrics.json --summary_output /tmp/mini_circle_audit/selector_summary.json --epochs 1 --max_examples 8
python3 scripts/train_dual_network.py --dataset_path reports/candidate_reward_dataset.jsonl --episode_dir /tmp/mini_circle_audit_episode --scorer_checkpoint /tmp/mini_circle_audit/scorer_fallback.json --selector_checkpoint /tmp/mini_circle_audit/selector_fallback.json --output /tmp/mini_circle_audit/selector_dual.pt --metrics_output /tmp/mini_circle_audit/dual_metrics.json --epochs 1 --max_examples 6 --smoke_question_id q1 --smoke_top_k 3
python3 scripts/selector_inference_smoke_test.py --checkpoint /tmp/mini_circle_audit/selector_dual.pt --episode_dir /tmp/mini_circle_audit_episode --question_id q1 --top_k 3
```

## Local Check Results

- `git status --short`: working tree was already dirty before the audit
- `python3 -m compileall mini_eqa scripts`: passed
- all requested `--help` commands: passed
- local `python3 -c "import torch"`: failed with `ModuleNotFoundError: No module named 'torch'`
- local `backend=torch` scorer/selector invocations: failed clearly with the intended error message
- fallback scorer training: passed
- fallback selector training: passed
- dual-network training: passed after schema compatibility fix
- selector inference smoke test: passed after switching to the dual JSON-format checkpoint

Local environment limitation:
- `torch` is not installed
- `numpy` is not installed

Because of that, no real torch training was executed locally.

## Safe To Push?

Yes, for small-scale server training.

Conditions:
- run on a server with `torch`, `numpy`, `sentence-transformers`, and `pyyaml`
- ensure caption embeddings already exist
- use the provided server configs
- use `deepseek`, not `mock`, for real candidate reward generation

Not yet safe to claim:
- end-to-end validation of the torch selector checkpoint at inference time
- full torch-based dual/joint selector-scorer training

## Recommended Server Commands

One-shot pipeline:

```bash
export DEEPSEEK_API_KEY=...
pip install torch numpy sentence-transformers pyyaml
bash scripts/server_train_selector_scorer.sh
```

Step-by-step:

```bash
export DEEPSEEK_API_KEY=...
pip install torch numpy sentence-transformers pyyaml

python3 scripts/generate_candidate_reward_dataset.py \
  --config configs/server/generate_candidate_reward_dataset.yaml

python3 scripts/train_scorer.py \
  --config configs/server/train_scorer.yaml

python3 scripts/train_selector.py \
  --config configs/server/train_selector.yaml

python3 scripts/train_dual_network.py \
  --config configs/server/train_dual_network.yaml

python3 scripts/selector_inference_smoke_test.py \
  --config configs/server/run_selector_inference.yaml
```

Pre-flight checks recommended on the server:

```bash
python3 -c "import torch, numpy, yaml; print(torch.__version__)"
test -f data/openeqa_prepared/sample_episode/captions.json
test -f data/openeqa_prepared/sample_episode/questions.json
test -f data/openeqa_prepared/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2/caption_embeddings.npy
test -f data/openeqa_prepared/sample_episode/embeddings/sentence-transformers_all-MiniLM-L6-v2/caption_embedding_meta.json
```
