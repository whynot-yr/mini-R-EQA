PR-6：先修 selector/scorer 训练路径

这一 PR 的目标是：让 MLPSelector 和 MLPScorer 真正参与训练。

现在的问题是，selector/mlp_selector.py 和 scorer/mlp_scorer.py 只是定义了 PyTorch 类，但训练脚本没有用它们。Claude 报告明确说这些类目前只在 smoke test 中 import，实际训练路径绕过它们。

你要让 CC 做：

train_scorer.py 使用真正的 MLPScorer
train_selector.py 使用真正的 MLPSelector
checkpoint 用 torch.save
如果 --backend torch 但没有 torch，直接报错
fallback linear 只能保留给 smoke test，不能作为正式训练默认路径

给 Claude Code 的 prompt：

Writing

You are working in my repository mini-R-EQA.

Goal:
Make the selector/scorer implementation server-training-ready by connecting the real PyTorch MLP models to the training pipeline.

Current problem:
The repository already has:

mini_eqa/selector/mlp_selector.py
mini_eqa/scorer/mlp_scorer.py
mini_eqa/scorer/mlp_supervisor.py

But the current training scripts use pure-Python fallback linear trainers and save JSON checkpoints with .pt suffix. This is only acceptable for smoke tests, not for real server training.

Task:
Implement a real torch-backed training path for selector and scorer.

Requirements:

Scorer training
Update:
mini_eqa/training/train_scorer.py
scripts/train_scorer.py

The default real training backend should be torch.

Scorer model:

Use mini_eqa/scorer/mlp_scorer.py.
Input should be:
SBERT question embedding
mean-pooled candidate frame embeddings
optional max-pooled candidate frame embeddings
optional retrieval-score statistics if available
Output:
scalar predicted utility / reward
Loss:
MSELoss for candidate reward regression
Save checkpoint with torch.save.
Selector training
Update:
mini_eqa/training/train_selector.py
scripts/train_selector.py

Selector model:

Use mini_eqa/selector/mlp_selector.py.
Input should be:
SBERT question embedding
frame caption embedding
cosine similarity between question and frame
optional normalized temporal index
optional retrieval score
Output:
scalar frame score
Loss:
BCEWithLogitsLoss using pseudo frame labels from high-reward / low-reward candidate sets.
Save checkpoint with torch.save.
Backend behavior
Add a CLI/config option:
--backend torch
--backend fallback

Rules:

backend=torch is required for real training.
If backend=torch and torch is unavailable, fail loudly with a clear error.
backend=fallback may keep the old pure-Python linear behavior only for smoke tests.
Do not silently fall back to linear training during real training.
Checkpoint naming
Use:
scorer_torch.pt
selector_torch.pt

If fallback checkpoints are still saved, use:

scorer_fallback.json
selector_fallback.json

Do not save JSON checkpoints with .pt.

Documentation
Update README or add:
docs/server_training_ready.md

Explain:

torch backend vs fallback backend
expected inputs
expected outputs
checkpoint format
minimal server training commands
Tests / smoke checks
Run:
python3 -m compileall mini_eqa scripts
python3 scripts/train_scorer.py --help
python3 scripts/train_selector.py --help

If torch is unavailable locally, also run fallback smoke tests. But make sure torch backend errors clearly when torch is missing.

Do not modify candidate reward dataset generation in this PR.
Do not call DeepSeek in this PR.

PR-7：修 candidate reward dataset

这一 PR 的目标是：让训练数据真的有意义。

现在 candidate generation 有两个问题：一是 rank_frames_for_question 用 lexical overlap，不是真正的 dense retrieval；二是 mock runner 会让 reward 大概率全零，导致 selector 伪标签退化。报告里这两个风险都被明确指出了。

你要让 CC 做：

candidate ranking 改成 cached SBERT cosine similarity
question embedding 使用同一个 SBERT 模型
runner 支持 deepseek
mock 只用于 smoke test
candidate JSONL 里保存 retrieval score / rank / candidate type
hard negative 标记：高 retrieval score，但低 reward

给 Claude Code 的 prompt：

Writing

Continue working in mini-R-EQA.

Goal:
Make candidate reward dataset generation suitable for real selector/scorer training.

Current problem:
The current candidate generation path uses lexical overlap as a fallback ranking and the mock runner can produce all-zero rewards. This is fine for smoke tests but not for real training.

Task:
Update candidate reward dataset generation to use real cached SBERT retrieval and support DeepSeek-based answer generation.

Files to inspect/update:

scripts/generate_candidate_reward_dataset.py
mini_eqa/inference/candidate_generation.py
mini_eqa/evaluation/reward_utils.py
mini_eqa/data_loading/selector_scorer_loader.py
existing retrieval modules under mini_eqa/retrieval/

Requirements:

Do not recompute captions or caption embeddings.
Always load existing:
captions.json
questions.json
caption_embeddings.npy
caption_embedding_meta.json
Replace lexical candidate ranking.
Use cached SBERT ranking:
encode question using the same sentence-transformer model as the caption embeddings
compute cosine similarity between question embedding and cached caption embeddings
rank frames by similarity
Parameterize embedding path/model.
Do not hard-code:
sentence-transformers_all-MiniLM-L6-v2

in multiple files.

Add config/CLI fields:

--embedding_model
--embedding_cache_dir

Centralize path resolution in:

mini_eqa/data_loading/selector_scorer_loader.py
Candidate types
For each question, generate candidate sets with top_k=3 by default:
reqa_top3
reqa_top6_random3
uniform3
random3
reqa_perturbed3
middle_rank_random3
low_rank_random3

Also add hard-negative metadata:
A hard negative is a candidate or frame with high retrieval rank/similarity but low answer reward.

Add fields to JSONL:

{
  "question_id": "...",
  "episode_id": "...",
  "candidate_id": "...",
  "candidate_type": "...",
  "top_k": 3,
  "frame_ids": [],
  "selected_items": [],
  "retrieval_scores": [],
  "retrieval_ranks": [],
  "predicted_answer": "...",
  "gold_answer": "...",
  "reward": 0.0,
  "reward_breakdown": {},
  "is_hard_negative": false,
  "debug": {}
}
Runner behavior
Support:
--runner mock for local smoke tests
--runner deepseek for real dataset generation

Do not let mock-generated all-zero rewards be used silently for real training. Add a warning if reward variance is zero.

Reward behavior
Keep local reward metrics for debugging:
exact match
contains gold
token F1

But structure the code so a future LLM-judge reward can be added cleanly.

Outputs
Write:
candidate_reward_dataset.jsonl
candidate_reward_summary.json

Summary should include:

number of questions
number of candidates
reward mean/std/min/max
number of all-zero rewards
hard negative count
candidate type distribution
Smoke tests
Run:
python3 -m compileall mini_eqa scripts
python3 scripts/generate_candidate_reward_dataset.py --help
python3 scripts/generate_candidate_reward_dataset.py --episode_dir data/sample_episode --runner mock --output reports/candidate_reward_dataset_debug.jsonl --limit 2 --dry_run

Do not train selector or scorer in this PR.

PR-8：准备服务器训练版本

这一 PR 的目标是：不是再改算法，而是让你能上传服务器直接跑。

你需要 CC 给你生成：

scripts/server_train_selector_scorer.sh
configs/server/train_scorer.yaml
configs/server/train_selector.yaml
configs/server/train_dual_network.yaml
configs/server/generate_candidate_reward_dataset.yaml
docs/server_training_guide.md

给 Claude Code 的 prompt：

Writing

Continue working in mini-R-EQA.

Goal:
Prepare the project for server-side selector/scorer training.

Assumptions:

Captions and caption embeddings have already been generated.
Do not run captioning.
Do not rebuild caption embeddings.
Server environment will have torch, numpy, sentence-transformers, transformers, and API access for DeepSeek if needed.

Please add server-oriented configs and scripts.

Required files:

configs/server/generate_candidate_reward_dataset.yaml
configs/server/train_scorer.yaml
configs/server/train_selector.yaml
configs/server/train_dual_network.yaml
configs/server/run_selector_inference.yaml
scripts/server_train_selector_scorer.sh
docs/server_training_guide.md

Requirements:

Configs should expose:
prepared root / episode root
caption path
question path
embedding cache dir
embedding model
top_k = 3
runner = deepseek for candidate dataset generation
backend = torch for training
output dirs
batch size
epochs
learning rate
seed
device
Shell script should run in order:
python3 scripts/generate_candidate_reward_dataset.py --config configs/server/generate_candidate_reward_dataset.yaml
python3 scripts/train_scorer.py --config configs/server/train_scorer.yaml
python3 scripts/train_selector.py --config configs/server/train_selector.yaml
python3 scripts/train_dual_network.py --config configs/server/train_dual_network.yaml
python3 scripts/selector_inference_smoke_test.py --config configs/server/run_selector_inference.yaml
Add safety checks:
fail if torch is unavailable
fail if candidate reward dataset does not exist before training
fail if reward variance is zero unless allow_zero_variance_reward: true
fail if embedding files are missing
print selected config values before running
Documentation
docs/server_training_guide.md should explain:
required environment
expected data layout
commands to run
how to verify outputs
how to resume or rerun
what files should be copied back locally
Do not add new model architecture.
Do not run expensive training locally.
Run only compile checks and help commands:
python3 -m compileall mini_eqa scripts
python3 scripts/generate_candidate_reward_dataset.py --help
python3 scripts/train_scorer.py --help
python3 scripts/train_selector.py --help
python3 scripts/train_dual_network.py --help
