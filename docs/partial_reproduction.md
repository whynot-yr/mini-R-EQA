# Partial Reproduction

The partial reproduction runner is meant to validate the experiment chain for:

- R-EQA-style cached dense retrieval with top-3
- uniform sampling with top-10

This is not equivalent to a full R-EQA reproduction.

## Scope

- It runs on prepared episode directories.
- It expects `questions.json` to already exist.
- It expects `captions.json` and cached caption embeddings to be present when running the R-EQA branch.
- It produces internal prediction reports and simple answer-evaluation reports.

## Important Limitations

- Official evaluation still requires the OpenEQA evaluator and LLM-Match protocol.
- Real results require real captions, not `filename_stub`.
- The current answer metrics are sanity checks only.
