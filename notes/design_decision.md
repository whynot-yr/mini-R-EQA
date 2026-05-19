# Design Decisions

## Refactor into R-EQA-style structure

The original `v0.1-TF-IDF` implementation used a single file:

```text
src/run_demo.py
```

This file handled data loading, TF-IDF retrieval, prompt construction, and mock answer generation together.

This was acceptable for the first demo, but it was not ideal for extension. If new retrieval methods such as SBERT or uniform sampling are added directly into `run_demo.py`, the file will become hard to maintain.

Therefore, the project is refactored into a simplified R-EQA-style structure:

```
mini_eqa/baselines/    runnable baselines
mini_eqa/retrieval/    retrieval methods
mini_eqa/runners/      answer generation runners
mini_eqa/utils/        shared utilities
prompts/               prompt templates
reports/               saved experiment outputs
notes/                 development and experiment notes
```

The current TF-IDF pipeline is now organized as:

```
mini_eqa/baselines/tfidf_rag.py
mini_eqa/retrieval/tfidf.py
mini_eqa/runners/mock_runner.py
mini_eqa/utils/io_utils.py
mini_eqa/utils/prompt_utils.py
prompts/mini_rag.txt
```

This refactor does not change the retrieval behavior. It only makes the codebase easier to extend.

## Why use `mini_eqa` instead of `openeqa`

The original R-EQA repository is built on top of OpenEQA and uses the package name `openeqa`.

This mini project uses `mini_eqa` instead of `openeqa` to avoid import conflicts with the real OpenEQA/R-EQA codebase.

## Why use `runners/mock_runner.py`

The current version does not call a real LLM. It uses a mock answer generator to verify that retrieval and prompt construction work.

Later, this can be extended to:

```
mini_eqa/runners/llm_runner.py
mini_eqa/runners/openai_runner.py
mini_eqa/runners/local_llama_runner.py
```

