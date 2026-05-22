from __future__ import annotations

from mini_eqa.runners.mock_runner import mock_answer


RUNNER_NAMES = ["mock", "deepseek", "openai_compatible", "llama_local"]


def get_runner(name: str):
    if name == "mock":
        return mock_answer
    if name == "deepseek":
        from mini_eqa.runners.deepseek_runner import deepseek_answer

        return deepseek_answer
    if name in {"openai_compatible", "llama_local"}:
        from mini_eqa.runners.openai_compatible_runner import openai_compatible_answer

        return openai_compatible_answer
    raise ValueError(f"Unsupported runner: {name}")
