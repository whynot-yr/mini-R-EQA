from __future__ import annotations

from mini_eqa.runners.mock_runner import mock_answer


RUNNER_NAMES = ["mock", "deepseek"]


def get_runner(name: str):
    if name == "mock":
        return mock_answer
    if name == "deepseek":
        from mini_eqa.runners.deepseek_runner import deepseek_answer

        return deepseek_answer
    raise ValueError(f"Unsupported runner: {name}")
