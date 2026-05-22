from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def record(rows: list[dict], name: str, status: str, message: str) -> None:
    rows.append({"name": name, "status": status, "message": message})


def main() -> None:
    rows: list[dict] = []
    fail_count = 0

    try:
        from mini_eqa.captioning.backends import get_captioner
        from mini_eqa.runners.registry import get_runner
        from mini_eqa.evaluation.run_predictions import run_prediction_pipeline
        import mini_eqa.exporters.openeqa_predictions  # noqa: F401

        record(rows, "core-imports", "PASS", "Core imports available.")
    except Exception as exc:
        record(rows, "core-imports", "FAIL", str(exc))
        fail_count += 1
        get_captioner = None
        get_runner = None
        run_prediction_pipeline = None

    if get_runner is not None:
        for runner_name in ("mock", "openai_compatible", "llama_local"):
            try:
                get_runner(runner_name)
                record(rows, f"runner:{runner_name}", "PASS", "Runner resolved.")
            except Exception as exc:
                record(rows, f"runner:{runner_name}", "FAIL", str(exc))
                fail_count += 1

    if get_captioner is not None:
        try:
            get_captioner("filename_stub")
            record(rows, "captioner:filename_stub", "PASS", "Captioner resolved.")
        except Exception as exc:
            record(rows, "captioner:filename_stub", "FAIL", str(exc))
            fail_count += 1

        try:
            importlib.import_module("mini_eqa.captioning.backends.qwen_vl")
            record(rows, "captioner:qwen_vl_module", "PASS", "Qwen-VL module importable.")
        except Exception as exc:
            record(
                rows,
                "captioner:qwen_vl_module",
                "WARN",
                f"Qwen-VL module import warning: {exc}",
            )

    for req_path in ("requirements.txt", "requirements-vlm.txt", "requirements-llama.txt"):
        path = REPO_ROOT / req_path
        if path.exists():
            record(rows, f"file:{req_path}", "PASS", "File exists.")
        else:
            record(rows, f"file:{req_path}", "FAIL", "Missing file.")
            fail_count += 1

    for script_path in (
        "scripts/validate_openeqa_data.py",
        "scripts/caption_prepared_episodes.py",
        "scripts/build_embeddings_for_prepared.py",
        "scripts/run_partial_reproduction.py",
        "scripts/check_reproduction_status.py",
        "scripts/evaluate_with_openeqa.py",
    ):
        path = REPO_ROOT / script_path
        if path.exists():
            record(rows, f"file:{script_path}", "PASS", "File exists.")
        else:
            record(rows, f"file:{script_path}", "FAIL", "Missing file.")
            fail_count += 1

    for config_path in (
        "configs/reqa_small_qwen_llama4bit.yaml",
        "configs/reqa_full_qwen_llama4bit.yaml",
        "configs/uniform_full_qwen_llama4bit.yaml",
    ):
        path = REPO_ROOT / config_path
        if path.exists():
            record(rows, f"file:{config_path}", "PASS", "File exists.")
        else:
            record(rows, f"file:{config_path}", "FAIL", "Missing file.")
            fail_count += 1

    print("=" * 100)
    print(f"{'Check':40} {'Status':8} Message")
    print("=" * 100)
    for row in rows:
        print(f"{row['name'][:40]:40} {row['status']:8} {row['message']}")

    if fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
