from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.evaluation.run_predictions import print_prediction_summary, run_prediction_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a mini-R-EQA prediction pipeline from a YAML config file."
    )
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping.")

    required_keys = {
        "episode_dir",
        "retriever",
        "runner",
        "top_k",
        "prompt",
        "model",
        "max_output_tokens",
        "output",
    }
    missing_keys = sorted(key for key in required_keys if key not in data)
    if missing_keys:
        raise KeyError(
            f"Config file {config_path} is missing required keys: {missing_keys}"
        )

    return data


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = load_config(config_path)
    report = run_prediction_pipeline(
        episode_dir=config["episode_dir"],
        retriever=config["retriever"],
        runner=config["runner"],
        top_k=config["top_k"],
        cache_dir=config.get("cache_dir"),
        prompt_name=config["prompt"],
        model=config["model"],
        max_output_tokens=config["max_output_tokens"],
        output=config["output"],
        limit=config.get("limit"),
    )
    print_prediction_summary(report, config["output"])


if __name__ == "__main__":
    main()
