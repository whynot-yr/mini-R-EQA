from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_eqa.preprocess.build_caption_embeddings import DEFAULT_MODEL_NAME
from mini_eqa.utils.io_utils import load_json, save_json
from scripts._prepared_episode_utils import (
    count_image_files,
    expected_embedding_paths,
    list_prepared_episode_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check reproduction status for prepared episodes."
    )
    parser.add_argument("--prepared_root", type=str, required=True)
    parser.add_argument("--reports_root", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def build_markdown(rows: list[dict]) -> str:
    lines = [
        "# Reproduction Status",
        "",
        "| episode | status | has_captions | has_embeddings | has_reqa_predictions | has_uniform_predictions | message |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['episode']} | {row['status']} | {row['has_captions']} | "
            f"{row['has_embeddings']} | {row['has_reqa_predictions']} | "
            f"{row['has_uniform_predictions']} | {row['message']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    prepared_root = Path(args.prepared_root)
    reports_root = Path(args.reports_root)
    output_path = Path(args.output)

    if not prepared_root.exists():
        raise FileNotFoundError(f"Prepared root does not exist: {prepared_root}")

    rows = []
    for episode_dir in list_prepared_episode_dirs(prepared_root):
        episode = episode_dir.name
        questions_path = episode_dir / "questions.json"
        episode_meta_path = episode_dir / "episode_meta.json"
        captions_path = episode_dir / "captions.json"
        embeddings_path, metadata_path = expected_embedding_paths(
            episode_dir, args.embedding_model
        )
        reqa_predictions = reports_root / f"predictions_reqa_{episode}.json"
        uniform_predictions = reports_root / f"predictions_uniform_{episode}.json"
        reqa_eval = reports_root / f"answer_eval_reqa_{episode}.json"
        uniform_eval = reports_root / f"answer_eval_uniform_{episode}.json"

        frames_dir_value = None
        if episode_meta_path.exists():
            meta = load_json(episode_meta_path)
            frames_dir_value = meta.get("frames_dir")

        num_captions = 0
        if captions_path.exists():
            try:
                num_captions = len(load_json(captions_path))
            except Exception:
                num_captions = 0

        has_embeddings = embeddings_path.exists() and metadata_path.exists()
        has_reqa_predictions = reqa_predictions.exists()
        has_uniform_predictions = uniform_predictions.exists()
        has_reqa_eval = reqa_eval.exists()
        has_uniform_eval = uniform_eval.exists()

        status = "ready"
        message = "OK"
        if not questions_path.exists():
            status = "missing_questions"
            message = "questions.json is missing."
        elif not episode_meta_path.exists():
            status = "missing_episode_meta"
            message = "episode_meta.json is missing."
        elif not captions_path.exists():
            status = "missing_captions"
            message = "captions.json is missing."
        elif not has_embeddings:
            status = "missing_embeddings"
            message = "caption embeddings are missing."
        elif not has_reqa_predictions or not has_uniform_predictions:
            status = "missing_predictions"
            message = "prediction reports are missing."
        elif not has_reqa_eval or not has_uniform_eval:
            status = "missing_eval"
            message = "answer eval reports are missing."

        rows.append(
            {
                "episode": episode,
                "has_questions": questions_path.exists(),
                "has_episode_meta": episode_meta_path.exists(),
                "frames_dir": frames_dir_value,
                "has_captions": captions_path.exists(),
                "num_captions": num_captions,
                "has_embeddings": has_embeddings,
                "has_reqa_predictions": has_reqa_predictions,
                "has_uniform_predictions": has_uniform_predictions,
                "has_reqa_eval": has_reqa_eval,
                "has_uniform_eval": has_uniform_eval,
                "status": status,
                "message": message,
            }
        )

    report = {
        "prepared_root": str(prepared_root),
        "reports_root": str(reports_root),
        "rows": rows,
    }
    save_json(report, output_path)
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(build_markdown(rows), encoding="utf-8")

    print("=" * 80)
    print("Check Reproduction Status")
    print("=" * 80)
    print(f"Prepared root: {prepared_root}")
    print(f"Reports root: {reports_root}")
    print(f"Saved JSON report to: {output_path}")
    print(f"Saved markdown report to: {markdown_path}")


if __name__ == "__main__":
    main()
