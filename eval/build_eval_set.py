from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_humaneval_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["task_id"],
        "dataset": "humaneval",
        "prompt": row["prompt"],
        "entry_point": row.get("entry_point"),
        "tests": row["test"],
        "test_setup_code": "",
        "timeout_sec": 3,
    }


def convert_mbpp_row(row: dict[str, Any]) -> dict[str, Any]:
    test_list = row.get("test_list", [])
    test_setup_code = row.get("test_setup_code", "")

    return {
        "id": f"mbpp_{row['task_id']}",
        "dataset": "mbpp",
        "prompt": row["text"],
        "entry_point": None,
        "tests": "\n".join(test_list),
        "test_setup_code": test_setup_code,
        "timeout_sec": 3,
    }


def build_eval_set(
    dataset_name: str,
    split: str,
    dataset_type: str,
) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)

    converted_rows: list[dict[str, Any]] = []

    for row in ds:
        if dataset_type == "humaneval":
            converted_rows.append(convert_humaneval_row(row))
        elif dataset_type == "mbpp":
            converted_rows.append(convert_mbpp_row(row))
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    return converted_rows


def main(args: argparse.Namespace) -> None:
    rows = build_eval_set(
        dataset_name=args.dataset_name,
        split=args.split,
        dataset_type=args.dataset_type,
    )

    output_path = Path(args.output_path)
    write_jsonl(rows, output_path)

    print(f"Saved {len(rows)} tasks to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build fixed eval JSONL from Hugging Face datasets"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split to use",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["humaneval", "mbpp"],
        help="Converter type for the dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output JSONL file",
    )

    args = parser.parse_args()
    main(args)
