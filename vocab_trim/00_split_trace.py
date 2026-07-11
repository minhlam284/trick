#!/usr/bin/env python3
"""Split trace requests by conversation, never by individual request."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="vocab_trim/trace.jsonl")
    parser.add_argument(
        "--calibration", default="vocab_trim/calibration.jsonl"
    )
    parser.add_argument(
        "--validation", default="vocab_trim/validation.jsonl"
    )
    parser.add_argument("--calibration-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_ratio < 1.0:
        raise ValueError("calibration-ratio must be between 0 and 1")

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    with open(args.input, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            group_id = row.get("conversation_id", row.get("id"))
            if group_id is None:
                raise ValueError(
                    f"{args.input}:{line_number}: needs conversation_id or id"
                )
            groups[group_id].append(row)
    if len(groups) < 2:
        raise ValueError("At least two conversation groups are required")

    group_ids = list(groups)
    random.Random(args.seed).shuffle(group_ids)
    cut = int(len(group_ids) * args.calibration_ratio)
    cut = max(1, min(cut, len(group_ids) - 1))
    calibration_groups = set(group_ids[:cut])
    calibration: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for group_id, rows in groups.items():
        target = calibration if group_id in calibration_groups else validation
        target.extend(rows)

    write_jsonl(args.calibration, calibration)
    write_jsonl(args.validation, validation)
    print("Conversation groups:", len(groups))
    print("Calibration requests:", len(calibration))
    print("Validation requests:", len(validation))
    print("Calibration:", args.calibration)
    print("Validation:", args.validation)


if __name__ == "__main__":
    main()
