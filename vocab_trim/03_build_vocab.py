#!/usr/bin/env python3
"""Build a workload-specific kept-token set from calibration generations."""

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Calibration token JSONL; repeat to include exploration output.",
    )
    parser.add_argument("--target-k", type=int, default=64_000)
    parser.add_argument("--output-dir", default="vocab_trim/output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.input or [
        "vocab_trim/output/calibration_tokens.jsonl"
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if args.target_k <= 0 or args.target_k > len(tokenizer):
        raise ValueError(
            f"target-k must be in [1, {len(tokenizer)}], got {args.target_k}"
        )

    prompt_counts: Counter[int] = Counter()
    output_counts: Counter[int] = Counter()
    candidate_counts: Counter[int] = Counter()
    row_count = 0
    for input_path in inputs:
        with open(input_path, encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt_counts.update(row["prompt_token_ids"])
                output_counts.update(row["output_token_ids"])
                candidate_counts.update(row["top_candidate_ids"])
                row_count += 1
    if row_count == 0:
        raise ValueError("Calibration token files contain no records")

    score: Counter[int] = Counter()
    for token_id, count in output_counts.items():
        score[token_id] += count * 100.0
    for token_id, count in candidate_counts.items():
        score[token_id] += count
    for token_id, count in prompt_counts.items():
        score[token_id] += count * 0.1

    must_keep = set(tokenizer.all_special_ids)
    for token_id in (
        tokenizer.eos_token_id,
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
    ):
        if token_id is not None:
            must_keep.add(token_id)
    must_keep = {
        token_id for token_id in must_keep if 0 <= token_id < len(tokenizer)
    }
    if len(must_keep) > args.target_k:
        raise ValueError("target-k is smaller than the special-token set")

    ranked_ids = [
        token_id
        for token_id, _ in score.most_common()
        if token_id not in must_keep and 0 <= token_id < len(tokenizer)
    ]
    kept_ids = sorted((list(must_keep) + ranked_ids)[: args.target_k])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"kept_ids_{len(kept_ids)}.json"
    output_path.write_text(json.dumps(kept_ids), encoding="utf-8")

    total = sum(output_counts.values())
    covered = sum(output_counts[token_id] for token_id in kept_ids)
    print("Tokenizer vocab:", len(tokenizer))
    print("Requested K:", args.target_k)
    print("Actual kept:", len(kept_ids))
    print("Calibration coverage:", covered / total if total else 0.0)
    if len(kept_ids) < args.target_k:
        print(
            "WARNING: fewer observed/ranked tokens than target K; "
            "unseen IDs were deliberately not used as filler."
        )
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
