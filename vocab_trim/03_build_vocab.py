#!/usr/bin/env python3
"""Build a workload-specific kept-token set from calibration generations."""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Calibration token JSONL; repeat to include exploration output.",
    )
    parser.add_argument("--target-k", type=int, default=16_000)
    parser.add_argument("--output-dir", default="vocab_trim/output")
    return parser.parse_args()


def select_kept_ids(
    *,
    score: Counter[int],
    must_keep: set[int],
    tokenizer_size: int,
    target_k: int,
) -> tuple[list[int], int, int]:
    """Select observed tokens first, then fill low-rank tokenizer IDs."""
    if target_k <= 0 or target_k > tokenizer_size:
        raise ValueError(
            f"target-k must be in [1, {tokenizer_size}], got {target_k}"
        )
    mandatory = sorted(
        token_id for token_id in must_keep if 0 <= token_id < tokenizer_size
    )
    if len(mandatory) > target_k:
        raise ValueError("target-k is smaller than the special-token set")

    selected = list(mandatory)
    selected_set = set(selected)
    for token_id, _ in score.most_common():
        if 0 <= token_id < tokenizer_size and token_id not in selected_set:
            selected.append(token_id)
            selected_set.add(token_id)
            if len(selected) == target_k:
                break

    observed_count = len(selected)
    if len(selected) < target_k:
        # Token IDs follow tokenizer vocabulary/merge rank. Prefer lower IDs as
        # deterministic fallback because they are generally the base/common
        # portion of the vocabulary. Never use random filler.
        for token_id in range(tokenizer_size):
            if token_id in selected_set:
                continue
            selected.append(token_id)
            selected_set.add(token_id)
            if len(selected) == target_k:
                break

    if len(selected) != target_k:
        raise RuntimeError(
            f"Could only select {len(selected)} IDs for target K={target_k}"
        )
    return sorted(selected), observed_count, target_k - observed_count


def main() -> None:
    from transformers import AutoTokenizer

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

    kept_ids, observed_count, filler_count = select_kept_ids(
        score=score,
        must_keep=must_keep,
        tokenizer_size=len(tokenizer),
        target_k=args.target_k,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"kept_ids_{args.target_k}.json"
    output_path.write_text(json.dumps(kept_ids), encoding="utf-8")

    total = sum(output_counts.values())
    covered = sum(output_counts[token_id] for token_id in kept_ids)
    print("Tokenizer vocab:", len(tokenizer))
    print("Requested K:", args.target_k)
    print("Actual kept:", len(kept_ids))
    print("Observed/scored kept:", observed_count)
    print("Deterministic fallback added:", filler_count)
    print("Calibration coverage:", covered / total if total else 0.0)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
