#!/usr/bin/env python3
"""Measure validation coverage and compare unrestricted/restricted decoding."""

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument(
        "--input", default="vocab_trim/validation.jsonl"
    )
    parser.add_argument(
        "--kept-ids", default="vocab_trim/output/kept_ids_64000.json"
    )
    parser.add_argument(
        "--baseline-tokens",
        default="vocab_trim/output/validation_tokens.jsonl",
        help="Optional Step-3-format file used for coverage before generation.",
    )
    parser.add_argument(
        "--output", default="vocab_trim/output/restricted_comparison.jsonl"
    )
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=0.9999)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=("auto", "triton", "flashinfer"),
        default="auto",
        help="Use triton to avoid FlashInfer GDN JIT when nvcc is unavailable.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def body_of(row: dict[str, Any]) -> dict[str, Any]:
    body = row.get("body")
    return body if isinstance(body, dict) else {}


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def messages_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if messages is None:
        messages = body_of(row).get("messages")
    if not isinstance(messages, list):
        raise ValueError("validation row is missing 'messages' or 'body.messages'")
    return messages


def row_id(row: dict[str, Any]) -> Any:
    body = body_of(row)
    return first_present(row.get("id"), body.get("id"), row.get("request_id"))


def report_coverage(
    baseline_path: str, kept: set[int], tokenizer: Any
) -> float:
    total = covered = 0
    missed: list[int] = []
    for row in load_jsonl(baseline_path):
        for token_id in row["output_token_ids"]:
            total += 1
            if token_id in kept:
                covered += 1
            else:
                missed.append(token_id)
    coverage = covered / total if total else 0.0
    print("Validation coverage:", coverage)
    print("Missing occurrences:", len(missed))
    print("Missing unique tokens:", len(set(missed)))
    for token_id in sorted(set(missed)):
        print("MISSING", token_id, repr(tokenizer.decode([token_id])))
    return coverage


def main() -> None:
    args = parse_args()
    kept_ids = json.loads(Path(args.kept_ids).read_text(encoding="utf-8"))
    kept = set(kept_ids)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    special_ids = set(tokenizer.all_special_ids)
    if not special_ids.issubset(kept):
        raise ValueError(
            f"kept vocabulary is missing special IDs: {special_ids - kept}"
        )

    baseline_path = Path(args.baseline_tokens)
    coverage = None
    if baseline_path.exists():
        coverage = report_coverage(str(baseline_path), kept, tokenizer)
        print(
            "Accuracy gate:",
            "PASS" if coverage >= args.min_coverage else "FAIL",
            f"(required >= {args.min_coverage})",
        )
    elif args.coverage_only:
        raise FileNotFoundError(baseline_path)
    if args.coverage_only:
        if coverage is not None and coverage < args.min_coverage:
            raise SystemExit(2)
        return

    rows = load_jsonl(args.input)
    if not rows:
        raise ValueError(f"No validation requests found in {args.input}")
    prompt_ids = [
        tokenizer.apply_chat_template(
            messages_of(row), tokenize=True, add_generation_prompt=True
        )
        for row in rows
    ]
    prompts = [
        tokenizer.apply_chat_template(
            messages_of(row), tokenize=False, add_generation_prompt=True
        )
        for row in rows
    ]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gdn_prefill_backend=args.gdn_prefill_backend,
    )
    common = dict(temperature=0.0, max_tokens=args.max_tokens, seed=42)
    baseline_outputs = llm.generate(
        prompts, SamplingParams(**common), use_tqdm=True
    )
    restricted_outputs = llm.generate(
        prompts,
        SamplingParams(**common, allowed_token_ids=kept_ids),
        use_tqdm=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identical_count = empty_count = eos_missing_count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for row, baseline, restricted in zip(
            rows, baseline_outputs, restricted_outputs, strict=True
        ):
            baseline_completion = baseline.outputs[0]
            restricted_completion = restricted.outputs[0]
            baseline_ids = list(baseline_completion.token_ids)
            restricted_ids = list(restricted_completion.token_ids)
            identical = baseline_ids == restricted_ids
            identical_count += int(identical)
            empty_count += int(not restricted_ids)
            if (
                tokenizer.eos_token_id in baseline_ids
                and tokenizer.eos_token_id not in restricted_ids
            ):
                eos_missing_count += 1
            record = {
                "id": row_id(row),
                "identical": identical,
                "baseline_token_ids": baseline_ids,
                "restricted_token_ids": restricted_ids,
                "baseline_text": baseline_completion.text,
                "restricted_text": restricted_completion.text,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Requests:", len(rows))
    print("Greedy exact-match:", identical_count / len(rows))
    print("Restricted empty outputs:", empty_count)
    print("Restricted outputs missing baseline EOS:", eos_missing_count)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
