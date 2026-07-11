#!/usr/bin/env python3
"""Generate with the baseline model and collect token IDs and candidates."""

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
        "--input", default="vocab_trim/calibration.jsonl"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--exploration", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=("auto", "triton", "flashinfer"),
        default="triton",
        help="Use triton to avoid FlashInfer GDN JIT when nvcc is unavailable.",
    )
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages_of(row, path, line_number)
            rows.append(row)
    if not rows:
        raise ValueError(f"No requests found in {path}")
    return rows


def body_of(row: dict[str, Any]) -> dict[str, Any]:
    body = row.get("body")
    return body if isinstance(body, dict) else {}


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def messages_of(
    row: dict[str, Any], path: str = "<row>", line_number: int = 0
) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if messages is None:
        messages = body_of(row).get("messages")
    if not isinstance(messages, list):
        location = f"{path}:{line_number}" if line_number else path
        raise ValueError(f"{location}: missing 'messages' or 'body.messages'")
    return messages


def row_id(row: dict[str, Any]) -> Any:
    body = body_of(row)
    return first_present(row.get("id"), body.get("id"), row.get("request_id"))


def conversation_id(row: dict[str, Any]) -> Any:
    body = body_of(row)
    return first_present(row.get("conversation_id"), body.get("conversation_id"))


def resolve_output_path(output: str | None, input_path: str, exploration: bool) -> Path:
    suffix = "_exploration" if exploration else ""
    input_stem = Path(input_path).stem
    filename = f"{input_stem}{suffix}_tokens.jsonl"
    if output is None:
        return Path("vocab_trim/output") / filename
    output_path = Path(output)
    if output_path.exists() and output_path.is_dir():
        return output_path / filename
    if output_path.suffix == "":
        return output_path / filename
    return output_path


def candidate_ids(logprobs: Any) -> list[int]:
    ids: list[int] = []
    for step in logprobs or []:
        if step:
            ids.extend(int(token_id) for token_id in step)
    return ids


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output, args.input, args.exploration)
    rows = load_rows(args.input)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    prompt_token_ids = [
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
    sampling_kwargs = {
        "temperature": 0.7 if args.exploration else 0.0,
        "max_tokens": args.max_tokens,
        "seed": 42,
        "logprobs": 20,
    }
    if args.exploration:
        sampling_kwargs.update(n=4, top_p=0.95)
    outputs = llm.generate(
        prompts, SamplingParams(**sampling_kwargs), use_tqdm=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row, prompt_ids, request_output in zip(
            rows, prompt_token_ids, outputs, strict=True
        ):
            # Exploration has n=4. Preserve every branch so all sampled and
            # top-candidate tokens contribute to vocabulary construction.
            completions = request_output.outputs
            record = {
                "id": row_id(row),
                "conversation_id": conversation_id(row),
                "prompt_token_ids": prompt_ids,
                "output_token_ids": [
                    int(token_id)
                    for completion in completions
                    for token_id in completion.token_ids
                ],
                "top_candidate_ids": [
                    token_id
                    for completion in completions
                    for token_id in candidate_ids(completion.logprobs)
                ],
                "output_text": [completion.text for completion in completions],
                "exploration": args.exploration,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} records to {output_path}")


if __name__ == "__main__":
    main()
