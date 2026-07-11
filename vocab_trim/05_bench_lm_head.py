#!/usr/bin/env python3
"""Microbenchmark full and trimmed LM-head matrix multiplication."""

import argparse

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--full-vocab", type=int, default=248_320)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[248_320, 128_000, 96_000, 64_000, 32_000]
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32]
    )
    return parser.parse_args()


def benchmark(
    weight: torch.Tensor,
    batch_size: int,
    hidden_size: int,
    warmup: int,
    iterations: int,
) -> float:
    hidden = torch.randn(
        batch_size,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    for _ in range(warmup):
        F.linear(hidden, weight)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        F.linear(hidden, weight)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if max(args.ks) > args.full_vocab:
        raise ValueError("Every K must be <= full-vocab")
    torch.manual_seed(42)
    full_weight = torch.randn(
        args.full_vocab,
        args.hidden,
        dtype=torch.bfloat16,
        device="cuda",
    )
    for k in args.ks:
        weight = full_weight[:k].contiguous()
        for batch_size in args.batch_sizes:
            latency = benchmark(
                weight,
                batch_size,
                args.hidden,
                args.warmup,
                args.iterations,
            )
            print(
                f"K={k:6d} batch={batch_size:2d} latency={latency:.4f} ms"
            )


if __name__ == "__main__":
    main()
