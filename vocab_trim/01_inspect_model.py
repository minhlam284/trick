#!/usr/bin/env python3
"""Inspect the model/tokenizer metadata used by vocabulary trimming."""

import argparse

from transformers import AutoConfig, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    text_config = getattr(config, "text_config", config)

    print("Model vocab size:", text_config.vocab_size)
    print("Tokenizer size:", len(tokenizer))
    print("Hidden size:", text_config.hidden_size)
    print("Tied embeddings:", text_config.tie_word_embeddings)
    print("EOS:", tokenizer.eos_token_id)
    print("BOS:", tokenizer.bos_token_id)
    print("PAD:", tokenizer.pad_token_id)
    print("Special IDs:", tokenizer.all_special_ids)


if __name__ == "__main__":
    main()
