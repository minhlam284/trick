# Vocabulary trimming experiment

This directory keeps vocabulary trimming isolated from scheduler work. Run every
model command inside the target Docker image, with the API server stopped, and
from the repository root.

## Data contract

Each non-empty line in `trace.jsonl` can be either:

- Flat chat JSON with `messages` plus `id` or `conversation_id`.
- Trace-envelope JSON with `request_id` and chat payload under `body.messages`,
  like the official workload traces.

Never use validation generations to build a kept vocabulary.

## Reproducible run

```bash
python3 vocab_trim/00_split_trace.py
python3 vocab_trim/01_inspect_model.py
mkdir -p vocab_trim/output

# Calibration: greedy plus a separate exploration pass.
python3 vocab_trim/02_collect_tokens.py
python3 vocab_trim/02_collect_tokens.py --exploration

# If --output points at a directory, the script writes
# <input_stem>_tokens.jsonl inside that directory.
python3 vocab_trim/02_collect_tokens.py \
  --model /home/coder/data/vocab/qwen3.5 \
  --input /home/coder/data/vocab/trick/vocab_trim/calibration.jsonl \
  --output /home/coder/data/vocab/output

# Build each candidate only from calibration artifacts.
for k in 128000 96000 64000 32000; do
  python3 vocab_trim/03_build_vocab.py \
    --input vocab_trim/output/calibration_tokens.jsonl \
    --input vocab_trim/output/calibration_exploration_tokens.jsonl \
    --target-k "$k"
done

# Baseline validation generation is for evaluation only.
python3 vocab_trim/02_collect_tokens.py \
  --input vocab_trim/validation.jsonl

# Coverage gate without loading vLLM's model, then exact restricted decoding.
python3 vocab_trim/04_test_allowed_vocab.py --coverage-only
python3 vocab_trim/04_test_allowed_vocab.py

# Run on the exact MIG/profile used by the service.
python3 vocab_trim/05_bench_lm_head.py
```

The coverage gate is at least `0.9999` for equivalent accuracy; exact greedy
reproduction requires `1.0`. Restricted decoding is only a correctness test:
`allowed_token_ids` masks already-computed logits and does not make the LM head
faster.

## Runtime modes

Keep deployment selection explicit in the eventual optimized integration:

```text
TRIM_VOCAB=0  baseline full LM head
TRIM_VOCAB=1  optimized gathered/trimmed LM head
```

This experiment does not claim an optimized vLLM patch. Only implement that
patch after the accuracy gates pass and the LM-head microbenchmark shows a clear
win at the service's real batch sizes. The baseline mode must remain unchanged
in the same image for A/B testing and rollback.
