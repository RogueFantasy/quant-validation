# quant-validation

A small, reproducible harness for measuring how much quality a quantized LLM actually keeps versus its full-precision baseline — run on a single consumer GPU (RTX 3070, 8GB).

Quantization shrinks a model so it runs cheaper and faster. The open question this measures: **how much capability does it cost?** The widely-circulated "AWQ keeps ~95%" style numbers are mostly ungrounded (no named model, no named eval, no error bars) and derived from large models. This harness produces named, error-barred, reproducible retention numbers for *small* models and the *community quant formats people actually download*.

## First result

Qwen2.5-Coder-1.5B-Instruct, GSM8K, 200 questions, 5-shot, greedy:

| Model | flexible-extract | strict-match | Retention (flex) |
|---|---|---|---|
| Full (BF16 baseline) | 0.505 ±0.035 | 0.465 ±0.035 | 100% (ref) |
| AWQ 4-bit | 0.390 ±0.035 | 0.335 ±0.034 | **77%** |

The 4-bit AWQ model kept ~77% of the baseline's GSM8K performance — a real, statistically-significant drop (well outside the error bars), and notably worse than the blanket "~95%" claims. (Note: this is a *math* eval on a *coding* model; a coding eval — HumanEval/MBPP — is the next measurement.)

## What it does

```
get a model  →  serve it (vLLM)  →  eval it (lm-eval)  →  retention = quant / baseline
```

- `make_awq.py` — generates a 4-bit AWQ quant of the baseline model.
- `sweep.sh` — the eval harness: given an already-running vLLM server, runs the eval suite, saves each result, and skips runs that already completed cleanly (resumable via `.done` sentinels). Includes a preflight server check and writes an audit file recording what the server claimed to be serving.

## Reproducibility

Every result is produced with a fixed configuration so comparisons are valid:

- **Harness:** lm-evaluation-harness 0.4.12 (pinned)
- **Decoding:** greedy (temperature 0.0)
- **Seeds:** fixed (random 0, numpy/torch/fewshot 1234)
- **Serving:** vLLM, `--enforce-eager`
- Same eval config held constant across every model; only the quantization method varies.

## Usage

```bash
# 1. serve a model (FlashInfer sampler disabled to avoid a compile error)
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve <model-path> --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager
#   for an AWQ model, add: --quantization awq --dtype float16

# 2. run the sweep against the running server
./sweep.sh <model-path>

# results land in ./sweep-results/<model>_<task>/
```

## Hardware reality (RTX 3070, 8GB)

The 8GB ceiling shapes the project. The 1.5B baseline fits comfortably at full precision; the 3B does **not** (weights ~5.79GB leave negative room for the KV cache once ~900MB of desktop/Xwayland overhead is counted). So 1.5B is the practical full-precision baseline on this card — which is fine, since small-model quantization is exactly the under-measured regime this targets.

## Status

Early but real. The eval-sweep harness works end-to-end and reproduces its results. Next: add a coding eval (HumanEval/MBPP), a retention-computation script, and more quant formats (GGUF, GPTQ) for a method-vs-method comparison at the same bit-width.
