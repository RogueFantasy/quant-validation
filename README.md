# quant-validation

A continuously-updated benchmark for how much quality quantized LLMs actually keep — run on a single consumer GPU (RTX 3070, 8GB), growing as new models and quant formats are validated.

Quantization shrinks a model so it runs cheaper and faster. The open question this measures, systematically and reproducibly: **how much capability does it cost?** The widely-circulated "AWQ keeps ~95%" style numbers are mostly ungrounded — no named model, no named eval, no error bars — and derived from large models. This project is the missing validation layer: point it at any quantized checkpoint, get trustworthy, error-barred retention numbers, across methods and capabilities. **The dataset grows over time; the tables below are a current snapshot, not a finished study.**

## What this is (and isn't)

This is **not** a one-off experiment. It's a reusable harness plus a living results dataset:

- **The harness** — generate or download a quant, serve it, run a fixed eval suite, record retention. Resumable, reproducible, designed to keep absorbing new models and formats.
- **The dataset** — retention numbers that accumulate as more quants (AWQ / GGUF / GPTQ / EXL2 at various bit-widths) and more capabilities (math, coding, reasoning) get measured. New rows land as they're run.

The goal is the corner of the space nobody benchmarks rigorously: **the community quant formats people actually download, on small models, across distinct capabilities.**

## Current results (snapshot — updated as runs complete)

Model under test: Qwen2.5-Coder-1.5B-Instruct. Retention = quant score / full-precision baseline.

**Coding — HumanEval pass@1, 0-shot:**

| Method | Bits | pass@1 | Retention |
|---|---|---|---|
| Full (BF16 baseline) | 16 | 0.628 ±0.038 | 100% (ref) |
| AWQ 4-bit | 4 | 0.591 ±0.039 | **94%** |

**Math — GSM8K, 5-shot:**

| Method | Bits | flexible-extract | Retention |
|---|---|---|---|
| Full (BF16 baseline) | 16 | 0.505 ±0.035 | 100% (ref) |
| AWQ 4-bit | 4 | 0.390 ±0.035 | **77%** |

### Headline finding so far: quantization degrades capabilities *unevenly*

Same model, same 4-bit AWQ quant — but it costs ~6% of **coding** ability and ~23% of **math** ability. "AWQ keeps ~95%" is true *for coding* and false *for math* on the same checkpoint. **The capability matters as much as the bit-width** — which is exactly the nuance the blanket numbers hide. (Single-seed so far; the coding drop is within ~1 stderr, the math drop is unambiguous. Multi-seed validation in progress.)

## What it does

```
get a model  →  serve it (vLLM)  →  eval it (lm-eval)  →  retention = quant / baseline
```

- `make_awq.py` — generates a 4-bit AWQ quant of a baseline model.
- `sweep.sh` — the eval harness: given a running vLLM server, runs the eval suite, saves each result, and skips runs that already completed cleanly (resumable via `.done` sentinels). Includes a preflight server check and an audit file recording what was served. New evals/models are added by extending the task and model lists.

## Reproducibility

Every result is produced with a fixed configuration so comparisons stay valid as the dataset grows:

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
#   coding evals also need: export HF_ALLOW_CODE_EVAL=1   (executes generated code)

# results land in ./sweep-results/<model>_<task>/
```

## Hardware reality (RTX 3070, 8GB)

The 8GB ceiling shapes the project. The 1.5B baseline fits comfortably at full precision; the 3B does **not** (weights ~5.79GB leave negative room for the KV cache once ~900MB of desktop/Xwayland overhead is counted). So 1.5B is the practical full-precision baseline on this card — which is fine, since small-model quantization is exactly the under-measured regime this targets.

## Roadmap

The dataset is actively expanding. Next:

- **More methods** — GGUF (Q3/Q4_K_M/Q5/Q8), GPTQ, EXL2 — to compare methods at the same bit-width and find where low-bit quality falls off a cliff.
- **More capabilities** — MBPP (coding), MMLU (knowledge) — to extend the uneven-degradation map.
- **Community quants** — validate the actual checkpoints people download (bartowski et al.), prioritized by download popularity × how unmeasured they are.
- **Automation** — orchestrate the serve→eval→record loop to run continuously as new quants are published.
