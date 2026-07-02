# quant-validation

A reproducible benchmark for how much capability quantized LLMs actually keep — run on a single consumer GPU (RTX 3070, 8GB), across quant methods and distinct capabilities.

Quantization shrinks a model so it runs cheaper and faster. The open question this measures, systematically and with error bars: **how much capability does it cost?** The widely-circulated "AWQ keeps ~95%" numbers are mostly ungrounded — no named model, no named eval, no seeds, no error bars — and derived from large models. This project measures the corner nobody benchmarks rigorously: **the community quant formats people actually download (AWQ / GGUF), on small coding models, across math and coding, with multi-seed error bars.**

## Current results

Model under test: **Qwen2.5-Coder-1.5B-Instruct**. Retention = quant score / full-precision baseline. Each number is the **mean of 5 seeds** (varying the few-shot seed), greedy decoding, full datasets. Std shown.

| Method | Bits | GSM8K (math) | Retention | HumanEval (coding) | Retention |
|---|---|---|---|---|---|
| Full (fp16 baseline) | 16 | 0.494 ±0.011 | 100% | 0.632 ±0.005 | 100% |
| GGUF Q4_K_M | 4 | 0.478 ±0.004 | **97%** | 0.621 ±0.003 | **98%** |
| AWQ 4-bit | 4 | 0.446 ±0.007 | **90%** | 0.591 ±0.000 | **94%** |

### Findings

**1. At the same 4-bit width, GGUF beats AWQ.** GGUF retains 97-98% on both tasks; AWQ 90-94%. "4-bit is 4-bit" is false — two 4-bit quants of the same model differ by up to 7 points, and the error bars (std 0.004-0.011) are far smaller than the gap. The method matters more than the bit count.

**2. Quantization degrades capabilities unevenly; AWQ's weak spot is math.** AWQ costs ~6% of coding ability but ~10% of math; GGUF stays near-lossless on both. Capability matters, not just bit-width.

*Caveats: single base model, greedy decoding, variance is few-shot-selection variance. Treat sub-2-point gaps as noise.*

## What it does

get a model -> serve it (vLLM) -> eval it (lm-eval, N seeds) -> retention = quant / baseline

- `run_loop.sh` — automated sweep: for each model, tears down any running server, serves with the right flags for its format (AWQ / GGUF / GPTQ / fp16 via lookup table), waits until serving, runs every task at every seed, tears down, next. Skip-and-log on failure.
- `retention.py` — reads result JSONs, aggregates across seeds, prints the grid above.
- `make_awq.py` — generates a 4-bit AWQ quant.

## Reproducibility

- Harness: lm-evaluation-harness 0.4.12 (pinned)
- Decoding: greedy (temperature 0.0)
- Seeds: 5 (few-shot seed varied: 1234, 42, 7, 0, 100)
- Serving: vLLM, `--enforce-eager`, `--max-model-len 4096`
- Same eval config across every model; only quantization varies.

## Hardware reality (RTX 3070, 8GB)

The 1.5B baseline fits at full precision; 3B does not (weights ~5.79GB leave no room for KV cache after ~900MB desktop overhead). So 1.5B is the practical baseline — fine, since small-model quantization is the under-measured regime this targets.

## Roadmap

- More methods: GGUF at other bit-widths (Q3/Q5/Q8), GPTQ, EXL2.
- More capabilities: MBPP, MMLU.
- Throughput / cost axis: tokens/sec and VRAM per method.
- Community quants: validate what people actually download, by popularity.
