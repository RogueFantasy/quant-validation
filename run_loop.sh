#!/usr/bin/env bash
set -u

VENV=~/llm/bin/activate
BASE_URL=http://localhost:8000/v1/completions
MODELS_URL=http://localhost:8000/v1/models
SWEEP_DIR=~/quant-validation/sweep-results
TOKENIZER=Qwen/Qwen2.5-Coder-1.5B-Instruct

declare -A QUANT_FLAGS=(
  [awq]="--quantization awq --dtype float16"
  [gptq]="--quantization gptq --dtype float16"
  [gguf]="--tokenizer $TOKENIZER"
  [fp16]=""
)

# wait until the server is up AND serving the model we expect
wait_for_server() {
  local want="$1"
  for i in $(seq 1 60); do
    got=$(curl -sf "$MODELS_URL" 2>/dev/null | grep -o "$want")
    if [ -n "$got" ]; then echo "ready: $want"; return 0; fi
    sleep 5
  done
  echo "TIMEOUT waiting for $want"; return 1
}

# kill any vllm server and wait for the GPU to clear
teardown() {
  pkill -f "vllm serve" 2>/dev/null
  sleep 8
  pkill -9 -f "vllm serve" 2>/dev/null
  sleep 3
  echo "teardown done"
}

# start a server for one model, picking flags by type
serve_model() {
  local model="$1"
  local common="--max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager"
  source "$VENV"
  export VLLM_USE_FLASHINFER_SAMPLER=0
  local fmt
  case "$model" in
    *.gguf)  fmt=gguf ;;
    *awq*)   fmt=awq ;;
    *gptq*)  fmt=gptq ;;
    *)       fmt=fp16 ;;
  esac
  vllm serve "$model" ${QUANT_FLAGS[$fmt]} $common > /tmp/vllm.log 2>&1 &
}

# ---- config: what to test ----
MODELS=(
  "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  "/home/user/qwen-1.5b-awq"
  "/home/user/gguf/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"
)

SEEDS=(1234 42 7 0 100)
TASKS=(gsm8k humaneval_instruct)

# ---- main loop ----
mkdir -p "$SWEEP_DIR"
source "$VENV"
export HF_ALLOW_CODE_EVAL=1

for model in "${MODELS[@]}"; do
  echo "===== $model ====="
  teardown
  serve_model "$model"
  wait_for_server "$model" || { echo "SKIP $model (never came up)"; continue; }

  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "--- $task seed=$seed ---"
      lm_eval --model local-completions \
        --model_args "model=$model,base_url=$BASE_URL,tokenizer=$TOKENIZER,num_concurrent=4" \
        --tasks "$task" \
        --apply_chat_template \
        --seed "0,1234,1234,$seed" \
        --confirm_run_unsafe_code \
        --output_path "$SWEEP_DIR" \
        || echo "FAIL $model $task seed=$seed"
    done
  done
done

teardown
echo "===== sweep complete ====="
