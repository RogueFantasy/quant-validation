#!/bin/bash
# Incremental eval sweep. Assumes a vLLM server is ALREADY running
# for the model you're testing, on localhost:8000.
# Loops a model over multiple evals, saves each result, and skips
# runs that previously completed cleanly (tracked via a .done sentinel).

set -u  # error on unset variables (not -e: we want to handle failures ourselves)

MODEL="$1"   # pass the model name/path as the first argument
if [ -z "${MODEL:-}" ]; then
  echo "Usage: ./sweep.sh <model-name-or-path>"
  exit 1
fi

# --- Config (override via env, e.g. LIMIT=0 ./sweep.sh my-model) ---------------
BASE_URL="${BASE_URL:-http://localhost:8000}"
LIMIT="${LIMIT:-200}"            # 0 = no limit (full set, slower, comparable scores)
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
NUM_CONCURRENT="${NUM_CONCURRENT:-4}"
# Chat template: ON for instruct/chat models, OFF for base models.
# Set CHAT_TEMPLATE=0 when evaluating a base model.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-1}"

# Evals to run (start small; add more later)
TASKS=("gsm8k")

# Where results go
RESULTS_DIR="${RESULTS_DIR:-$HOME/sweep-results}"
mkdir -p "$RESULTS_DIR"

# --- Preflight: is the server actually up? ------------------------------------
# NOTE: this checks only that *a* server is up and what name it reports. It
# CANNOT verify the weights behind that name. The real discipline is to restart
# the server for each model before sweeping it; the served-name check and the
# audit file written below just make a mismatch catchable after the fact.
MODELS_JSON=$(curl -sf "${BASE_URL}/v1/models")
if [ -z "$MODELS_JSON" ]; then
  echo "ERROR: no vLLM server responding at ${BASE_URL}/v1/models"
  echo "Start the server for '$MODEL' first, then re-run."
  exit 1
fi

# Weak guard: substring match can false-warn on quantized paths / escaping
# quirks, so treat WARN as a heads-up, not an error.
if ! printf '%s' "$MODELS_JSON" | grep -q "$MODEL"; then
  echo "WARN: '$MODEL' not found in the served model list (may be a false alarm)."
  echo "      Check the served name matches (vLLM --served-model-name)."
fi

# --- Assemble optional flags --------------------------------------------------
EXTRA_ARGS=()
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi
if [ "$CHAT_TEMPLATE" = "1" ]; then
  EXTRA_ARGS+=(--apply_chat_template --fewshot_as_multiturn)
fi

# --- Sweep --------------------------------------------------------------------
for TASK in "${TASKS[@]}"; do
  # Build a safe filename from model + task
  SAFE=$(echo "${MODEL}_${TASK}" | tr '/' '_')
  OUT="$RESULTS_DIR/${SAFE}"

  if [ -f "$OUT/.done" ]; then
    echo "SKIP  $MODEL x $TASK  (already done)"
    continue
  fi

  echo "RUN   $MODEL x $TASK"
  if lm_eval --model local-completions \
       --model_args "model=${MODEL},base_url=${BASE_URL}/v1/completions,num_concurrent=${NUM_CONCURRENT}" \
       --tasks "${TASK}" \
       --num_fewshot "${NUM_FEWSHOT}" \
       --output_path "$OUT" \
       "${EXTRA_ARGS[@]}"; then
    # Record what the server claimed to be serving, for after-the-fact auditing.
    printf '%s' "$MODELS_JSON" > "$OUT/served_models.json"
    touch "$OUT/.done"
    echo "OK    $MODEL x $TASK"
  else
    rc=$?   # capture immediately: any command before this would clobber $?
    echo "FAIL  $MODEL x $TASK  (exit $rc) — not marking done, will retry next run"
  fi
done

echo "Sweep complete for $MODEL"
