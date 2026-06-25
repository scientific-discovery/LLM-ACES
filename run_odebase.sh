#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$SCRIPT_DIR/data/odebase"
LOG_ROOT="$SCRIPT_DIR/logs/odebase"
OUTPUT_DIR="$SCRIPT_DIR/outputs/odebase"
PYTHON_SCRIPT="$SCRIPT_DIR/llm-aces/active_llm_aces.py"

USE_API="true"
API_MODEL="gpt-4o-mini"
API_PROVIDER="openai"
N_ITERATIONS=10
MAX_CONCEPTS=3
N_VIRTUAL=10
BO_INIT_POINTS=3
PYSR_NITER=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --use_api)       USE_API="$2";       shift 2 ;;
    --api_model)     API_MODEL="$2";     shift 2 ;;
    --api_provider)  API_PROVIDER="$2";  shift 2 ;;
    --n_iterations)  N_ITERATIONS="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$USE_API" == "true" ]]; then
  OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY:-}}"
  : "${OPENAI_API_KEY:?Set OPENAI_API_KEY (or API_KEY) before running with --use_api true}"
fi

run_dataset() {
  local data_path="$1"
  local system_name
  system_name="$(basename "$(dirname "$data_path")")"
  local log_path="$LOG_ROOT/$system_name"
  local out_path="$OUTPUT_DIR/${system_name}.json"

  if [[ -f "$out_path" ]]; then
    echo "[SKIP] $system_name — already done"
    return
  fi

  echo ""
  echo "══════════════════════════════════════════════"
  echo "  System : $system_name"
  echo "══════════════════════════════════════════════"
  mkdir -p "$log_path"

  set +e
  python "$PYTHON_SCRIPT" \
    --data_path              "$data_path" \
    --log_path               "$log_path" \
    --output_dir             "$OUTPUT_DIR" \
    --n_iterations           "$N_ITERATIONS" \
    --max_concepts_per_round "$MAX_CONCEPTS" \
    --n_virtual              "$N_VIRTUAL" \
    --bo_init_points         "$BO_INIT_POINTS" \
    --use_api                "$USE_API" \
    --api_model              "$API_MODEL" \
    --api_provider           "$API_PROVIDER" \
    --pysr_niterations       "$PYSR_NITER" \
    2>&1 | tee "$log_path/stdout.log"
  local status=$?
  set -e

  if (( status == 0 )); then
    echo "[DONE] $system_name"
  else
    echo "[FAILED] $system_name (exit $status) — continuing"
  fi
}

mkdir -p "$LOG_ROOT" "$OUTPUT_DIR"

# Discover NPZ files (skip noisy _snr_ variants)
mapfile -t VARS2 < <(find "$DATA_ROOT" -name "odebase_vars2_prog*.npz" | grep -v snr | sort 2>/dev/null)
mapfile -t VARS3 < <(find "$DATA_ROOT" -name "odebase_vars3_prog*.npz" | grep -v snr | sort 2>/dev/null)

if [[ ${#VARS2[@]} -eq 0 && ${#VARS3[@]} -eq 0 ]]; then
  echo "No ODEBase NPZ files found under $DATA_ROOT"
  echo "Place files matching odebase_vars{2,3}_prog*.npz there and re-run."
  exit 1
fi

echo "ODEBase run — data root: $DATA_ROOT"
echo "  vars2 datasets : ${#VARS2[@]}"
echo "  vars3 datasets : ${#VARS3[@]}"

echo ""
echo "── 2-D systems (vars2) ──"
for f in "${VARS2[@]}"; do run_dataset "$f"; done

echo ""
echo "── 3-D systems (vars3) ──"
for f in "${VARS3[@]}"; do run_dataset "$f"; done

echo ""
echo "All done. Outputs: $OUTPUT_DIR"
