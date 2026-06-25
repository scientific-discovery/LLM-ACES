#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$SCRIPT_DIR/data/ode"
LOG_ROOT="$SCRIPT_DIR/logs/odebench"
OUTPUT_DIR="$SCRIPT_DIR/outputs/odebench"
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

PROBLEMS_1D=(
  "autocatalysis"
  "autocatalytic-gene-switching"
  "budworm-outbreak-model"
  "budworm-outbreak-predation"
  "dimensionally-reduced-sir"
  "gompertz-law-tumor-growth"
  "improved-logistic-equation-harvesting"
  "improved-logistic-equation-harvesting-dimensionless"
  "landau-equation"
  "language-death-model"
  "logistic-equation-allee-effect"
  "logistic-equation-harvesting"
  "naive-critical-slowing-down"
  "overdamped-bead"
  "overdamped-pendulum"
  "photons-in-a-laser"
  "population-growth-carrying-capacity"
  "population-growth-naive"
  "protein-expression"
  "rc-circuit"
  "rc-circuit-non-linear-resistor"
  "refined-language-death-model"
  "velocity-falling-object"
)

PROBLEMS_2D=(
  "bacterial-respiration-model"
  "binocular-rivalry-model"
  "brusselator"
  "catalyzing-rna-molecules"
  "cdima-reaction"
  "cell-cycle-model"
  "damped-double-well-oscillator"
  "dipole-fixed-point"
  "driven-pendulum-linear-damping"
  "driven-pendulum-quadratic-damping"
  "duffing-equation"
  "frictionless-bead"
  "glider"
  "glycolytic-oscillator"
  "gray-scott-model"
  "harmonic-oscillator"
  "harmonic-oscillator-damping"
  "interacting-bar-magnets"
  "lotka-volterra-competition"
  "lotka-volterra-simple"
  "oscillator-death-model"
  "pendulum-non-linear-damping"
  "pendulum-without-friction"
  "rotational-dynamics"
  "schnackenberg-model"
  "sir-infection"
  "van-der-pol-oscillator"
  "van-der-pol-oscillator-simplified"
)

run_problem() {
  local problem="$1"
  local data_path="$DATA_ROOT/$problem/$problem.npz"
  local log_path="$LOG_ROOT/$problem"
  local out_path="$OUTPUT_DIR/${problem}.json"

  if [[ ! -f "$data_path" ]]; then
    echo "[SKIP] $problem — no data at $data_path"
    return
  fi

  if [[ -f "$out_path" ]]; then
    echo "[SKIP] $problem — already done ($out_path)"
    return
  fi

  echo ""
  echo "══════════════════════════════════════════════"
  echo "  Problem : $problem"
  echo "══════════════════════════════════════════════"
  mkdir -p "$log_path"

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

  echo "[DONE] $problem"
}

mkdir -p "$LOG_ROOT" "$OUTPUT_DIR"

echo "ODEBench run — data root: $DATA_ROOT"
echo ""
echo "── 1-D problems ──"
for p in "${PROBLEMS_1D[@]}"; do run_problem "$p"; done

echo ""
echo "── 2-D problems ──"
for p in "${PROBLEMS_2D[@]}"; do run_problem "$p"; done

echo ""
echo "All done. Outputs: $OUTPUT_DIR"
