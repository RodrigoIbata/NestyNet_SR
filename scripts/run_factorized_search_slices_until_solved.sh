#!/bin/bash
# Run one AI Feynman closure-benchmark problem across disjoint CSV data slices
# until the benchmark JSON reports a solved result.
#
# Intended for SRBench workspaces, e.g. from ../SRBench_0.000_basinSR:
#   IDS=037 ../NestyNet_SR/scripts/run_factorized_search_slices_until_solved.sh
#
# Key env vars:
#   IDS=037                         Problem id; accepts 37, 037, pb037, feynman_037
#   TOTAL_POINTS=100000             Total available CSV rows (default: 100000)
#   N_POINTS=2000                   Fit points and probe points per slice
#   OUTPUT_DIR=results/slice_runs   Output JSON directory
#   LOG_DIR="$OUTPUT_DIR/logs"      Log directory
#   SLICE_START=0                   First DATA_SLICE to try
#   SLICE_COUNT=...                 Optional override; default TOTAL_POINTS/(2*N_POINTS)

set -euo pipefail

CALLER_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_ROOT"

normalize_id() {
  local raw="$1"
  raw="${raw#feynman_}"
  raw="${raw#pb}"
  if [[ -z "$raw" || "$raw" == *[!0-9]* ]]; then
    echo "Invalid problem id: $1" >&2
    return 2
  fi
  printf "%03d" "$((10#$raw))"
}

raw_id="${ID:-${IDS:-${ONLY_IDS:-037}}}"
if [[ "$raw_id" == *","* || "$raw_id" == *" "* ]]; then
  echo "This until-solved runner expects one problem id; got: $raw_id" >&2
  exit 2
fi
id="$(normalize_id "$raw_id")"
eq_id="feynman_${id}"

if [[ -z "${EQUATIONS:-}" ]]; then
  if [[ -f "$CALLER_CWD/data/equations.txt" ]]; then
    EQUATIONS="$CALLER_CWD/data/equations.txt"
  else
    EQUATIONS="data/equations.txt"
  fi
fi
if [[ ! -f "$EQUATIONS" ]]; then
  echo "Equations file not found: $EQUATIONS" >&2
  exit 1
fi

ORACLE_DATA_SOURCE="${ORACLE_DATA_SOURCE:-csv}"
case "$ORACLE_DATA_SOURCE" in
  csv|CSV) ORACLE_DATA_SOURCE="csv" ;;
  *) echo "This slice runner is intended for ORACLE_DATA_SOURCE=csv" >&2; exit 2 ;;
esac

if [[ -z "${DATA_DIR:-}" ]]; then
  if [[ -d "$CALLER_CWD/data" ]]; then
    DATA_DIR="$CALLER_CWD/data"
  else
    DATA_DIR="data"
  fi
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "DATA_DIR not found: $DATA_DIR" >&2
  exit 1
fi

PYTHON="${PYTHON:-python}"
if ! PYTHON="$(command -v "$PYTHON")"; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

TOTAL_POINTS="${TOTAL_POINTS:-100000}"
N_POINTS="${N_POINTS:-${N_FIT:-${N_PROBE:-2000}}}"
SLICE_START="${SLICE_START:-0}"

for name in TOTAL_POINTS N_POINTS SLICE_START; do
  value="${!name}"
  if [[ -z "$value" || "$value" == *[!0-9]* ]]; then
    echo "$name must be a nonnegative integer" >&2
    exit 2
  fi
done
if [[ "$N_POINTS" -le 0 ]]; then
  echo "N_POINTS must be positive" >&2
  exit 2
fi

if [[ -z "${SLICE_COUNT:-}" ]]; then
  SLICE_COUNT="$(( TOTAL_POINTS / (2 * N_POINTS) ))"
fi
if [[ -z "$SLICE_COUNT" || "$SLICE_COUNT" == *[!0-9]* || "$SLICE_COUNT" -le 0 ]]; then
  echo "SLICE_COUNT must be positive; got '$SLICE_COUNT'" >&2
  exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/factorized_search_aif_slices}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

N_ITER="${N_ITER:-1400}"
N_SEEDS="${N_SEEDS:-1}"
MAX_PROPOSALS="${MAX_PROPOSALS:-48}"
ANCHORS="${ANCHORS:-8}"
PREVIEW_TOPK="${PREVIEW_TOPK:-12}"
EXACT_TOPK="${EXACT_TOPK:-8}"
BEAM_WIDTH="${BEAM_WIDTH:-6}"
SEED_EXACT_TOPK="${SEED_EXACT_TOPK:-6}"
SEED_BEAM_WIDTH="${SEED_BEAM_WIDTH:-4}"
SEED_SCAFFOLD_RESERVE="${SEED_SCAFFOLD_RESERVE:-8}"
PAIR_RESCUE_TOPK="${PAIR_RESCUE_TOPK:-4}"
PAIR_RESCUE_MAX_PAIRS="${PAIR_RESCUE_MAX_PAIRS:-6}"
PAIR_NORMAL_ENABLE="${PAIR_NORMAL_ENABLE:-0}"
PAIR_NORMAL_TOPK="${PAIR_NORMAL_TOPK:-3}"
PAIR_NORMAL_MAX_PAIRS="${PAIR_NORMAL_MAX_PAIRS:-1}"
CLOSURE_DEBUG_TOPK="${CLOSURE_DEBUG_TOPK:-0}"
MAX_NVARS="${MAX_NVARS:-6}"
SEED="${SEED:-42}"
SUCCESS_MSE="${SUCCESS_MSE:-1e-8}"
Y_CHECK="${Y_CHECK:-1}"
Y_CHECK_ABS_TOL="${Y_CHECK_ABS_TOL:-1e-8}"
Y_CHECK_REL_TOL="${Y_CHECK_REL_TOL:-1e-8}"
BENCHMARK_MODULE="${BENCHMARK_MODULE:-nestynet_sr.sr_search.factorized_search.aif_closure_benchmark}"

echo "Running $eq_id over $SLICE_COUNT slice(s), starting at DATA_SLICE=$SLICE_START"
echo "Per slice: n_fit=$N_POINTS n_probe=$N_POINTS; TOTAL_POINTS=$TOTAL_POINTS"
echo "Outputs: $OUTPUT_DIR/${eq_id}_<slice>.json"
echo "Logs:    $LOG_DIR/${eq_id}_<slice>.log"

for ((offset = 0; offset < SLICE_COUNT; offset++)); do
  s="$((SLICE_START + offset))"
  output_path="${OUTPUT_DIR}/${eq_id}_${s}.json"
  log_path="${LOG_DIR}/${eq_id}_${s}.log"

  cmd=(
    "$PYTHON" -u -m "$BENCHMARK_MODULE"
    --equations "$EQUATIONS"
    --only "$id"
    --max_nvars "$MAX_NVARS"
    --n_iter "$N_ITER"
    --n_seeds "$N_SEEDS"
    --max_proposals "$MAX_PROPOSALS"
    --anchors "$ANCHORS"
    --preview_topk "$PREVIEW_TOPK"
    --exact_topk "$EXACT_TOPK"
    --beam_width "$BEAM_WIDTH"
    --seed_exact_topk "$SEED_EXACT_TOPK"
    --seed_beam_width "$SEED_BEAM_WIDTH"
    --seed_scaffold_reserve "$SEED_SCAFFOLD_RESERVE"
    --pair_normal_topk "$PAIR_NORMAL_TOPK"
    --pair_normal_max_pairs "$PAIR_NORMAL_MAX_PAIRS"
    --pair_rescue_topk "$PAIR_RESCUE_TOPK"
    --pair_rescue_max_pairs "$PAIR_RESCUE_MAX_PAIRS"
    --closure_debug_topk "$CLOSURE_DEBUG_TOPK"
    --seed "$SEED"
    --success_mse "$SUCCESS_MSE"
    --data_dir "$DATA_DIR"
    --n_fit "$N_POINTS"
    --n_probe "$N_POINTS"
    --data_slice "$s"
    --y_check_abs_tol "$Y_CHECK_ABS_TOL"
    --y_check_rel_tol "$Y_CHECK_REL_TOL"
    --output "$output_path"
    "$@"
  )
  if [[ "$PAIR_NORMAL_ENABLE" == "1" ]]; then
    cmd+=(--pair_normal_enable)
  fi
  case "$Y_CHECK" in
    0|false|False|off|OFF|no|NO) cmd+=(--no_y_check) ;;
    1|true|True|on|ON|yes|YES) ;;
    *) echo "Y_CHECK must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
  esac

  printf "[START] %s DATA_SLICE=%s\n" "$eq_id" "$s"
  if "${cmd[@]}" >"$log_path" 2>&1; then
    parse_out="$("$PYTHON" - "$output_path" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path) as handle:
    payload = json.load(handle)
rows = payload.get("results") or []
row = rows[0] if rows else {}
status = str(row.get("status", "missing"))
try:
    mse = float(row.get("mse", math.inf))
except Exception:
    mse = math.inf
expr = str(row.get("expr", "?")).replace("\t", " ")[:120]
solved = status == "solved"
print(f"{1 if solved else 0}\t{status}\t{mse:.16g}\t{expr}")
PY
)"
    solved="${parse_out%%	*}"
    rest="${parse_out#*	}"
    status="${rest%%	*}"
    rest="${rest#*	}"
    mse="${rest%%	*}"
    expr="${rest#*	}"
    printf "[DONE ] %s DATA_SLICE=%s status=%s mse=%s expr=%s\n" "$eq_id" "$s" "$status" "$mse" "$expr"
    if [[ "$solved" == "1" ]]; then
      printf "[SOLVED] %s DATA_SLICE=%s -> %s\n" "$eq_id" "$s" "$output_path"
      exit 0
    fi
  else
    status=$?
    printf "[FAIL ] %s DATA_SLICE=%s -> %s\n" "$eq_id" "$s" "$log_path" >&2
    exit "$status"
  fi
done

printf "[UNSOLVED] %s after %s slice(s). Last slice: %s\n" "$eq_id" "$SLICE_COUNT" "$((SLICE_START + SLICE_COUNT - 1))"
exit 1
