#!/bin/bash
# Run factorized symbolic search AI Feynman benchmark problems in parallel.
#
# Optional env vars:
#   JOBS=8                         Number of parallel workers (default: 8)
#   EQUATIONS=data/equations.txt   Source equations file (default: data/equations.txt)
#   START_ID=0                     First equation ID to include (default: all)
#   END_ID=119                     Last equation ID to include (default: all)
#   ONLY_IDS="018 090"             Explicit include list; accepts 18,018,feynman_018
#   SKIP_IDS="004 018"             Exclude list; accepts 4,004,feynman_004
#   OUTPUT_DIR=results/factorized_search_aif Output directory for per-problem JSON
#   LOG_DIR="$OUTPUT_DIR/logs"     Log directory for per-problem stdout/stderr
#   PYTHON=python                  Python executable (default: python)
#   N_ITER=1400                    Default --n_iter
#   N_SEEDS=1                      Default --n_seeds
#   MAX_PROPOSALS=48               Default --max_proposals
#   ANCHORS=8                      Default --anchors
#   PREVIEW_TOPK=12                Default --preview_topk
#   EXACT_TOPK=8                   Default --exact_topk
#   BEAM_WIDTH=6                   Default --beam_width
#   SEED_EXACT_TOPK=6              Default --seed_exact_topk
#   SEED_BEAM_WIDTH=4              Default --seed_beam_width
#   SEED_SCAFFOLD_RESERVE=8        Default --seed_scaffold_reserve
#   PAIR_RESCUE_TOPK=4             Default --pair_rescue_topk
#   PAIR_RESCUE_MAX_PAIRS=6        Default --pair_rescue_max_pairs
#   CLOSURE_DEBUG_TOPK=0           Default --closure_debug_topk
#   EMERGENT_AUX_ATOMS=1           Add --emergent-aux-atoms / --no-emergent-aux-atoms when set
#   EMERGENT_BASIS=1               Legacy row-promotion ablation flag
#   MAX_NVARS=6                    Default --max_nvars
#   ORACLE_DATA_SOURCE=synthetic    Data source: synthetic or csv (default: synthetic)
#   DATA_DIR=data                   CSV directory when ORACLE_DATA_SOURCE=csv
#   N_FIT=...                       Fit points for oracle data; defaults to NDATA_TRAIN or 2000 in csv mode
#   N_PROBE=...                     Probe/validation points; defaults to NDATA_VAL or 2000 in csv mode
#   SEARCH_N_FIT=...                Fit rows used during search; full N_FIT is kept for optional validation
#   SEARCH_N_PROBE=...              Probe rows used during search; full N_PROBE is kept for optional validation
#   FINAL_VALIDATE_FULL=1           Re-score returned candidates on the full N_FIT/N_PROBE split
#   FINAL_VALIDATE_RERANK=1         Let full validation rerank candidates after search (default: 0)
#   NESTY_ATOMIZED_LINEAR_SPAN_USE_OBS_POOL=1
#                                   Allow atomized spans to use observed-but-not-retained atoms
#   NESTY_ATOMIZED_LINEAR_SPAN_SAME_ROUND=1
#                                   Allow same-round atomized spans after atom harvesting
#   NESTY_ATOMIZED_LINEAR_SPAN_EXACT_QUOTA=2
#                                   Max atomized rows exact-scored per round (default: min(2, exact_topk-1))
#   NESTY_ATOM_POLICY_USE_OBS_POOL=1
#                                   Allow atom policy steering from observed-but-not-retained atoms
#   DATA_SLICE=0                    Disjoint external data block index
#   Y_CHECK=1                       Check CSV y values against oracle target expression
#   Y_CHECK_ABS_TOL=1e-8            Absolute tolerance for y check
#   Y_CHECK_REL_TOL=1e-8            Relative tolerance for y check
#   SEED=42                        Default --seed
#   SUCCESS_MSE=1e-8               Default --success_mse
#   REFINE_SKELETON=1              Add --refine-skeleton / --no-refine-skeleton when set
#   REFINE_PROFILE=rare_slate       Optional continuous-refinement profile
#   REFINE_MODE=slate              Optional refinement placement mode
#   DRY_RUN=0                      Print commands without executing them
#
# Extra CLI args are passed through to:
#   python -m nestynet_sr.sr_search.factorized_search.aif_closure_benchmark
#
# Reserved flags:
#   --equations / --only / --output
# Use EQUATIONS=... and the env vars above instead.

set -euo pipefail

# Go to project root (parent of scripts/)
cd "$(dirname "$0")/.."

JOBS="${JOBS:-8}"
EQUATIONS="${EQUATIONS:-data/equations.txt}"
START_ID="${START_ID:-}"
END_ID="${END_ID:-}"
ONLY_IDS="${ONLY_IDS:-}"
SKIP_IDS="${SKIP_IDS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/factorized_search_aif}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
PYTHON="${PYTHON:-python}"
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
EMERGENT_AUX_ATOMS="${EMERGENT_AUX_ATOMS:-}"
EMERGENT_BASIS="${EMERGENT_BASIS:-}"
MAX_NVARS="${MAX_NVARS:-6}"
ORACLE_DATA_SOURCE="${ORACLE_DATA_SOURCE:-synthetic}"
DATA_DIR="${DATA_DIR:-data}"
N_FIT="${N_FIT:-${NDATA_TRAIN:-}}"
N_PROBE="${N_PROBE:-${NDATA_VAL:-}}"
SEARCH_N_FIT="${SEARCH_N_FIT:-}"
SEARCH_N_PROBE="${SEARCH_N_PROBE:-}"
FINAL_VALIDATE_FULL="${FINAL_VALIDATE_FULL:-0}"
FINAL_VALIDATE_RERANK="${FINAL_VALIDATE_RERANK:-0}"
DATA_SLICE="${DATA_SLICE:-0}"
Y_CHECK="${Y_CHECK:-1}"
Y_CHECK_ABS_TOL="${Y_CHECK_ABS_TOL:-1e-8}"
Y_CHECK_REL_TOL="${Y_CHECK_REL_TOL:-1e-8}"
SEED="${SEED:-42}"
SUCCESS_MSE="${SUCCESS_MSE:-1e-8}"
REFINE_SKELETON="${REFINE_SKELETON:-}"
REFINE_PROFILE="${REFINE_PROFILE:-}"
REFINE_MODE="${REFINE_MODE:-}"
REFINE_OPTIMIZER="${REFINE_OPTIMIZER:-}"
REFINE_LBFGS_STEPS="${REFINE_LBFGS_STEPS:-}"
REFINE_FIT_SUBSET="${REFINE_FIT_SUBSET:-}"
REFINE_NUM_RESTARTS="${REFINE_NUM_RESTARTS:-}"
REFINE_MAX_VARIANTS="${REFINE_MAX_VARIANTS:-}"
REFINE_MAX_PARAMS="${REFINE_MAX_PARAMS:-}"
REFINE_MAX_TRIALS="${REFINE_MAX_TRIALS:-}"
REFINE_GATE_BEST_FACTOR="${REFINE_GATE_BEST_FACTOR:-}"
REFINE_LINEAR_COMBO="${REFINE_LINEAR_COMBO:-}"
DRY_RUN="${DRY_RUN:-0}"
BENCHMARK_MODULE="${BENCHMARK_MODULE:-nestynet_sr.sr_search.factorized_search.aif_closure_benchmark}"

if [ ! -f "$EQUATIONS" ]; then
  echo "Equations file not found: $EQUATIONS" >&2
  exit 1
fi

if ! PYTHON="$(command -v "$PYTHON")"; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

case "$ORACLE_DATA_SOURCE" in
  synthetic|SYNTHETIC) ORACLE_DATA_SOURCE="synthetic" ;;
  csv|CSV) ORACLE_DATA_SOURCE="csv" ;;
  *) echo "ORACLE_DATA_SOURCE must be synthetic or csv" >&2; exit 2 ;;
esac

require_positive_int() {
  name="$1"
  value="$2"
  if [ -z "$value" ]; then
    return 0
  fi
  case "$value" in
    *[!0-9]*)
      echo "$name must be a positive integer when set" >&2
      exit 2
      ;;
  esac
  if [ "$value" -le 0 ]; then
    echo "$name must be a positive integer when set" >&2
    exit 2
  fi
}

require_nonnegative_int() {
  name="$1"
  value="$2"
  case "$value" in
    ''|*[!0-9]*)
      echo "$name must be a nonnegative integer" >&2
      exit 2
      ;;
  esac
}

if [ "$ORACLE_DATA_SOURCE" = "csv" ]; then
  if [ ! -d "$DATA_DIR" ]; then
    echo "DATA_DIR not found for ORACLE_DATA_SOURCE=csv: $DATA_DIR" >&2
    exit 1
  fi
  N_FIT="${N_FIT:-2000}"
  N_PROBE="${N_PROBE:-2000}"
fi
require_positive_int N_FIT "$N_FIT"
require_positive_int N_PROBE "$N_PROBE"
require_positive_int SEARCH_N_FIT "$SEARCH_N_FIT"
require_positive_int SEARCH_N_PROBE "$SEARCH_N_PROBE"
require_nonnegative_int DATA_SLICE "$DATA_SLICE"
if [ -n "$SEARCH_N_FIT" ] && [ -n "$N_FIT" ] && [ "$SEARCH_N_FIT" -gt "$N_FIT" ]; then
  echo "SEARCH_N_FIT cannot exceed N_FIT" >&2
  exit 2
fi
if [ -n "$SEARCH_N_PROBE" ] && [ -n "$N_PROBE" ] && [ "$SEARCH_N_PROBE" -gt "$N_PROBE" ]; then
  echo "SEARCH_N_PROBE cannot exceed N_PROBE" >&2
  exit 2
fi
case "$Y_CHECK" in
  0|false|False|off|OFF|no|NO) Y_CHECK_BOOL=0 ;;
  1|true|True|on|ON|yes|YES) Y_CHECK_BOOL=1 ;;
  *) echo "Y_CHECK must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
case "$FINAL_VALIDATE_FULL" in
  0|false|False|off|OFF|no|NO) FINAL_VALIDATE_FULL_BOOL=0 ;;
  1|true|True|on|ON|yes|YES) FINAL_VALIDATE_FULL_BOOL=1 ;;
  *) echo "FINAL_VALIDATE_FULL must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
case "$FINAL_VALIDATE_RERANK" in
  0|false|False|off|OFF|no|NO) FINAL_VALIDATE_RERANK_BOOL=0 ;;
  1|true|True|on|ON|yes|YES) FINAL_VALIDATE_RERANK_BOOL=1 ;;
  *) echo "FINAL_VALIDATE_RERANK must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac

for arg in "$@"; do
  case "$arg" in
    --equations|--equations=*|--only|--only=*|--output|--output=*|--data_csv|--data_csv=*|--data_dir|--data_dir=*|--n_fit|--n_fit=*|--n_probe|--n_probe=*|--search-n-fit|--search-n-fit=*|--search_n_fit|--search_n_fit=*|--search-n-probe|--search-n-probe=*|--search_n_probe|--search_n_probe=*|--final-validate-full|--no-final-validate-full|--final_validate_full|--no_final_validate_full|--final-validate-rerank|--no-final-validate-rerank|--final_validate_rerank|--no_final_validate_rerank|--data_slice|--data_slice=*)
      echo "Reserved benchmark flag '$arg' is managed by scripts/run_factorized_search_all.sh." >&2
      echo "Use EQUATIONS=... and the script env vars instead." >&2
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

id_file="$(mktemp "${TMPDIR:-/tmp}/factorized_search_ids.XXXXXX")"
trap 'rm -f "$id_file"' EXIT

awk \
  -v start_id="$START_ID" \
  -v end_id="$END_ID" \
  -v only_ids="$ONLY_IDS" \
  -v skip_ids="$SKIP_IDS" \
  '
function normalize(tok, out) {
  out = tok
  sub(/^feynman_/, "", out)
  if (out == "") return ""
  if (out ~ /^[0-9]+$/) return sprintf("%03d", out + 0)
  return ""
}
BEGIN {
  if (start_id == "") {
    start_num = -1
  } else {
    start_norm = normalize(start_id)
    if (start_norm == "") {
      print "Invalid START_ID: " start_id > "/dev/stderr"
      exit 2
    }
    start_num = start_norm + 0
  }

  if (end_id == "") {
    end_num = 1000000
  } else {
    end_norm = normalize(end_id)
    if (end_norm == "") {
      print "Invalid END_ID: " end_id > "/dev/stderr"
      exit 2
    }
    end_num = end_norm + 0
  }

  n_only = split(only_ids, raw_only, /[ ,]+/)
  for (i = 1; i <= n_only; i++) {
    if (raw_only[i] == "") continue
    only_norm = normalize(raw_only[i])
    if (only_norm == "") {
      print "Invalid ONLY_IDS entry: " raw_only[i] > "/dev/stderr"
      exit 2
    }
    only[only_norm] = 1
    only_count++
  }

  n_skip = split(skip_ids, raw_skip, /[ ,]+/)
  for (i = 1; i <= n_skip; i++) {
    if (raw_skip[i] == "") continue
    skip_norm = normalize(raw_skip[i])
    if (skip_norm == "") {
      print "Invalid SKIP_IDS entry: " raw_skip[i] > "/dev/stderr"
      exit 2
    }
    skip[skip_norm] = 1
  }
}
NF && $1 !~ /^#/ {
  id = normalize($1)
  if (id == "") next
  id_num = id + 0
  if (id_num < start_num || id_num > end_num) next
  if (only_count > 0 && !(id in only)) next
  if (id in skip) next
  print id
}
' "$EQUATIONS" > "$id_file"

if [ ! -s "$id_file" ]; then
  echo "No AI Feynman equations matched the current filters." >&2
  exit 0
fi

selected_count="$(wc -l < "$id_file" | tr -d '[:space:]')"
echo "Launching $selected_count AI Feynman problems with JOBS=$JOBS"
echo "Equations: $EQUATIONS"
echo "Data:      $ORACLE_DATA_SOURCE"
if [ "$ORACLE_DATA_SOURCE" = "csv" ]; then
  echo "Data dir:  $DATA_DIR"
  echo "Split:     n_fit=$N_FIT n_probe=$N_PROBE data_slice=$DATA_SLICE"
fi
if [ -n "$SEARCH_N_FIT" ] || [ -n "$SEARCH_N_PROBE" ]; then
  echo "Search:    n_fit=${SEARCH_N_FIT:-$N_FIT} n_probe=${SEARCH_N_PROBE:-$N_PROBE}"
fi
if [ "$FINAL_VALIDATE_FULL_BOOL" = "1" ]; then
  if [ "$FINAL_VALIDATE_RERANK_BOOL" = "1" ]; then
    echo "Validate:  full split after search (rerank enabled)"
  else
    echo "Validate:  full split after search (audit only)"
  fi
fi
case "$EMERGENT_AUX_ATOMS" in
  1|true|True|on|ON|yes|YES) echo "Aux atoms: emergent SeedBlocks enabled" ;;
esac
atomized_span_notes=()
case "${NESTY_ATOMIZED_LINEAR_SPAN_USE_OBS_POOL:-}" in
  1|true|True|on|ON|yes|YES) atomized_span_notes+=("observed-pool") ;;
esac
case "${NESTY_ATOMIZED_LINEAR_SPAN_SAME_ROUND:-}" in
  1|true|True|on|ON|yes|YES) atomized_span_notes+=("same-round") ;;
esac
if [ -n "${NESTY_ATOMIZED_LINEAR_SPAN_BUDGET:-}" ]; then
  atomized_span_notes+=("budget=${NESTY_ATOMIZED_LINEAR_SPAN_BUDGET}")
fi
if [ "${#atomized_span_notes[@]}" -gt 0 ]; then
  echo "Atomized spans: ${atomized_span_notes[*]}"
fi
case "$EMERGENT_BASIS" in
  1|true|True|on|ON|yes|YES) echo "Emergent basis rows: legacy row promotion enabled" ;;
esac
echo "Outputs:   $OUTPUT_DIR"
echo "Logs:      $LOG_DIR"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1, commands will be printed but not executed."
  XARGS_JOBS=1
else
  XARGS_JOBS="$JOBS"
fi

export OUTPUT_DIR
export LOG_DIR
export EQUATIONS
export PYTHON
export N_ITER
export N_SEEDS
export MAX_PROPOSALS
export ANCHORS
export PREVIEW_TOPK
export EXACT_TOPK
export BEAM_WIDTH
export SEED_EXACT_TOPK
export SEED_BEAM_WIDTH
export SEED_SCAFFOLD_RESERVE
export PAIR_NORMAL_ENABLE
export PAIR_NORMAL_TOPK
export PAIR_NORMAL_MAX_PAIRS
export PAIR_RESCUE_TOPK
export PAIR_RESCUE_MAX_PAIRS
export CLOSURE_DEBUG_TOPK
export EMERGENT_AUX_ATOMS
export EMERGENT_BASIS
export MAX_NVARS
export ORACLE_DATA_SOURCE
export DATA_DIR
export N_FIT
export N_PROBE
export SEARCH_N_FIT
export SEARCH_N_PROBE
export FINAL_VALIDATE_FULL_BOOL
export FINAL_VALIDATE_RERANK_BOOL
export DATA_SLICE
export Y_CHECK_BOOL
export Y_CHECK_ABS_TOL
export Y_CHECK_REL_TOL
export SEED
export SUCCESS_MSE
export REFINE_SKELETON
export REFINE_PROFILE
export REFINE_MODE
export REFINE_OPTIMIZER
export REFINE_LBFGS_STEPS
export REFINE_FIT_SUBSET
export REFINE_NUM_RESTARTS
export REFINE_MAX_VARIANTS
export REFINE_MAX_PARAMS
export REFINE_MAX_TRIALS
export REFINE_GATE_BEST_FACTOR
export REFINE_LINEAR_COMBO
export DRY_RUN
export BENCHMARK_MODULE

xargs -P "$XARGS_JOBS" -I {} bash -c '
set -euo pipefail

id="$1"
shift

eq_id="feynman_${id}"
output_path="${OUTPUT_DIR}/${eq_id}.json"
log_path="${LOG_DIR}/${eq_id}.log"

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
  --output "$output_path"
  "$@"
)

case "$EMERGENT_BASIS" in
  "") ;;
  0|false|False|off|OFF|no|NO) cmd+=(--no-emergent-basis) ;;
  1|true|True|on|ON|yes|YES) cmd+=(--emergent-basis) ;;
  *) echo "EMERGENT_BASIS must be empty or 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac

case "$EMERGENT_AUX_ATOMS" in
  "") ;;
  0|false|False|off|OFF|no|NO) cmd+=(--no-emergent-aux-atoms) ;;
  1|true|True|on|ON|yes|YES) cmd+=(--emergent-aux-atoms) ;;
  *) echo "EMERGENT_AUX_ATOMS must be empty or 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac

case "$REFINE_SKELETON" in
  "") ;;
  0|false|False|off|OFF|no|NO) cmd+=(--no-refine-skeleton) ;;
  1|true|True|on|ON|yes|YES) cmd+=(--refine-skeleton) ;;
  *) echo "REFINE_SKELETON must be empty or 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
if [ -n "$REFINE_PROFILE" ]; then
  cmd+=(--refine-profile "$REFINE_PROFILE")
fi
if [ -n "$REFINE_MODE" ]; then
  cmd+=(--refine-mode "$REFINE_MODE")
fi
if [ -n "$REFINE_OPTIMIZER" ]; then
  cmd+=(--refine-optimizer "$REFINE_OPTIMIZER")
fi
if [ -n "$REFINE_LBFGS_STEPS" ]; then
  cmd+=(--refine-lbfgs-steps "$REFINE_LBFGS_STEPS")
fi
if [ -n "$REFINE_FIT_SUBSET" ]; then
  cmd+=(--refine-fit-subset "$REFINE_FIT_SUBSET")
fi
if [ -n "$REFINE_NUM_RESTARTS" ]; then
  cmd+=(--refine-num-restarts "$REFINE_NUM_RESTARTS")
fi
if [ -n "$REFINE_MAX_VARIANTS" ]; then
  cmd+=(--refine-max-variants "$REFINE_MAX_VARIANTS")
fi
if [ -n "$REFINE_MAX_PARAMS" ]; then
  cmd+=(--refine-max-params "$REFINE_MAX_PARAMS")
fi
if [ -n "$REFINE_MAX_TRIALS" ]; then
  cmd+=(--refine-max-trials "$REFINE_MAX_TRIALS")
fi
if [ -n "$REFINE_GATE_BEST_FACTOR" ]; then
  cmd+=(--refine-gate-best-factor "$REFINE_GATE_BEST_FACTOR")
fi
case "$REFINE_LINEAR_COMBO" in
  "") ;;
  0|false|False|off|OFF|no|NO) cmd+=(--no-refine-linear-combo) ;;
  1|true|True|on|ON|yes|YES) cmd+=(--refine-linear-combo) ;;
  *) echo "REFINE_LINEAR_COMBO must be empty or 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac

if [ "$ORACLE_DATA_SOURCE" = "csv" ]; then
  cmd+=(--data_dir "$DATA_DIR" --n_fit "$N_FIT" --n_probe "$N_PROBE" --data_slice "$DATA_SLICE")
  cmd+=(--y_check_abs_tol "$Y_CHECK_ABS_TOL" --y_check_rel_tol "$Y_CHECK_REL_TOL")
  if [ "$Y_CHECK_BOOL" = "0" ]; then
    cmd+=(--no_y_check)
  fi
elif [ -n "$N_FIT" ] || [ -n "$N_PROBE" ]; then
  if [ -n "$N_FIT" ]; then
    cmd+=(--n_fit "$N_FIT")
  fi
  if [ -n "$N_PROBE" ]; then
    cmd+=(--n_probe "$N_PROBE")
  fi
fi
if [ -n "$SEARCH_N_FIT" ]; then
  cmd+=(--search-n-fit "$SEARCH_N_FIT")
fi
if [ -n "$SEARCH_N_PROBE" ]; then
  cmd+=(--search-n-probe "$SEARCH_N_PROBE")
fi
if [ "$FINAL_VALIDATE_FULL_BOOL" = "1" ]; then
  cmd+=(--final-validate-full)
fi
if [ "$FINAL_VALIDATE_RERANK_BOOL" = "1" ]; then
  cmd+=(--final-validate-rerank)
fi

if [ "$PAIR_NORMAL_ENABLE" = "1" ]; then
  cmd+=(--pair_normal_enable)
fi

if [ "$DRY_RUN" = "1" ]; then
  printf "[DRY ] %s -> %s\n" "$eq_id" "$output_path"
  printf "       "
  printf "%q " "${cmd[@]}"
  printf "\n"
  exit 0
fi

printf "[START] %s\n" "$eq_id"
if "${cmd[@]}" >"$log_path" 2>&1; then
  printf "[DONE ] %s -> %s\n" "$eq_id" "$output_path"
else
  status=$?
  printf "[FAIL ] %s -> %s\n" "$eq_id" "$log_path" >&2
  exit "$status"
fi
' _ {} "$@" < "$id_file"
