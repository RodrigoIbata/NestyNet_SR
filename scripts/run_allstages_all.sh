#!/bin/bash
# Run Stage BC suite for all AI Feynman equations (pb000-pb119)
# Uses xargs for parallel execution.
# For automatic truth-blind cheap-to-CoE escalation, use
# scripts/run_allstages_escalating.sh around the campaign-local runner.
#
# Optional env vars:
#   JOBS=8                    Number of parallel workers (default: 8, or SLURM_NTASKS under SLURM)
#   THREADS_PER_JOB=1         OpenMP/BLAS threads per worker (default: 1)
#   START_ID=0                First canary ID (default: 0)
#   END_ID=119                Last canary ID (default: 119)
#   IDS="1 2 pb003"           Explicit IDs to run; overrides START_ID/END_ID
#   SKIP_IDS="101 111"        IDs to skip (also accepts pb101,pb111)
#   NDATA_TRAIN=2000          Optional training-point count passed to run_SR.py
#   NDATA_VAL=2000            Optional validation-point count passed to run_SR.py
#   BATCH_SIZE=2000           Optional batch size passed to run_SR.py
#   DATA_SLICE=0              Deterministic disjoint data block index
#   RESULTS_DIR=results       Output directory for this run/slice
#   FACTORIZED_SEARCH=auto              factorized symbolic search mode: auto/0/1 (default: auto)
#   REFINE_SKELETON=0            Enable continuous skeleton refinement only when factorized symbolic search is on (default: 0)
#   FORCE_CPU=0               Hide CUDA devices from PyTorch and run on CPU
#   COE_MODE=committee_gated   CoE mode passed to run_SR.py via run_allstages_suite.py
#   CHEAP_COE=1                Disable CoE scout lanes unless explicitly overridden
#   COE_WITNESS_PARALLELISM=1 Concurrent CoE committee witnesses per problem (default: run_SR.py default)
#   COE_SCOUT_COUNT=8          Number of initial CoE scout proposer slices
#   COE_SCOUT_PARALLELISM=1    Concurrent scout subprocesses per problem
#   COE_STAGEA_FIT_TOURNAMENT=1 Fit eligible Stage-A models on parallel slices
#   CANONICAL_INIT=1           Use deterministic data-dependent initialization
#   STAT_SELECTION=1           Enable sealed-audit statistical selection (default: 0)
#   STAT_ARGS="..."            Additional --stat-* arguments forwarded to run_SR.py
#   QUIET=1                   Suppress per-run stdout; logs still capture output

set -euo pipefail

# Go to project root (parent of scripts/)
cd "$(dirname "$0")/.."

DEFAULT_JOBS=8
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  DEFAULT_JOBS="${SLURM_NTASKS:-1}"
fi
JOBS="${JOBS:-$DEFAULT_JOBS}"
THREADS_PER_JOB="${THREADS_PER_JOB:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS_PER_JOB}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS_PER_JOB}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$THREADS_PER_JOB}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$THREADS_PER_JOB}"
START_ID="${START_ID:-0}"
END_ID="${END_ID:-119}"
IDS="${IDS:-}"
SKIP_IDS="${SKIP_IDS:-}"
DATA_SLICE="${DATA_SLICE:-0}"
RESULTS_DIR="${RESULTS_DIR:-results}"
FACTORIZED_SEARCH="${FACTORIZED_SEARCH:-auto}"
REFINE_SKELETON="${REFINE_SKELETON:-0}"
FORCE_CPU="${FORCE_CPU:-0}"
PYTHON_RUNNER=(python)
case "$FORCE_CPU" in
  0|false|False|off|OFF|no|NO) ;;
  1|true|True|on|ON|yes|YES) PYTHON_RUNNER=(env CUDA_VISIBLE_DEVICES= python) ;;
  *) echo "FORCE_CPU must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
QUIET="${QUIET:-1}"
QUIET_ARGS=()
case "$QUIET" in
  0|false|False|off|OFF|no|NO) ;;
  1|true|True|on|ON|yes|YES) QUIET_ARGS+=(--quiet) ;;
  *) echo "QUIET must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
STAT_FLAGS=()
case "${STAT_SELECTION:-0}" in
  0|false|False|off|OFF|no|NO) ;;
  1|true|True|on|ON|yes|YES) STAT_FLAGS+=(--stat_selection) ;;
  *) echo "STAT_SELECTION must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
if [[ -n "${STAT_ARGS:-}" ]]; then
  STAT_FLAGS+=(--sr_extra_args "$STAT_ARGS")
fi
FACTORIZED_SEARCH_ARGS=()
case "$FACTORIZED_SEARCH" in
  auto|Auto|AUTO)
    case "$REFINE_SKELETON" in
      0|false|False|off|OFF|no|NO) FACTORIZED_SEARCH_ARGS+=(--no-refine-skeleton) ;;
      1|true|True|on|ON|yes|YES) FACTORIZED_SEARCH_ARGS+=(--factorized-search --refine-skeleton) ;;
      *) echo "REFINE_SKELETON must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
    esac
    ;;
  0|false|False|off|OFF|no|NO) FACTORIZED_SEARCH_ARGS+=(--no-factorized-search --no-refine-skeleton) ;;
  1|true|True|on|ON|yes|YES)
    FACTORIZED_SEARCH_ARGS+=(--factorized-search)
    case "$REFINE_SKELETON" in
      0|false|False|off|OFF|no|NO) FACTORIZED_SEARCH_ARGS+=(--no-refine-skeleton) ;;
      1|true|True|on|ON|yes|YES) FACTORIZED_SEARCH_ARGS+=(--refine-skeleton) ;;
      *) echo "REFINE_SKELETON must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
    esac
    ;;
  *) echo "FACTORIZED_SEARCH must be auto or 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
DATA_ARGS=()
if [[ -n "${NDATA_TRAIN:-}" ]]; then
  DATA_ARGS+=(--ndata_train "$NDATA_TRAIN")
fi
if [[ -n "${NDATA_VAL:-}" ]]; then
  DATA_ARGS+=(--ndata_val "$NDATA_VAL")
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  DATA_ARGS+=(--batch_size "$BATCH_SIZE")
fi
if [[ "${DATA_SLICE:-0}" != "0" ]]; then
  DATA_ARGS+=(--data_slice "$DATA_SLICE")
fi
COE_ARGS=()

is_true() {
  case "${1:-}" in
    1|true|True|TRUE|on|ON|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

is_false() {
  case "${1:-}" in
    0|false|False|FALSE|off|OFF|no|NO) return 0 ;;
    *) return 1 ;;
  esac
}

append_if_set() {
  local env_name="$1"
  local opt_name="$2"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    COE_ARGS+=("$opt_name" "$value")
  fi
}

append_flag_if_true() {
  local env_name="$1"
  local opt_name="$2"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    if is_true "$value"; then
      COE_ARGS+=("$opt_name")
    elif ! is_false "$value"; then
      echo "$env_name must be 0/off/no or 1/on/yes" >&2
      exit 2
    fi
  fi
}

append_flag_if_false() {
  local env_name="$1"
  local opt_name="$2"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    if is_false "$value"; then
      COE_ARGS+=("$opt_name")
    elif ! is_true "$value"; then
      echo "$env_name must be 0/off/no or 1/on/yes" >&2
      exit 2
    fi
  fi
}

CHEAP_COE="${CHEAP_COE:-0}"
if [[ -n "$CHEAP_COE" ]]; then
  if is_true "$CHEAP_COE"; then
    COE_SCOUT_COUNT="${COE_SCOUT_COUNT:-0}"
    COE_CONTINUATION_SCOUTS="${COE_CONTINUATION_SCOUTS:-0}"
  elif ! is_false "$CHEAP_COE"; then
    echo "CHEAP_COE must be 0/off/no or 1/on/yes" >&2
    exit 2
  fi
fi
FIT_TOURNAMENT_ON=0
if is_true "${COE_STAGEA_FIT_TOURNAMENT:-0}"; then
  FIT_TOURNAMENT_ON=1
  COE_MODE="${COE_MODE:-reservoir_discovery}"
  COE_NUM_SLICES="${COE_NUM_SLICES:-16}"
  COE_START_SLICE="${COE_START_SLICE:-9}"
  COE_SCOUT_COUNT="${COE_SCOUT_COUNT:-8}"
  COE_SCOUT_PARALLELISM="${COE_SCOUT_PARALLELISM:-8}"
fi
CANONICAL_INIT="${CANONICAL_INIT:-$FIT_TOURNAMENT_ON}"
CANONICAL_ARGS=()
if is_true "$CANONICAL_INIT"; then
  CANONICAL_ARGS+=(--canonical_init)
elif ! is_false "$CANONICAL_INIT"; then
  echo "CANONICAL_INIT must be 0/off/no or 1/on/yes" >&2
  exit 2
elif [[ "$FIT_TOURNAMENT_ON" == "1" ]]; then
  echo "COE_STAGEA_FIT_TOURNAMENT=1 requires CANONICAL_INIT=1" >&2
  exit 2
fi

append_if_set COE_MODE --coe_mode
append_if_set COE_NUM_SLICES --coe_num_slices
append_if_set COE_START_SLICE --coe_start_slice
append_if_set COE_MAX_CANDIDATES --coe_max_candidates
append_if_set COE_RESERVOIR_PATHS --coe_reservoir_paths
append_if_set COE_NOISE_MULT --coe_noise_mult
append_if_set COE_REL_TOL --coe_rel_tol
append_if_set COE_MIN_VALID_FRACTION --coe_min_valid_fraction
append_if_set COE_WITNESS_PARALLELISM --coe_witness_parallelism
append_if_set COE_RESERVOIR_SUPPORT_BONUS --coe_reservoir_support_bonus
append_flag_if_true COE_STAGEB_DRY_RUN --coe_stageB_dry_run
append_if_set COE_STAGEB_GATE_SLICES --coe_stageB_gate_slices
append_if_set COE_STAGEB_INITIAL_GATE_SLICES --coe_stageB_initial_gate_slices
append_flag_if_false COE_STAGEB_REFIT_GATE --no_coe_stageB_refit_gate
append_if_set COE_STAGEB_REFIT_EPOCHS --coe_stageB_refit_epochs
append_if_set COE_STAGEB_REFIT_ESCALATE_EPOCHS --coe_stageB_refit_escalate_epochs
append_flag_if_true COE_STAGEA_DRY_RUN --coe_stageA_dry_run
append_flag_if_true COE_STAGEA_FIT_TOURNAMENT --coe_stageA_fit_tournament
append_if_set COE_STAGEA_FIT_SLICES --coe_stageA_fit_slices
append_if_set COE_STAGEA_FIT_ALPHA --coe_stageA_fit_alpha
append_if_set COE_STAGEA_FIT_COMPARISON_FRACTION --coe_stageA_fit_comparison_fraction
append_if_set COE_STAGEA_FIT_MIN_REL_IMPROVEMENT --coe_stageA_fit_min_rel_improvement
append_if_set COE_SCOUT_COUNT --coe_scout_count
append_if_set COE_SCOUT_SLICES --coe_scout_slices
append_if_set COE_SCOUT_TIMEOUT_SECONDS --coe_scout_timeout_seconds
append_if_set COE_SCOUT_PARALLELISM --coe_scout_parallelism
append_if_set COE_SCOUT_STAGEB_EPOCHS --coe_scout_stageB_epochs
append_if_set COE_SCOUT_STAGEB_MAX_OUTER_ITERS --coe_scout_stageB_max_outer_iters
append_if_set COE_SCOUT_MAX_AB_ITERS --coe_scout_max_ab_iters
append_if_set COE_SCOUT_STAGEA_MAX_PASSES --coe_scout_stageA_max_passes
append_flag_if_false COE_CONTINUATION_SCOUTS --no_coe_continuation_scouts
append_if_set COE_CONTINUATION_SCOUT_COUNT --coe_continuation_scout_count
append_if_set COE_CONTINUATION_SCOUT_MAX_PHASES --coe_continuation_scout_max_phases
append_flag_if_true COE_SCOUT_FINAL_POLISH --coe_scout_final_polish
mkdir -p "$RESULTS_DIR"

emit_ids() {
  if [[ -n "$IDS" ]]; then
    awk -v ids="$IDS" '
BEGIN {
  n = split(ids, raw, /[ ,]+/)
  emitted = 0
  for (i = 1; i <= n; i++) {
    id = raw[i]
    if (id == "") continue
    if (id ~ /^pb[0-9]+$/) {
      sub(/^pb/, "", id)
    } else if (id !~ /^[0-9]+$/) {
      printf("Invalid IDS entry: %s\n", raw[i]) > "/dev/stderr"
      exit 2
    }
    norm[i] = sprintf("pb%03d", id + 0)
    emitted = 1
  }
  if (!emitted) {
    print "IDS was set but no valid IDs were provided" > "/dev/stderr"
    exit 2
  }
  for (i = 1; i <= n; i++) {
    pb = norm[i]
    if (pb == "") continue
    if (!seen[pb]++) {
      print pb
    }
  }
}'
  else
    seq -f "pb%03g" "$START_ID" "$END_ID"
  fi
}

emit_ids \
  | awk -v skip_ids="$SKIP_IDS" '
BEGIN {
  n = split(skip_ids, raw, /[ ,]+/)
  for (i = 1; i <= n; i++) {
    if (raw[i] == "") continue
    if (raw[i] ~ /^pb[0-9][0-9][0-9]$/) {
      skip[raw[i]] = 1
    } else if (raw[i] ~ /^[0-9]+$/) {
      skip[sprintf("pb%03d", raw[i])] = 1
    }
  }
}
!($0 in skip) { print }
' \
  | xargs -P "$JOBS" -I {} "${PYTHON_RUNNER[@]}" nestynet_sr/run_allstages_suite.py \
      --only {} \
      ${QUIET_ARGS[@]+"${QUIET_ARGS[@]}"} \
      ${STAT_FLAGS[@]+"${STAT_FLAGS[@]}"} \
      --results_dir "$RESULTS_DIR" \
      ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} \
      ${COE_ARGS[@]+"${COE_ARGS[@]}"} \
      ${CANONICAL_ARGS[@]+"${CANONICAL_ARGS[@]}"} \
      ${FACTORIZED_SEARCH_ARGS[@]+"${FACTORIZED_SEARCH_ARGS[@]}"} \
      --output "$RESULTS_DIR/allstages_suite_summary_{}.json"
