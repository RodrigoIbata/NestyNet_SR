#!/usr/bin/env bash
# Run the noiseless AI Feynman all-stages suite from ../SRBench_0.000.
# This script keeps outputs local to this benchmark directory and links the
# source tree instead of copying it.
#
# Optional env vars:
#   JOBS=8                    Number of parallel workers (default: 8, or SLURM_NTASKS under SLURM)
#   DATA_DIR=/path/to/data    Exact benchmark CSV/metadata directory (default: BENCH_ROOT/data)
#   THREADS_PER_JOB=1         OpenMP/BLAS threads per worker (default: 1)
#   START_ID=0                First canary ID (default: 0)
#   END_ID=119                Last canary ID (default: 119)
#   IDS="1 2 pb003"           Explicit IDs to run; overrides START_ID/END_ID
#   SKIP_IDS="101 111"        IDs to skip (also accepts pb101,pb111)
#   PROBLEMS_IGNORE=PROBLEMS.ignore  Optional local file of IDs to skip, one per line
#   NOISE_FRAC=0.000          Known sigma_y / RMS(y) noise fraction
#   NDATA_TRAIN=2000          Optional training-point count passed to run_SR.py
#   NDATA_VAL=2000            Optional validation-point count passed to run_SR.py
#   BATCH_SIZE=2000           Optional batch size passed to run_SR.py
#   DATA_SLICE=0              Deterministic disjoint data block index
#   RESULTS_DIR=results       Output directory for this run/slice; defaults to results_CoE in CoE mode
#   FACTORIZED_SEARCH=0       Enable/disable factorized symbolic search in Stage B
#   REFINE_SKELETON=0         Enable continuous skeleton refinement only when factorized search is on
#   CANONICAL_INIT=1          Enable NestyNet canonical initialization for Stage-A NN teachers
#   EVIDENCE=1                Enable evidence/segment-prior LM (set 0 to disable)
#   FORCE_CPU=0               Hide CUDA devices from PyTorch and run on CPU
#   QUIET=1                   Suppress per-run stdout; logs still capture output
#   ALLOW_FAILURES=0          If 1, write failed summaries/logs but return success to the caller
#   SKIP_COMPLETED=0          If 1, skip IDs already completed in RESULTS_DIR
#   COE_MODE=off              CoE mode: off, audit_final, final_adjudicate, committee_gated, or reservoir_discovery
#   CHEAP_COE=0               If 1, use the older cheap CoE defaults: no scouts, 5 gate slices
#   COE_NUM_SLICES=16         Optional CoE validation-slice count; balanced default uses slices 9..24
#   COE_START_SLICE=9         Optional first CoE slice id; balanced default leaves 1..8 for scouts
#   COE_MAX_CANDIDATES=16     Optional max final candidates audited by CoE
#   COE_RESERVOIR_PATHS=...   Optional comma/path-separator list of reservoir reports/dirs
#   COE_RESERVOIR_SUPPORT_BONUS=... Optional support bonus for tied reservoir candidates
#   COE_WITNESS_PARALLELISM=1 Optional maximum concurrent CoE committee witness evaluations
#   COE_SCOUT_COUNT=8         Optional bounded scout proposer count for balanced reservoir_discovery
#   COE_SCOUT_SLICES=...      Optional explicit scout slice ids, overriding COE_SCOUT_COUNT
#   COE_SCOUT_PARALLELISM=1   Optional maximum concurrent scout subprocesses; each scout uses one BLAS/OpenMP thread
#   COE_SCOUT_STAGEA_MAX_PASSES=1 Optional Stage-A pass cap for bounded scout proposer runs (0 disables)
#   COE_CONTINUATION_SCOUTS=1 Optional Stage-A/Stage-B continuation scouts when scouts are enabled
#   COE_CONTINUATION_SCOUT_COUNT=$COE_SCOUT_COUNT Optional scout count cap for each continuation phase
#   COE_CONTINUATION_SCOUT_MAX_PHASES=6 Optional continuation phase cap (0 disables)
#   COE_STAGEB_DRY_RUN=0      If 1, log observe-only Stage-B CoE risk diagnostics
#   COE_STAGEB_GATE_SLICES=11 Optional max slices for Stage-B CoE gating
#   COE_STAGEB_INITIAL_GATE_SLICES=5 Optional initial slices for adaptive Stage-B CoE gating
#   COE_STAGEB_REFIT_GATE=1   If 1, allow short committee refits for NN-containing Stage-B gates
#   COE_STAGEB_REFIT_EPOCHS=200 Optional epoch cap for those short committee refits
#   COE_STAGEB_REFIT_ESCALATE_EPOCHS=0 Optional Tier-1 epoch cap before a refit veto
#   COE_STAGEA_DRY_RUN=0      If 1, log observe-only Stage-A CoE risk diagnostics
#   EQUATIONS_TXT=/path       Override equations.txt path
#   CODE_ROOT=/path/to/repo   Fallback source checkout if benchmark data has no equations.txt

set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$BENCH_ROOT/data}"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "data directory not found: $DATA_DIR" >&2
  exit 2
fi
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
if [[ -z "${EQUATIONS_TXT:-}" ]]; then
  if [[ -f "$DATA_DIR/equations.txt" ]]; then
    EQUATIONS_TXT="$DATA_DIR/equations.txt"
  else
    CODE_ROOT="${CODE_ROOT:-$BENCH_ROOT/../NestyNet_SR}"
    EQUATIONS_TXT="$CODE_ROOT/data/equations.txt"
  fi
fi
if [[ ! -f "$EQUATIONS_TXT" ]]; then
  echo "equations.txt not found: $EQUATIONS_TXT" >&2
  exit 2
fi
NOISE_FRAC="${NOISE_FRAC:-0.000}"
DATA_SLICE="${DATA_SLICE:-0}"
COE_MODE="${COE_MODE:-off}"
case "$COE_MODE" in
  off|OFF|0|false|False|no|NO) DEFAULT_RESULTS_DIR="$BENCH_ROOT/results" ;;
  audit_final|final_adjudicate|committee_gated|reservoir_discovery) DEFAULT_RESULTS_DIR="$BENCH_ROOT/results_CoE" ;;
  *) echo "COE_MODE must be off, audit_final, final_adjudicate, committee_gated, or reservoir_discovery" >&2; exit 2 ;;
esac
RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"
FACTORIZED_SEARCH="${FACTORIZED_SEARCH:-0}"
REFINE_SKELETON="${REFINE_SKELETON:-0}"
CANONICAL_INIT="${CANONICAL_INIT:-1}"
EVIDENCE="${EVIDENCE:-1}"
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
ALLOW_FAILURES="${ALLOW_FAILURES:-0}"
case "$ALLOW_FAILURES" in
  0|false|False|off|OFF|no|NO) ALLOW_FAILURES_BOOL=0 ;;
  1|true|True|on|ON|yes|YES) ALLOW_FAILURES_BOOL=1 ;;
  *) echo "ALLOW_FAILURES must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
case "$SKIP_COMPLETED" in
  0|false|False|off|OFF|no|NO) SKIP_COMPLETED_BOOL=0 ;;
  1|true|True|on|ON|yes|YES) SKIP_COMPLETED_BOOL=1 ;;
  *) echo "SKIP_COMPLETED must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
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
EVIDENCE_ARGS=()
case "$EVIDENCE" in
  0|false|False|off|OFF|no|NO) ;;
  1|true|True|on|ON|yes|YES) EVIDENCE_ARGS+=(--evidence) ;;
  *) echo "EVIDENCE must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
CANONICAL_INIT_ARGS=()
case "$CANONICAL_INIT" in
  0|false|False|off|OFF|no|NO) ;;
  1|true|True|on|ON|yes|YES) CANONICAL_INIT_ARGS+=(--canonical_init) ;;
  *) echo "CANONICAL_INIT must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
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
CHEAP_COE="${CHEAP_COE:-${cheap_CoE:-0}}"
case "$CHEAP_COE" in
  0|false|False|off|OFF|no|NO) CHEAP_COE_BOOL=0 ;;
  1|true|True|on|ON|yes|YES) CHEAP_COE_BOOL=1 ;;
  *) echo "CHEAP_COE must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
esac
case "$COE_MODE" in
  committee_gated|reservoir_discovery)
    if [[ "$CHEAP_COE_BOOL" == "1" ]]; then
      : "${COE_NUM_SLICES:=25}"
      : "${COE_START_SLICE:=0}"
      : "${COE_STAGEB_GATE_SLICES:=5}"
      : "${COE_STAGEB_INITIAL_GATE_SLICES:=3}"
      : "${COE_SCOUT_COUNT:=0}"
      : "${COE_SCOUT_PARALLELISM:=1}"
      : "${COE_SCOUT_STAGEA_MAX_PASSES:=1}"
      : "${COE_CONTINUATION_SCOUTS:=1}"
      : "${COE_CONTINUATION_SCOUT_COUNT:=$COE_SCOUT_COUNT}"
      : "${COE_CONTINUATION_SCOUT_MAX_PHASES:=6}"
    else
      COE_MODE="reservoir_discovery"
      : "${COE_NUM_SLICES:=16}"
      : "${COE_START_SLICE:=9}"
      : "${COE_STAGEB_GATE_SLICES:=11}"
      : "${COE_STAGEB_INITIAL_GATE_SLICES:=5}"
      : "${COE_SCOUT_COUNT:=8}"
      : "${COE_SCOUT_PARALLELISM:=1}"
      : "${COE_SCOUT_STAGEA_MAX_PASSES:=1}"
      : "${COE_CONTINUATION_SCOUTS:=1}"
      : "${COE_CONTINUATION_SCOUT_COUNT:=$COE_SCOUT_COUNT}"
      : "${COE_CONTINUATION_SCOUT_MAX_PHASES:=6}"
    fi
    ;;
esac
COE_ARGS=()
case "$COE_MODE" in
  off|OFF|0|false|False|no|NO) ;;
  audit_final|final_adjudicate|committee_gated|reservoir_discovery)
    COE_ARGS+=(--coe_mode "$COE_MODE")
    if [[ -n "${COE_NUM_SLICES:-}" ]]; then
      COE_ARGS+=(--coe_num_slices "$COE_NUM_SLICES")
    fi
    if [[ -n "${COE_START_SLICE:-}" ]]; then
      COE_ARGS+=(--coe_start_slice "$COE_START_SLICE")
    fi
    if [[ -n "${COE_MAX_CANDIDATES:-}" ]]; then
      COE_ARGS+=(--coe_max_candidates "$COE_MAX_CANDIDATES")
    fi
    if [[ -n "${COE_RESERVOIR_PATHS:-}" ]]; then
      COE_ARGS+=(--coe_reservoir_paths "$COE_RESERVOIR_PATHS")
    fi
    if [[ -n "${COE_NOISE_MULT:-}" ]]; then
      COE_ARGS+=(--coe_noise_mult "$COE_NOISE_MULT")
    fi
    if [[ -n "${COE_REL_TOL:-}" ]]; then
      COE_ARGS+=(--coe_rel_tol "$COE_REL_TOL")
    fi
    if [[ -n "${COE_MIN_VALID_FRACTION:-}" ]]; then
      COE_ARGS+=(--coe_min_valid_fraction "$COE_MIN_VALID_FRACTION")
    fi
    if [[ -n "${COE_WITNESS_PARALLELISM:-}" ]]; then
      COE_ARGS+=(--coe_witness_parallelism "$COE_WITNESS_PARALLELISM")
    fi
    if [[ -n "${COE_RESERVOIR_SUPPORT_BONUS:-}" ]]; then
      COE_ARGS+=(--coe_reservoir_support_bonus "$COE_RESERVOIR_SUPPORT_BONUS")
    fi
    case "${COE_STAGEB_DRY_RUN:-0}" in
      0|false|False|off|OFF|no|NO) ;;
      1|true|True|on|ON|yes|YES) COE_ARGS+=(--coe_stageB_dry_run) ;;
      *) echo "COE_STAGEB_DRY_RUN must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
    esac
    if [[ -n "${COE_STAGEB_GATE_SLICES:-}" ]]; then
      COE_ARGS+=(--coe_stageB_gate_slices "$COE_STAGEB_GATE_SLICES")
    fi
    if [[ -n "${COE_STAGEB_INITIAL_GATE_SLICES:-}" ]]; then
      COE_ARGS+=(--coe_stageB_initial_gate_slices "$COE_STAGEB_INITIAL_GATE_SLICES")
    fi
    case "${COE_STAGEB_REFIT_GATE:-1}" in
      1|true|True|on|ON|yes|YES) ;;
      0|false|False|off|OFF|no|NO) COE_ARGS+=(--no_coe_stageB_refit_gate) ;;
      *) echo "COE_STAGEB_REFIT_GATE must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
    esac
    if [[ -n "${COE_STAGEB_REFIT_EPOCHS:-}" ]]; then
      COE_ARGS+=(--coe_stageB_refit_epochs "$COE_STAGEB_REFIT_EPOCHS")
    fi
    if [[ -n "${COE_STAGEB_REFIT_ESCALATE_EPOCHS:-}" ]]; then
      COE_ARGS+=(--coe_stageB_refit_escalate_epochs "$COE_STAGEB_REFIT_ESCALATE_EPOCHS")
    fi
    case "${COE_STAGEA_DRY_RUN:-0}" in
      0|false|False|off|OFF|no|NO) ;;
      1|true|True|on|ON|yes|YES) COE_ARGS+=(--coe_stageA_dry_run) ;;
      *) echo "COE_STAGEA_DRY_RUN must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
    esac
    if [[ -n "${COE_SCOUT_COUNT:-}" ]]; then
      COE_ARGS+=(--coe_scout_count "$COE_SCOUT_COUNT")
    fi
    if [[ -n "${COE_SCOUT_SLICES:-}" ]]; then
      COE_ARGS+=(--coe_scout_slices "$COE_SCOUT_SLICES")
    fi
    if [[ -n "${COE_SCOUT_TIMEOUT_SECONDS:-}" ]]; then
      COE_ARGS+=(--coe_scout_timeout_seconds "$COE_SCOUT_TIMEOUT_SECONDS")
    fi
    if [[ -n "${COE_SCOUT_PARALLELISM:-}" ]]; then
      COE_ARGS+=(--coe_scout_parallelism "$COE_SCOUT_PARALLELISM")
    fi
    if [[ -n "${COE_SCOUT_STAGEB_EPOCHS:-}" ]]; then
      COE_ARGS+=(--coe_scout_stageB_epochs "$COE_SCOUT_STAGEB_EPOCHS")
    fi
    if [[ -n "${COE_SCOUT_STAGEB_MAX_OUTER_ITERS:-}" ]]; then
      COE_ARGS+=(--coe_scout_stageB_max_outer_iters "$COE_SCOUT_STAGEB_MAX_OUTER_ITERS")
    fi
    if [[ -n "${COE_SCOUT_MAX_AB_ITERS:-}" ]]; then
      COE_ARGS+=(--coe_scout_max_ab_iters "$COE_SCOUT_MAX_AB_ITERS")
    fi
    if [[ -n "${COE_SCOUT_STAGEA_MAX_PASSES:-}" ]]; then
      COE_ARGS+=(--coe_scout_stageA_max_passes "$COE_SCOUT_STAGEA_MAX_PASSES")
    fi
    case "${COE_CONTINUATION_SCOUTS:-1}" in
      0|false|False|off|OFF|no|NO) COE_ARGS+=(--no_coe_continuation_scouts) ;;
      1|true|True|on|ON|yes|YES) ;;
      *) echo "COE_CONTINUATION_SCOUTS must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
    esac
    if [[ -n "${COE_CONTINUATION_SCOUT_COUNT:-}" ]]; then
      COE_ARGS+=(--coe_continuation_scout_count "$COE_CONTINUATION_SCOUT_COUNT")
    fi
    if [[ -n "${COE_CONTINUATION_SCOUT_MAX_PHASES:-}" ]]; then
      COE_ARGS+=(--coe_continuation_scout_max_phases "$COE_CONTINUATION_SCOUT_MAX_PHASES")
    fi
    case "${COE_SCOUT_FINAL_POLISH:-0}" in
      0|false|False|off|OFF|no|NO) ;;
      1|true|True|on|ON|yes|YES) COE_ARGS+=(--coe_scout_final_polish) ;;
      *) echo "COE_SCOUT_FINAL_POLISH must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
    esac
    ;;
  *) echo "COE_MODE must be off, audit_final, final_adjudicate, committee_gated, or reservoir_discovery" >&2; exit 2 ;;
esac

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
PROBLEMS_IGNORE="${PROBLEMS_IGNORE:-$BENCH_ROOT/PROBLEMS.ignore}"
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

filter_completed_ids() {
  python -c '
import glob
import json
import os
import sys

skip_completed = sys.argv[1] == "1"
result_dir = sys.argv[2]


def is_completed(pb: str) -> bool:
    if glob.glob(os.path.join(result_dir, f"{pb}_*.report.json")):
        return True
    for path in glob.glob(os.path.join(result_dir, f"allstages_suite_summary_{pb}.json")):
        try:
            with open(path) as f:
                summary = json.load(f)
            if int(summary.get("successful", 0) or 0) > 0:
                return True
            for result in summary.get("results") or []:
                if result.get("success") is True:
                    return True
        except Exception:
            continue
    return False


kept = []
skipped = []
for raw in sys.stdin:
    pb = raw.strip()
    if not pb:
        continue
    if skip_completed and is_completed(pb):
        skipped.append(pb)
    else:
        kept.append(pb)

if skipped:
    print(
        f"[run_allstages_all] Skipping {len(skipped)} completed problem(s) in {result_dir}: "
        + " ".join(skipped),
        file=sys.stderr,
    )
for pb in kept:
    print(pb)
' "$SKIP_COMPLETED_BOOL" "$RESULTS_DIR"
}

run_status=0
if emit_ids \
  | awk -v skip_ids="$SKIP_IDS" -v ignore_file="$PROBLEMS_IGNORE" '
function add_skip(raw_id,    id) {
  id = raw_id
  sub(/#.*/, "", id)
  gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", id)
  if (id == "") return
  if (id ~ /^pb[0-9][0-9][0-9]$/) {
    skip[id] = 1
  } else if (id ~ /^[0-9]+$/) {
    skip[sprintf("pb%03d", id)] = 1
  }
}
BEGIN {
  n = split(skip_ids, raw, /[ ,]+/)
  for (i = 1; i <= n; i++) {
    add_skip(raw[i])
  }
  if (ignore_file != "" && (getline line < ignore_file) >= 0) {
    add_skip(line)
    while ((getline line < ignore_file) > 0) {
      add_skip(line)
    }
    close(ignore_file)
  }
}
!($0 in skip) { print }
' \
  | filter_completed_ids \
  | xargs -P "$JOBS" -I {} "${PYTHON_RUNNER[@]}" "$BENCH_ROOT/nestynet_sr/run_allstages_suite.py" \
      --only {} \
      --data_dir "$DATA_DIR" \
      --results_dir "$RESULTS_DIR" \
      --output "$RESULTS_DIR/allstages_suite_summary_{}.json" \
      ${QUIET_ARGS[@]+"${QUIET_ARGS[@]}"} \
      --equations_txt "$EQUATIONS_TXT" \
      --noise_sigma_frac_y_rms "$NOISE_FRAC" \
      ${DATA_ARGS[@]+"${DATA_ARGS[@]}"} \
      ${COE_ARGS[@]+"${COE_ARGS[@]}"} \
      --stageB_overcap_fallback \
      ${CANONICAL_INIT_ARGS[@]+"${CANONICAL_INIT_ARGS[@]}"} \
      ${EVIDENCE_ARGS[@]+"${EVIDENCE_ARGS[@]}"} \
      ${FACTORIZED_SEARCH_ARGS[@]+"${FACTORIZED_SEARCH_ARGS[@]}"} \
      "$@"
then
  run_status=0
else
  run_status=$?
fi

if (( run_status != 0 )); then
  if (( ALLOW_FAILURES_BOOL )); then
    echo "[run_allstages_all] One or more problem runs failed (status=${run_status}); continuing because ALLOW_FAILURES=1." >&2
    exit 0
  fi
  exit "$run_status"
fi
