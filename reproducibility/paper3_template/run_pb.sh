#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${EQUATIONS_TXT:-}" ]]; then
  if [[ -f "$BENCH_ROOT/data/equations.txt" ]]; then
    EQUATIONS_TXT="$BENCH_ROOT/data/equations.txt"
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
FACTORIZED_SEARCH="${FACTORIZED_SEARCH:-0}"
CANONICAL_INIT="${CANONICAL_INIT:-0}"
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
THREADS_PER_JOB="${THREADS_PER_JOB:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_JOB}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS_PER_JOB}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS_PER_JOB}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$THREADS_PER_JOB}"
FACTORIZED_SEARCH_ARGS=()
case "$FACTORIZED_SEARCH" in
  0|false|False|off|OFF|no|NO) FACTORIZED_SEARCH_ARGS+=(--no-factorized-search --no-refine-skeleton) ;;
  1|true|True|on|ON|yes|YES) FACTORIZED_SEARCH_ARGS+=(--factorized-search) ;;
  *) echo "FACTORIZED_SEARCH must be 0/off/no or 1/on/yes" >&2; exit 2 ;;
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

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 pb010 [extra run_allstages_suite.py args...]" >&2
  exit 2
fi

problem="$1"
shift

"${PYTHON_RUNNER[@]}" "$BENCH_ROOT/nestynet_sr/run_allstages_suite.py" \
  --only "$problem" \
  --data_dir "$BENCH_ROOT/data" \
  --results_dir "$BENCH_ROOT/results" \
  --output "$BENCH_ROOT/results/allstages_suite_summary_${problem}.json" \
  ${QUIET_ARGS[@]+"${QUIET_ARGS[@]}"} \
  --equations_txt "$EQUATIONS_TXT" \
  --noise_sigma_frac_y_rms "$NOISE_FRAC" \
  --stageB_overcap_fallback \
  ${CANONICAL_INIT_ARGS[@]+"${CANONICAL_INIT_ARGS[@]}"} \
  ${EVIDENCE_ARGS[@]+"${EVIDENCE_ARGS[@]}"} \
  ${FACTORIZED_SEARCH_ARGS[@]+"${FACTORIZED_SEARCH_ARGS[@]}"} \
  "$@"
