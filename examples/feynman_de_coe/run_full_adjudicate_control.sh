#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-results/feynman_de_coe_full_adjudicate_${STAMP}}"
DATA_DIR="${DATA_DIR:-data/feynman_de_coe_full}"
IDS="${IDS:-002,010,100,103,114,119,121,131}"
JOBS="${JOBS:-1}"
N_POINTS="${N_POINTS:-5000}"
DE_COE_MODE="${DE_COE_MODE:-adjudicate}"
WHOLE_RHS="${WHOLE_RHS:-auto}"
REFINE_MODE="${REFINE_MODE:-rare_final_polish}"
FACTOR_SEARCH_MAX_ATTEMPTS="${FACTOR_SEARCH_MAX_ATTEMPTS:-}"
SIM_VALIDATE_MAX_CANDIDATES="${SIM_VALIDATE_MAX_CANDIDATES:-3}"
SIM_VALIDATE_TRAJ_TIME_BUDGET_S="${SIM_VALIDATE_TRAJ_TIME_BUDGET_S:-20}"
DE_COE_CSR_ON_TIES="${DE_COE_CSR_ON_TIES:-1}"

mkdir -p "$RESULTS_ROOT" "$DATA_DIR"

cmd=(
  python -u scripts/run_feynman_de_coe_control_suite.py
  --ids "$IDS"
  --engine factorized_de
  --full
  --jobs "$JOBS"
  --n_points "$N_POINTS"
  --results_root "$RESULTS_ROOT"
  --data_dir "$DATA_DIR"
  --sim_validate_max_candidates "$SIM_VALIDATE_MAX_CANDIDATES"
  --sim_validate_traj_time_budget_s "$SIM_VALIDATE_TRAJ_TIME_BUDGET_S"
  --de-coe-mode "$DE_COE_MODE"
  --factorized-de-whole-rhs "$WHOLE_RHS"
  --factorized-search-de-refine-mode "$REFINE_MODE"
)

if [[ -n "$FACTOR_SEARCH_MAX_ATTEMPTS" ]]; then
  cmd+=(--factorized-search-max-attempts "$FACTOR_SEARCH_MAX_ATTEMPTS")
fi

if [[ "$DE_COE_CSR_ON_TIES" != "0" ]]; then
  cmd+=(--de-coe-csr-on-ties)
fi

dry_run=0
for arg in "$@"; do
  if [[ "$arg" == "--dry_run" ]]; then
    dry_run=1
  fi
done

"${cmd[@]}" "$@"

if [[ "$dry_run" == "1" ]]; then
  exit 0
fi

python -u scripts/summarize_feynman_de_coe_control.py \
  "$RESULTS_ROOT" \
  --json "$RESULTS_ROOT/summary_compact.json" \
  --csv "$RESULTS_ROOT/summary_compact.csv"

echo
echo "Results root: $RESULTS_ROOT"
echo "Compact JSON: $RESULTS_ROOT/summary_compact.json"
echo "Compact CSV:  $RESULTS_ROOT/summary_compact.csv"
