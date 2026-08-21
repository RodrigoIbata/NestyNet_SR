#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" != "" ]]; then
  RESULTS_ROOT="$1"
elif [[ "${RESULTS_ROOT:-}" != "" ]]; then
  RESULTS_ROOT="$RESULTS_ROOT"
else
  RESULTS_ROOT="$(find results -maxdepth 1 -type d -name 'feynman_de_coe_full_adjudicate_*' | sort | tail -n 1)"
fi

if [[ "$RESULTS_ROOT" == "" || ! -d "$RESULTS_ROOT" ]]; then
  echo "Could not find a results directory to summarize." >&2
  exit 1
fi

python scripts/summarize_feynman_de_coe_control.py \
  "$RESULTS_ROOT" \
  --json "$RESULTS_ROOT/summary_compact.json" \
  --csv "$RESULTS_ROOT/summary_compact.csv"

echo "Compact JSON: $RESULTS_ROOT/summary_compact.json"
echo "Compact CSV:  $RESULTS_ROOT/summary_compact.csv"
