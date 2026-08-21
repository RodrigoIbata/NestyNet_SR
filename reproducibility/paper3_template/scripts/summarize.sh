#!/usr/bin/env bash
set -euo pipefail

BENCH_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CODE_ROOT="${CODE_ROOT:-$BENCH_ROOT/../NestyNet_SR}"
EQUATIONS_TXT="${EQUATIONS_TXT:-$BENCH_ROOT/data/equations.txt}"
if [[ ! -f "$EQUATIONS_TXT" ]]; then
  echo "equations.txt not found: $EQUATIONS_TXT" >&2
  exit 2
fi

python "$CODE_ROOT/scripts/summarize_aifeyn_results.py" \
  "$BENCH_ROOT/results" \
  --equations "$EQUATIONS_TXT" \
  --csv "$BENCH_ROOT/results/summary.csv" \
  "$@"
