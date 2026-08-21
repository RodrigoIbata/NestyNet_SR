#!/bin/bash
# Run Stage A suite for all AI Feynman equations (pb000-pb119)
# Uses xargs for parallel execution.
#
# Optional env vars:
#   JOBS=8                    Number of parallel workers (default: 8)
#   START_ID=0                First canary ID (default: 0)
#   END_ID=119                Last canary ID (default: 119)
#   SKIP_IDS="101 111"        IDs to skip (also accepts pb101,pb111)

set -euo pipefail

# Go to project root (parent of scripts/)
cd "$(dirname "$0")/.."

JOBS="${JOBS:-8}"
START_ID="${START_ID:-0}"
END_ID="${END_ID:-119}"
SKIP_IDS="${SKIP_IDS:-}"

seq -f "pb%03g" "$START_ID" "$END_ID" \
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
  | xargs -P "$JOBS" -I {} python nestynet_sr/run_stageA_suite.py --only {} --quiet
