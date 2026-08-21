#!/usr/bin/env bash
# Run an AI Feynman campaign cheap-first, escalating only internal failures to CoE.
#
# The generated manifest is deterministic and truth-blind. Process failures are
# retried in their original phase; they are never interpreted as evidence for
# an expensive committee run.
#
# Optional environment variables:
#   CAMPAIGN_ROOT=$PWD          Benchmark workspace containing data/results
#   CAMPAIGN_RUNNER=...         Per-campaign run_allstages_all.sh
#   CHEAP_RESULTS_DIR=...       Cheap output directory (default: results)
#   COE_RESULTS_DIR=...         CoE output directory (default: results_CoE)
#   ESCALATION_MANIFEST=...     Deterministic manifest output path
#   RUN_CHEAP=1                 Run pending/retryable cheap problems
#   RUN_COE=1                   Run required/retryable CoE problems
#   ESCALATION_COE_MODE=reservoir_discovery
#   PYTHON_BIN=python           Python interpreter used for manifest operations
#   IDS="1 pb002"              Explicit problem IDs (otherwise START_ID..END_ID)
#   START_ID=0 END_ID=119       Inclusive default problem range
#   SKIP_IDS="101 111"         Explicit exclusions
#   PROBLEMS_IGNORE=...         One excluded ID per line
#
# All remaining arguments are forwarded to the campaign runner.

set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$PWD}"
CAMPAIGN_ROOT="$(cd "$CAMPAIGN_ROOT" && pwd)"
CAMPAIGN_RUNNER="${CAMPAIGN_RUNNER:-$CAMPAIGN_ROOT/scripts/run_allstages_all.sh}"
CHEAP_RESULTS_DIR="${CHEAP_RESULTS_DIR:-$CAMPAIGN_ROOT/results}"
COE_RESULTS_DIR="${COE_RESULTS_DIR:-$CAMPAIGN_ROOT/results_CoE}"
ESCALATION_MANIFEST="${ESCALATION_MANIFEST:-$CAMPAIGN_ROOT/coe_escalation_manifest.json}"
ESCALATION_COE_MODE="${ESCALATION_COE_MODE:-reservoir_discovery}"
PYTHON_BIN="${PYTHON_BIN:-python}"
START_ID="${START_ID:-0}"
END_ID="${END_ID:-119}"
IDS="${IDS:-}"
SKIP_IDS="${SKIP_IDS:-}"
PROBLEMS_IGNORE="${PROBLEMS_IGNORE:-$CAMPAIGN_ROOT/PROBLEMS.ignore}"

bool_value() {
  case "${1:-}" in
    1|true|True|TRUE|on|ON|yes|YES) printf '1' ;;
    0|false|False|FALSE|off|OFF|no|NO) printf '0' ;;
    *)
      echo "$2 must be 0/off/no or 1/on/yes" >&2
      exit 2
      ;;
  esac
}

RUN_CHEAP_BOOL="$(bool_value "${RUN_CHEAP:-1}" RUN_CHEAP)"
RUN_COE_BOOL="$(bool_value "${RUN_COE:-1}" RUN_COE)"

if [[ ! -x "$CAMPAIGN_RUNNER" ]]; then
  echo "Campaign runner is not executable: $CAMPAIGN_RUNNER" >&2
  exit 2
fi

CLI_MODE=""
CLI_PATH=""
if [[ -n "${ESCALATION_CLI:-}" ]]; then
  CLI_MODE="file"
  CLI_PATH="$ESCALATION_CLI"
elif [[ -f "$CAMPAIGN_ROOT/nestynet_sr/campaign_escalation.py" ]]; then
  CLI_MODE="module"
elif [[ -f "$SOURCE_ROOT/scripts/build_coe_escalation_manifest.py" ]]; then
  CLI_MODE="file"
  CLI_PATH="$SOURCE_ROOT/scripts/build_coe_escalation_manifest.py"
else
  echo "Could not locate the campaign escalation CLI" >&2
  exit 2
fi

if [[ "$CLI_MODE" == "file" && ! -f "$CLI_PATH" ]]; then
  echo "Campaign escalation CLI not found: $CLI_PATH" >&2
  exit 2
fi

run_cli() {
  if [[ "$CLI_MODE" == "module" ]]; then
    (
      cd "$CAMPAIGN_ROOT"
      "$PYTHON_BIN" -m nestynet_sr.campaign_escalation "$@"
    )
  else
    "$PYTHON_BIN" "$CLI_PATH" "$@"
  fi
}

MANIFEST_ID_ARGS=()
if [[ -n "$IDS" ]]; then
  MANIFEST_ID_ARGS+=(--ids "$IDS")
else
  MANIFEST_ID_ARGS+=(--start-id "$START_ID" --end-id "$END_ID")
fi

build_manifest() {
  run_cli build \
    --cheap-results "$CHEAP_RESULTS_DIR" \
    --coe-results "$COE_RESULTS_DIR" \
    --output "$ESCALATION_MANIFEST" \
    --skip-ids "$SKIP_IDS" \
    --ignore-file "$PROBLEMS_IGNORE" \
    "${MANIFEST_ID_ARGS[@]}"
}

list_actions() {
  local action_args=()
  local action
  for action in "$@"; do
    action_args+=(--action "$action")
  done
  run_cli list --manifest "$ESCALATION_MANIFEST" "${action_args[@]}"
}

run_cheap_ids() {
  local selected_ids="$1"
  shift
  echo "[escalation] Running cheap phase for: $selected_ids" >&2
  if ! env \
    COE_MODE=off \
    RESULTS_DIR="$CHEAP_RESULTS_DIR" \
    SKIP_COMPLETED=0 \
    ALLOW_FAILURES=1 \
    IDS="$selected_ids" \
    "$CAMPAIGN_RUNNER" "$@"
  then
    echo "[escalation] Cheap runner returned nonzero; classifying its artifacts." >&2
  fi
}

run_coe_ids() {
  local selected_ids="$1"
  shift
  echo "[escalation] Running CoE phase for: $selected_ids" >&2
  if ! env \
    COE_MODE="$ESCALATION_COE_MODE" \
    COE_RESERVOIR_PATHS="${COE_RESERVOIR_PATHS:-$CHEAP_RESULTS_DIR}" \
    RESULTS_DIR="$COE_RESULTS_DIR" \
    SKIP_COMPLETED=0 \
    ALLOW_FAILURES=1 \
    IDS="$selected_ids" \
    "$CAMPAIGN_RUNNER" "$@"
  then
    echo "[escalation] CoE runner returned nonzero; classifying its artifacts." >&2
  fi
}

build_manifest

if [[ "$RUN_CHEAP_BOOL" == "1" ]]; then
  cheap_ids="$(list_actions pending retry_cheap)"
  if [[ -n "$cheap_ids" ]]; then
    run_cheap_ids "$cheap_ids" "$@"
    build_manifest
  fi
fi

cheap_unresolved="$(list_actions pending retry_cheap)"
if [[ -n "$cheap_unresolved" ]]; then
  echo "[escalation] Cheap phase remains incomplete: $cheap_unresolved" >&2
  echo "[escalation] No CoE work was started for those process failures." >&2
  exit 1
fi

if [[ "$RUN_COE_BOOL" == "0" ]]; then
  coe_preview="$(list_actions run_coe retry_coe)"
  echo "[escalation] CoE execution disabled. Eligible CoE queue: ${coe_preview:-none}" >&2
  exit 0
fi

coe_ids="$(list_actions run_coe retry_coe)"
if [[ -n "$coe_ids" ]]; then
  run_coe_ids "$coe_ids" "$@"
  build_manifest
fi

unfinished_coe="$(list_actions run_coe retry_coe)"
terminal_failures="$(list_actions terminal_failure)"
if [[ -n "$unfinished_coe" || -n "$terminal_failures" ]]; then
  if [[ -n "$unfinished_coe" ]]; then
    echo "[escalation] CoE phase remains incomplete: $unfinished_coe" >&2
  fi
  if [[ -n "$terminal_failures" ]]; then
    echo "[escalation] Completed CoE runs without an eligible selection: $terminal_failures" >&2
  fi
  exit 1
fi

echo "[escalation] Campaign settled. Manifest: $ESCALATION_MANIFEST" >&2
