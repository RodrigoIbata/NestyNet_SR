#!/bin/bash
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2025 Rodrigo Ibata

# Convenience script to run all three logistic growth tests

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR="$REPO_ROOT/results/logistic_growth"

echo "========================================================================"
echo "Logistic Growth ODE Discovery - Full Test Suite"
echo "========================================================================"
echo ""
echo "This will run three tests:"
echo "  1. Baseline STLSQ (no templates)"
echo "  2. VarPro + heuristic PowerLaw template"
echo "  3. VarPro + LM-optimized PowerLaw template"
echo ""
echo "Estimated time: 15-30 minutes (depending on CPU)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Default parameters (can be overridden via environment variables)
EPOCHS=${EPOCHS:-2000}
TEMPLATE_LM_EPOCHS=${TEMPLATE_LM_EPOCHS:-200}

echo ""
echo "Running with:"
echo "  Surrogate epochs: $EPOCHS"
echo "  Template LM epochs: $TEMPLATE_LM_EPOCHS"
echo ""

# Run test suite
python "$SCRIPT_DIR/smoke_logistic_discovery.py" \
    --generate \
    --epochs $EPOCHS \
    --template_lm_epochs $TEMPLATE_LM_EPOCHS

echo ""
echo "========================================================================"
echo "All tests complete!"
echo "========================================================================"
echo ""
echo "Output files:"
echo "  $OUT_DIR/*_baseline_de.human"
echo "  $OUT_DIR/*_heuristic_de.human"
echo "  $OUT_DIR/*_lm_de.human"
echo "  $OUT_DIR/*_baseline_de.json"
echo "  $OUT_DIR/*_heuristic_de.json"
echo "  $OUT_DIR/*_lm_de.json"
echo ""
echo "Next steps:"
echo "  1. Review latest equation: cat $OUT_DIR/logistic_growth_de.human"
echo "  2. Create plots: python $SCRIPT_DIR/plot_results.py"
echo "  3. View data: python $SCRIPT_DIR/plot_logistic_data.py"
echo ""
