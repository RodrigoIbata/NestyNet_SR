#!/bin/bash
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2025 Rodrigo Ibata

# Convenience script to run all three Lane-Emden tests

set -e  # Exit on error

echo "========================================================================"
echo "Lane-Emden ODE Discovery - Full Test Suite"
echo "========================================================================"
echo ""
echo "This will run three tests:"
echo "  1. Baseline STLSQ (no templates)"
echo "  2. VarPro + heuristic PowerLaw template"
echo "  3. VarPro + LM-optimized PowerLaw template"
echo ""
echo "Estimated time: 20-40 minutes (depending on CPU)"
echo "Note: 2nd-order ODE discovery is more computationally intensive"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Default parameters (can be overridden via environment variables)
EPOCHS=${EPOCHS:-1200}
TEMPLATE_LM_EPOCHS=${TEMPLATE_LM_EPOCHS:-120}
POLYTROPIC_INDEX=${POLYTROPIC_INDEX:-1.5}
XI_MIN=${XI_MIN:-0.2}
XI_MAX=${XI_MAX:-2.0}

echo ""
echo "Running with:"
echo "  Surrogate epochs: $EPOCHS"
echo "  Template LM epochs: $TEMPLATE_LM_EPOCHS"
echo "  Polytropic index n: $POLYTROPIC_INDEX"
echo "  ξ range: [$XI_MIN, $XI_MAX]"
echo ""

# Run test suite
python smoke_lane_emden_discovery.py \
    --generate \
    --n $POLYTROPIC_INDEX \
    --xi_min $XI_MIN \
    --xi_max $XI_MAX \
    --epochs $EPOCHS \
    --template_lm_epochs $TEMPLATE_LM_EPOCHS

echo ""
echo "========================================================================"
echo "All tests complete!"
echo "========================================================================"
echo ""
echo "Output files:"
echo "  ../../results/lane_emden/*.human"
echo "  ../../results/lane_emden/*.json"
echo ""
echo "Next steps:"
echo "  1. Review latest equation: cat ../../results/lane_emden/lane_emden_de.human"
echo "  2. Create plots: python plot_results.py --n $POLYTROPIC_INDEX"
echo "  3. View data: python plot_lane_emden_data.py --n $POLYTROPIC_INDEX"
echo ""
