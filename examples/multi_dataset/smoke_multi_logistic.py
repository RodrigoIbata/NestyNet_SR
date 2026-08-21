#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""
Multi-Dataset ODE Discovery Example: Logistic Growth with Varying Parameters

This script demonstrates multi-dataset ODE discovery by:
1. Generating synthetic logistic growth data with different parameters per dataset
2. Running group-sparse STLSQ to discover shared term support
3. Validating coefficient recovery and shared structure

Usage:
    cd examples/multi_dataset
    python smoke_multi_logistic.py

Ground Truth: u' = r*u*(1 - u/K)  ⟹  u' = r*u - (r/K)*u²

Each dataset has different growth rate r and carrying capacity K, but the same
functional form {u, u²}.

Expected Outcome:
- Shared term support: {1, u, u²}
- Dataset-specific coefficients matching true (r, r/K) values
- Excellent coefficient recovery (error < 1%)
"""

import os
import sys
import numpy as np
import pandas as pd
import subprocess
import json
from pathlib import Path

# Script directory setup
script_dir = Path(__file__).parent
repo_root = script_dir.parent.parent  # NestyNet_SR root


def logistic_solution(t, u0, r, K):
    """Exact solution to logistic ODE: u' = r*u*(1 - u/K)

    Parameters
    ----------
    t : array
        Time points
    u0 : float
        Initial condition
    r : float
        Growth rate
    K : float
        Carrying capacity

    Returns
    -------
    u : array
        Solution u(t)
    """
    return K * u0 / (u0 + (K - u0) * np.exp(-r * t))


def generate_logistic_datasets(output_dir='data/multi_logistic', seed=42):
    """Generate multiple logistic growth datasets with different parameters.

    Parameters
    ----------
    output_dir : str
        Directory to save CSV files
    seed : int
        Random seed for reproducibility

    Returns
    -------
    datasets : list of dict
        List of dataset metadata with keys: name, filepath, r, K, u0
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Define 3 datasets with different parameters
    configs = [
        {'r': 0.5, 'K': 10.0, 'u0': 0.5, 'name': 'logistic_r0.5_K10'},
        {'r': 0.8, 'K': 10.0, 'u0': 0.5, 'name': 'logistic_r0.8_K10'},
        {'r': 1.2, 'K': 10.0, 'u0': 0.5, 'name': 'logistic_r1.2_K10'},
    ]

    datasets = []
    t = np.linspace(0, 10, 5000)  # Generate enough points for non-overlapping train/val splits

    for cfg in configs:
        # Generate exact solution
        u = logistic_solution(t, cfg['u0'], cfg['r'], cfg['K'])

        # Add small noise (SNR ~ 1000)
        noise_level = np.std(u) / 1000.0
        u_noisy = u + np.random.randn(len(u)) * noise_level

        # Save as CSV
        df = pd.DataFrame({'x0': t, 'y0': u_noisy})
        filepath = os.path.join(output_dir, f"{cfg['name']}.csv")
        df.to_csv(filepath, index=False)

        datasets.append({
            'name': cfg['name'],
            'filepath': filepath,
            'r': cfg['r'],
            'K': cfg['K'],
            'u0': cfg['u0'],
            'c_u': cfg['r'],           # Coefficient of u
            'c_u2': -cfg['r'] / cfg['K']  # Coefficient of u²
        })

        print(f"Generated {cfg['name']}.csv:")
        print(f"  Ground truth: u' = {cfg['r']:.1f}*u - {cfg['r']/cfg['K']:.3f}*u²")
        print(f"  Data points: {len(t)}, Noise level: {noise_level:.2e}")

    return datasets


def run_multi_dataset_discovery(datasets, output_dir='results'):
    """Run multi-dataset ODE discovery on generated datasets.

    Parameters
    ----------
    datasets : list of dict
        Dataset metadata from generate_logistic_datasets
    output_dir : str
        Output directory for results

    Returns
    -------
    result_file : str
        Path to JSON result file
    """
    # Construct command
    filepaths = [d['filepath'] for d in datasets]
    run_de_path = str(repo_root / 'nestynet_sr' / 'run_de.py')
    cmd = [
        'python', run_de_path,
        '--filepaths'] + filepaths + [
        '--order_candidates', '1',
        '--max_u_power', '2',
        '--max_x_power', '0',       # Exclude x terms (autonomous ODE)
        '--stlsq_lambda', '0.001',  # Lower threshold for cleaner fits
        '--num_segments', '64',     # Even more segments for smooth logistic curve
        '--epochs', '10000',        # More epochs
        '--epochs_min', '3000',     # Minimum epochs before early stopping
        '--nval_patience', '1500',  # More patience for validation
        '--loss_target', '1e-13',   # Very strict loss target
        '--ndata_train', '2000',    # Use first 2000 points for training
        '--ndata_val', '2000',      # Use next 2000 points for validation
        '--output_dir', output_dir,
        '--save_json',
    ]

    print("\n" + "=" * 70)
    print("RUNNING MULTI-DATASET ODE DISCOVERY")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")

    # Run discovery
    subprocess.run(cmd, cwd=repo_root, check=True)

    # Find result file
    result_files = list(Path(repo_root / output_dir).glob('*_de.json'))
    if not result_files:
        raise FileNotFoundError("No ODE result JSON found")

    # Return most recent
    result_file = max(result_files, key=lambda p: p.stat().st_mtime)
    return str(result_file)


def validate_results(result_file, datasets):
    """Validate discovered coefficients against ground truth.

    Parameters
    ----------
    result_file : str
        Path to JSON result file
    datasets : list of dict
        Ground truth dataset metadata

    Returns
    -------
    validation : dict
        Validation statistics
    """
    with open(result_file, 'r') as f:
        result = json.load(f)

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    # Extract discovered coefficients
    coeffs = np.array(result['ode_discovery']['coefficients'])  # Shape: (D, K)
    terms = result['ode_discovery']['terms']

    print(f"\nDiscovered terms: {terms}")
    print(f"Coefficient matrix shape: {coeffs.shape}")

    # Map terms to indices (expect: constant, u, u²)
    term_map = {}
    for i, term in enumerate(terms):
        if term == '1':
            term_map['const'] = i
        elif term in ['U()', 'u']:
            term_map['u'] = i
        # Match u² exactly, not composite terms like x0*u²
        elif term in ['Pow(U(), 2)', 'U()**2', '(u ** 2)', 'u^2', '(U() ** 2)']:
            term_map['u2'] = i

    print(f"\nTerm mapping: {term_map}")

    # Check if we found the expected terms
    if 'u' not in term_map or 'u2' not in term_map:
        print("\n✗ WARNING: Expected terms {u, u²} not all found in discovered terms!")
        print(f"  Found terms: {list(term_map.keys())}")
        print("\nThis may indicate:")
        print("  1. Surrogate training quality was poor (check val_losses)")
        print("  2. Sparsity threshold (--stlsq_lambda) too high")
        print("  3. Noise level too high or insufficient data")
        return {'success': False, 'message': 'Expected terms not found'}

    # Validate each dataset
    validation = {
        'datasets': [],
        'mean_error_u': 0.0,
        'mean_error_u2': 0.0,
        'max_error_u': 0.0,
        'max_error_u2': 0.0,
    }

    for d, ds in enumerate(datasets):
        c_discovered = coeffs[d]

        # Extract coefficients
        c_const = c_discovered[term_map['const']] if 'const' in term_map else 0.0
        c_u = c_discovered[term_map['u']]
        c_u2 = c_discovered[term_map['u2']]

        # Ground truth
        c_u_true = ds['c_u']
        c_u2_true = ds['c_u2']

        # Errors
        error_u = abs(c_u - c_u_true)
        error_u2 = abs(c_u2 - c_u2_true)
        rel_error_u = error_u / abs(c_u_true) * 100
        rel_error_u2 = error_u2 / abs(c_u2_true) * 100

        validation['datasets'].append({
            'name': ds['name'],
            'c_const': c_const,
            'c_u': c_u,
            'c_u_true': c_u_true,
            'error_u': error_u,
            'rel_error_u': rel_error_u,
            'c_u2': c_u2,
            'c_u2_true': c_u2_true,
            'error_u2': error_u2,
            'rel_error_u2': rel_error_u2,
        })

        validation['mean_error_u'] += rel_error_u
        validation['mean_error_u2'] += rel_error_u2
        validation['max_error_u'] = max(validation['max_error_u'], rel_error_u)
        validation['max_error_u2'] = max(validation['max_error_u2'], rel_error_u2)

        print(f"\nDataset {d}: {ds['name']}")
        print(f"  Constant term: {c_const:12.6g} (should be ~0)")
        print(f"  Coeff(u):      {c_u:12.6g}  (true: {c_u_true:12.6g})  error: {rel_error_u:.4f}%")
        print(f"  Coeff(u²):     {c_u2:12.6g}  (true: {c_u2_true:12.6g})  error: {rel_error_u2:.4f}%")

    validation['mean_error_u'] /= len(datasets)
    validation['mean_error_u2'] /= len(datasets)

    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Mean relative error (u term):  {validation['mean_error_u']:.4f}%")
    print(f"Mean relative error (u² term): {validation['mean_error_u2']:.4f}%")
    print(f"Max relative error (u term):   {validation['max_error_u']:.4f}%")
    print(f"Max relative error (u² term):  {validation['max_error_u2']:.4f}%")

    # Check success criteria
    success = (validation['max_error_u'] < 1.0 and validation['max_error_u2'] < 1.0)

    if success:
        print("\n✓ SUCCESS: Coefficient recovery excellent (all errors < 1%)")
    else:
        print("\n✗ WARNING: Some coefficients have >1% error")

    return validation


def analyze_coefficient_patterns(datasets, validation):
    """Analyze patterns in discovered coefficients.

    Parameters
    ----------
    datasets : list of dict
        Ground truth dataset metadata
    validation : dict
        Validation results from validate_results
    """
    print("\n" + "=" * 70)
    print("COEFFICIENT PATTERN ANALYSIS")
    print("=" * 70)

    # Extract discovered coefficients
    c_u_vals = [v['c_u'] for v in validation['datasets']]
    c_u2_vals = [v['c_u2'] for v in validation['datasets']]

    # Check if c_u2 = c_u / K (should be true for all datasets with same K)
    K_estimated = []
    for i, ds in enumerate(datasets):
        c_u = c_u_vals[i]
        c_u2 = c_u2_vals[i]
        K_est = -c_u / c_u2  # Since c_u2 = -r/K and c_u = r
        K_estimated.append(K_est)
        print(f"\nDataset {i}: {ds['name']}")
        print(f"  r = {c_u:.4f} (true: {ds['r']:.4f})")
        print(f"  K = {K_est:.4f} (true: {ds['K']:.4f})  [from -c_u/c_u2]")

    # Check consistency of K across datasets
    K_mean = np.mean(K_estimated)
    K_std = np.std(K_estimated)
    K_true = datasets[0]['K']  # Assume same K for all

    print("\n" + "-" * 70)
    print("Carrying Capacity K Consistency Check:")
    print(f"  True K:      {K_true:.4f}")
    print(f"  Mean K est:  {K_mean:.4f}")
    print(f"  Std K est:   {K_std:.6f}")
    print(f"  Error:       {abs(K_mean - K_true) / K_true * 100:.4f}%")

    if K_std < 0.1:
        print("\n✓ SUCCESS: K is consistent across datasets (std < 0.1)")
        print("  → Confirms shared carrying capacity K=10")

    # Check linear relationship between r and c_u
    r_vals = [ds['r'] for ds in datasets]
    correlation = np.corrcoef(r_vals, c_u_vals)[0, 1]
    print(f"\nCorrelation(r_true, c_u_discovered): {correlation:.6f}")

    if correlation > 0.9999:
        print("✓ SUCCESS: Perfect correlation between true r and discovered c_u")


def main():
    """Main workflow for multi-dataset ODE discovery example."""
    print("=" * 70)
    print("MULTI-DATASET ODE DISCOVERY: LOGISTIC GROWTH EXAMPLE")
    print("=" * 70)
    print("\nThis example demonstrates:")
    print("  1. Generating synthetic data with shared structure but different parameters")
    print("  2. Running group-sparse STLSQ for multi-dataset discovery")
    print("  3. Validating coefficient recovery and shared term support")
    print()

    # Step 1: Generate datasets
    print("\n" + "=" * 70)
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print("=" * 70)
    datasets = generate_logistic_datasets()

    # Step 2: Run discovery
    try:
        result_file = run_multi_dataset_discovery(datasets)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: ODE discovery failed with exit code {e.returncode}")
        print("Check that run_de.py is working correctly")
        return 1
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        return 1

    # Step 3: Validate results
    try:
        validation = validate_results(result_file, datasets)
    except Exception as e:
        print(f"\n✗ ERROR: Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Analyze coefficient patterns
    try:
        analyze_coefficient_patterns(datasets, validation)
    except Exception as e:
        print(f"\n✗ WARNING: Pattern analysis failed: {e}")

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {result_file}")
    print("Data files: symbolic_regression_DE/data/multi_logistic/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
