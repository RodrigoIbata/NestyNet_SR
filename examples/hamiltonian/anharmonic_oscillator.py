#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Hamiltonian Discovery: Anharmonic Oscillator

This example demonstrates Hamiltonian discovery on a 1-DOF anharmonic oscillator
with quartic potential:

    H(q, p) = p²/2 + q²/2 + q⁴/4

The quartic term introduces nonlinearity, causing the phase-space trajectories
to deviate from perfect ellipses (as in the harmonic case).

Physical features:
- Energy is conserved along trajectories
- Phase portraits show distortion at higher energies
- Frequency depends on amplitude (nonlinear oscillations)

Discovery challenge:
- Can NestyNet_SR recover both the quadratic (q²) and quartic (q⁴) terms?
- Are the coefficients accurately identified?
"""

import numpy as np
import torch
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from nestynet_sr.sr_de.hamiltonian_search import (
    discover_hamiltonian_from_data,
    HamiltonianSearchConfig,
)


# ============================================================================
# True Hamiltonian and Dynamics
# ============================================================================

def true_hamiltonian(q, p):
    """True Hamiltonian: H = p²/2 + q²/2 + q⁴/4"""
    return 0.5 * p**2 + 0.5 * q**2 + 0.25 * q**4


def hamiltonian_flow(state, t):
    """Hamilton's equations: q̇ = ∂H/∂p, ṗ = -∂H/∂q"""
    q, p = state
    dqdt = p  # ∂H/∂p = p
    dpdt = -q - q**3  # -∂H/∂q = -q - q³
    return [dqdt, dpdt]


# ============================================================================
# Data Generation
# ============================================================================

def generate_trajectory(q0, p0, t_span, n_points=500):
    """Generate a single trajectory via numerical integration."""
    t = np.linspace(t_span[0], t_span[1], n_points)
    sol = odeint(hamiltonian_flow, [q0, p0], t)
    q, p = sol[:, 0], sol[:, 1]

    # Compute derivatives numerically (central differences)
    dt = t[1] - t[0]
    dqdt = np.gradient(q, dt)
    dpdt = np.gradient(p, dt)

    return t, q, p, dqdt, dpdt


def generate_training_data(n_trajectories=10, n_points_per_traj=200, seed=42):
    """Generate training data from multiple trajectories at different energies."""
    np.random.seed(seed)

    all_q, all_p, all_dqdt, all_dpdt = [], [], [], []

    # Sample initial conditions at different energy levels
    energies = np.linspace(0.5, 4.0, n_trajectories)

    for E in energies:
        # Initial condition: q0 at turning point (p0 = 0)
        # H(q0, 0) = q0²/2 + q0⁴/4 = E
        # Solve for q0 (take positive root)
        roots = np.roots([0.25, 0, 0.5, 0, -E])
        q0 = float(np.real(roots[np.isreal(roots) & (np.real(roots) > 0)][0]))
        p0 = 0.0

        # Integrate one period (roughly 2π for harmonic, varies for anharmonic)
        T_approx = 2 * np.pi  # Approximate period
        t, q, p, dqdt, dpdt = generate_trajectory(
            q0, p0, [0, T_approx], n_points=n_points_per_traj
        )

        all_q.append(q)
        all_p.append(p)
        all_dqdt.append(dqdt)
        all_dpdt.append(dpdt)

    # Stack into arrays
    q = np.concatenate(all_q)
    p = np.concatenate(all_p)
    dqdt = np.concatenate(all_dqdt)
    dpdt = np.concatenate(all_dpdt)

    # Combine into phase-space format
    z = np.column_stack([q, p])
    zdot = np.column_stack([dqdt, dpdt])

    return torch.tensor(z, dtype=torch.float32), torch.tensor(zdot, dtype=torch.float32)


# ============================================================================
# Visualization
# ============================================================================

def plot_phase_portrait(save_path=None):
    """Plot phase portrait with multiple trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Plot trajectories at different energy levels
    energies = [0.5, 1.0, 2.0, 3.5, 5.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(energies)))

    for E, color in zip(energies, colors):
        # Initial condition at turning point
        roots = np.roots([0.25, 0, 0.5, 0, -E])
        q0 = float(np.real(roots[np.isreal(roots) & (np.real(roots) > 0)][0]))
        p0 = 0.0

        # Integrate trajectory
        t, q, p, _, _ = generate_trajectory(q0, p0, [0, 20], n_points=1000)

        ax.plot(q, p, color=color, linewidth=1.5, alpha=0.8, label=f'E = {E:.1f}')

    ax.set_xlabel('Position q', fontsize=12)
    ax.set_ylabel('Momentum p', fontsize=12)
    ax.set_title('Phase Portrait: Anharmonic Oscillator\n' +
                 r'$H = \frac{p^2}{2} + \frac{q^2}{2} + \frac{q^4}{4}$',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase portrait to {save_path}")

    return fig


def plot_energy_conservation(save_path=None):
    """Plot energy conservation along a trajectory."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Generate a single trajectory
    q0, p0 = 2.5, 0.0
    t, q, p, _, _ = generate_trajectory(q0, p0, [0, 20], n_points=1000)

    # Compute energy along trajectory
    H = true_hamiltonian(q, p)
    H0 = true_hamiltonian(q0, p0)

    # Top panel: trajectory
    ax1.plot(t, q, 'b-', label='q(t)', linewidth=1.5)
    ax1.plot(t, p, 'r-', label='p(t)', linewidth=1.5)
    ax1.set_xlabel('Time t', fontsize=11)
    ax1.set_ylabel('State', fontsize=11)
    ax1.set_title('Trajectory in Time', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: energy conservation
    ax2.plot(t, H, 'g-', linewidth=1.5, label=f'H(t), H₀ = {H0:.3f}')
    ax2.axhline(H0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Initial energy')
    ax2.set_xlabel('Time t', fontsize=11)
    ax2.set_ylabel('Energy H', fontsize=11)
    ax2.set_title('Energy Conservation', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Print energy drift
    energy_drift = np.abs(H - H0).max()
    ax2.text(0.02, 0.98, f'Max drift: {energy_drift:.2e}',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved energy conservation plot to {save_path}")

    return fig


def plot_discovery_results(result, save_path=None):
    """Visualize the discovered Hamiltonian terms."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Get canonicalized coefficients
    canon = result.canonicalize_coeffs(tol=1e-3)

    # Extract term labels and coefficients
    labels = [term_str for _, term_str in canon]
    coeffs = [coeff for coeff, _ in canon]

    # Bar plot
    x = np.arange(len(labels))
    bars = ax.bar(x, coeffs, color='steelblue', alpha=0.7, edgecolor='black')

    # Highlight expected terms (q² and q⁴)
    expected_coeffs = {'x0 ** 2': 0.5, 'x0 ** 4': 0.25, 'x1 ** 2': 0.5}
    for i, (coeff, label) in enumerate(canon):
        for exp_label, exp_coeff in expected_coeffs.items():
            if exp_label in label:
                bars[i].set_color('seagreen')
                bars[i].set_alpha(0.8)
                # Add expected value annotation
                ax.plot([i], [exp_coeff], 'r*', markersize=15,
                       label='Expected' if i == 0 else '')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=11)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Discovered Hamiltonian Terms\n' +
                 f'H(z) = {result.format_hamiltonian()}',
                 fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.legend(loc='upper right', fontsize=10)

    # Add RMS info
    ax.text(0.02, 0.98, f'Training RMS: {result.rms_train:.2e}\n' +
                        f'Validation RMS: {result.rms_val:.2e}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved discovery results to {save_path}")

    return fig


# ============================================================================
# Main Example
# ============================================================================

def main():
    print("=" * 80)
    print("HAMILTONIAN DISCOVERY: ANHARMONIC OSCILLATOR")
    print("=" * 80)
    print("\nTrue Hamiltonian: H = p²/2 + q²/2 + q⁴/4")
    print("Dynamics: q̇ = p, ṗ = -q - q³")
    print()

    # Create output directory
    output_dir = Path("results/hamiltonian_anharmonic")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Visualize true dynamics
    print("[1/5] Generating phase portrait...")
    plot_phase_portrait(save_path=output_dir / "phase_portrait.png")

    print("[2/5] Checking energy conservation...")
    plot_energy_conservation(save_path=output_dir / "energy_conservation.png")

    # Step 2: Generate training data
    print("\n[3/5] Generating training data...")
    z_train, zdot_train = generate_training_data(
        n_trajectories=10, n_points_per_traj=200, seed=42
    )
    z_val, zdot_val = generate_training_data(
        n_trajectories=5, n_points_per_traj=100, seed=999
    )

    print(f"   Training samples: {z_train.shape[0]}")
    print(f"   Validation samples: {z_val.shape[0]}")

    # Step 3: Configure Hamiltonian search
    print("\n[4/5] Discovering Hamiltonian...")
    cfg = HamiltonianSearchConfig(
        n_dof=1,
        max_q_power=4,  # Include up to q⁴
        max_p_power=2,  # Include up to p²
        mechanical_split=True,  # H = T(p) + V(q)
        include_const=False,
        stlsq_lambda=1e-3,  # Sparsity threshold
    )

    print(f"   Library: max_q_power={cfg.max_q_power}, max_p_power={cfg.max_p_power}")
    print(f"   STLSQ sparsity threshold: λ = {cfg.stlsq_lambda}")

    result = discover_hamiltonian_from_data(
        z_train, zdot_train,
        z_val, zdot_val,
        cfg=cfg
    )

    # Step 4: Display results
    print("\n" + "=" * 80)
    print("DISCOVERY RESULTS")
    print("=" * 80)
    print(f"\nDiscovered Hamiltonian:\n   {result.format_hamiltonian()}\n")
    print(f"Training RMS:   {result.rms_train:.6e}")
    print(f"Validation RMS: {result.rms_val:.6e}")

    print(f"\nDiscovered {len(result.term_asts)} terms:")
    canon = result.canonicalize_coeffs(tol=1e-3)
    for coeff, term_str in canon:
        print(f"   {coeff:+.6f}  *  {term_str}")

    # Verify correctness
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    expected = {
        'x0 ** 2': 0.5,   # q² coefficient
        'x0 ** 4': 0.25,  # q⁴ coefficient
        'x1 ** 2': 0.5,   # p² coefficient
    }

    found = {term.strip('()'): coeff for coeff, term in canon}

    all_correct = True
    for term, exp_coeff in expected.items():
        if term in found:
            error = abs(found[term] - exp_coeff)
            status = "✓" if error < 0.1 else "✗"
            print(f"   {status} {term:12s}: found {found[term]:+.4f}, expected {exp_coeff:+.4f}, error = {error:.4f}")
            if error >= 0.1:
                all_correct = False
        else:
            print(f"   ✗ {term:12s}: NOT FOUND (expected {exp_coeff:+.4f})")
            all_correct = False

    # Check for spurious terms
    spurious = set(found.keys()) - set(expected.keys())
    if spurious:
        print(f"\n   ⚠ Spurious terms: {', '.join(spurious)}")
        all_correct = False

    if all_correct:
        print("\n   ✓ SUCCESS: Correctly recovered anharmonic oscillator Hamiltonian!")
    else:
        print("\n   ⚠ Partial recovery or spurious terms detected")

    # Step 5: Plot results
    print("\n[5/5] Plotting discovery results...")
    plot_discovery_results(result, save_path=output_dir / "discovered_terms.png")

    print(f"\nAll outputs saved to: {output_dir.absolute()}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
