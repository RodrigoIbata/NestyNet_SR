# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Quick test of Hamiltonian discovery on simple harmonic oscillator.

The true Hamiltonian is:
    H = 0.5*p^2 + 0.5*q^2

The dynamics are:
    q̇ = ∂H/∂p = p
    ṗ = -∂H/∂q = -q
"""

import torch
import numpy as np
from nestynet_sr.sr_de.hamiltonian_search import (
    discover_hamiltonian_from_data,
    HamiltonianSearchConfig,
)


def generate_sho_data(n_samples=500, omega=1.0, seed=42):
    """Generate (z, ż) data for simple harmonic oscillator."""
    np.random.seed(seed)

    # Sample initial conditions
    q0 = np.random.randn(n_samples) * 2
    p0 = np.random.randn(n_samples) * 2

    # Dynamics: q̇ = ωp, ṗ = -ωq for SHO with H = 0.5*ω*(p^2 + q^2)
    # For simplicity, use ω=1 so H = 0.5*(p^2 + q^2)
    qdot = omega * p0
    pdot = -omega * q0

    # Stack into (N, 2) arrays
    z = np.column_stack([q0, p0])
    zdot = np.column_stack([qdot, pdot])

    return torch.tensor(z, dtype=torch.float32), torch.tensor(zdot, dtype=torch.float32)


def main():
    print("=" * 60)
    print("Test: Hamiltonian discovery for Simple Harmonic Oscillator")
    print("=" * 60)

    # Generate data
    print("\n1. Generating synthetic SHO data...")
    z_train, zdot_train = generate_sho_data(n_samples=500, omega=1.0)
    z_val, zdot_val = generate_sho_data(n_samples=200, omega=1.0, seed=123)

    print(f"   Training: {z_train.shape[0]} samples")
    print(f"   Validation: {z_val.shape[0]} samples")
    print("   True H = 0.5*q^2 + 0.5*p^2")

    # Configure search
    print("\n2. Configuring Hamiltonian search...")
    cfg = HamiltonianSearchConfig(
        n_dof=1,
        max_q_power=2,
        max_p_power=2,
        mechanical_split=True,  # H = T(p) + V(q)
        include_const=False,
        stlsq_lambda=1e-3,
    )
    print(f"   n_dof={cfg.n_dof}")
    print(f"   max_q_power={cfg.max_q_power}")
    print(f"   max_p_power={cfg.max_p_power}")
    print(f"   mechanical_split={cfg.mechanical_split}")

    # Discover Hamiltonian
    print("\n3. Running Hamiltonian discovery...")
    result = discover_hamiltonian_from_data(
        z_train, zdot_train,
        z_val, zdot_val,
        cfg=cfg
    )

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\n{result.format_hamiltonian()}")

    print(f"\nTraining RMS: {result.rms_train:.6f}")
    if result.rms_val is not None:
        print(f"Validation RMS: {result.rms_val:.6f}")

    print(f"\nDiscovered {len(result.term_asts)} terms:")
    for i, (term, coeff) in enumerate(zip(result.term_asts, result.coeffs)):
        term_str = "1" if term is None else repr(term)
        print(f"  {i}: {coeff.item():+.6f} * {term_str}")

    # Verify correctness
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Expected: coefficients near 0.5 for q^2 and p^2
    canon = result.canonicalize_coeffs(tol=1e-2)

    # Find q^2 and p^2 terms
    found_q2 = False
    found_p2 = False

    for coeff, term_str in canon:
        if 'x0 ** 2' in term_str and abs(coeff - 0.5) < 0.1:
            found_q2 = True
            print(f"✓ Found q^2 term with coeff ≈ 0.5: {coeff:.4f}")
        elif 'x1 ** 2' in term_str and abs(coeff - 0.5) < 0.1:
            found_p2 = True
            print(f"✓ Found p^2 term with coeff ≈ 0.5: {coeff:.4f}")

    if found_q2 and found_p2:
        print("\n✓ SUCCESS: Discovered correct SHO Hamiltonian!")
    else:
        print("\n✗ WARNING: Did not find expected q^2 and p^2 terms")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
