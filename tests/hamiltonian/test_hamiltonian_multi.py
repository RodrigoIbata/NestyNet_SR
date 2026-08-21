# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Test multi-dataset Hamiltonian discovery.

Tests discovering shared H structure with dataset-specific coefficients.
Example: SHO with different spring constants.

Dataset 1: H = 0.5*p^2 + 0.5*q^2   (ω=1.0)
Dataset 2: H = 0.5*p^2 + 2.0*q^2   (ω=2.0)
Dataset 3: H = 0.5*p^2 + 4.5*q^2   (ω=3.0)

Shared support: {p^2, q^2}
Dataset-specific coefficients: a_p = 0.5 (same), a_q varies
"""

import torch
import numpy as np
from nestynet_sr.sr_de.hamiltonian_search import (
    discover_hamiltonian_from_data_multi,
    HamiltonianSearchConfig,
)


def generate_sho_data(n_samples, omega, seed):
    """Generate (z, ż) data for SHO with H = 0.5*(p^2 + ω^2*q^2)."""
    np.random.seed(seed)

    # Sample initial conditions
    q0 = np.random.randn(n_samples) * 2
    p0 = np.random.randn(n_samples) * 2

    # Dynamics: q̇ = p, ṗ = -ω^2*q
    qdot = p0
    pdot = -omega**2 * q0

    z = np.column_stack([q0, p0])
    zdot = np.column_stack([qdot, pdot])

    return torch.tensor(z, dtype=torch.float32), torch.tensor(zdot, dtype=torch.float32)


def main():
    print("=" * 70)
    print("Test: Multi-dataset Hamiltonian discovery (SHO with varying ω)")
    print("=" * 70)

    # Generate multiple datasets with different spring constants
    omegas = [1.0, 2.0, 3.0]
    dataset_ids = [f"omega={omega:.1f}" for omega in omegas]

    print(f"\n1. Generating {len(omegas)} SHO datasets with different ω:")
    z_trains = []
    zdot_trains = []

    for i, omega in enumerate(omegas):
        z, zdot = generate_sho_data(n_samples=300, omega=omega, seed=42 + i)
        z_trains.append(z)
        zdot_trains.append(zdot)

        # True H = 0.5*p^2 + 0.5*ω^2*q^2
        a_q = 0.5 * omega**2
        print(f"   Dataset {i}: ω={omega:.1f}, H = 0.5*p^2 + {a_q:.1f}*q^2")

    # Configure search
    print("\n2. Configuring multi-dataset Hamiltonian search...")
    cfg = HamiltonianSearchConfig(
        n_dof=1,
        max_q_power=2,
        max_p_power=2,
        mechanical_split=True,
        include_const=False,
        stlsq_lambda=1e-3,
    )

    # Discover shared Hamiltonian
    print("\n3. Running multi-dataset discovery (group-STLSQ)...")
    result = discover_hamiltonian_from_data_multi(
        z_trains, zdot_trains,
        cfg=cfg,
        dataset_ids=dataset_ids
    )

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nShared support: {len(result.term_asts)} terms")
    for i, term in enumerate(result.term_asts):
        term_str = "1" if term is None else repr(term)
        print(f"  φ_{i}: {term_str}")

    print("\nPer-dataset Hamiltonians:")
    for d in range(len(dataset_ids)):
        print(f"\nDataset {d} ({dataset_ids[d]}):")
        print(f"  {result.format_hamiltonian_for_dataset(d)}")
        print(f"  Training RMS: {result.rms_train[d]:.6f}")

        # Show coefficients
        for i, (term, coeff) in enumerate(zip(result.term_asts, result.coeffs[d])):
            term_str = "1" if term is None else repr(term)
            print(f"    a_{i} = {coeff.item():+.6f} ({term_str})")

    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    # Check that we found p^2 and q^2 terms
    term_strs = [("1" if t is None else repr(t)) for t in result.term_asts]

    # Find indices
    idx_p2 = None
    idx_q2 = None
    for i, ts in enumerate(term_strs):
        if 'x1 ** 2' in ts:  # x1 = p
            idx_p2 = i
        elif 'x0 ** 2' in ts:  # x0 = q
            idx_q2 = i

    if idx_p2 is None or idx_q2 is None:
        print("✗ ERROR: Did not find expected p^2 and q^2 terms")
        return

    print("✓ Found p^2 and q^2 terms in shared support")

    # Verify coefficients match expected values
    all_correct = True
    for d, omega in enumerate(omegas):
        a_p_expected = 0.5
        a_q_expected = 0.5 * omega**2

        a_p_found = result.coeffs[d, idx_p2].item()
        a_q_found = result.coeffs[d, idx_q2].item()

        p_error = abs(a_p_found - a_p_expected)
        q_error = abs(a_q_found - a_q_expected)

        status_p = "✓" if p_error < 0.1 else "✗"
        status_q = "✓" if q_error < 0.1 else "✗"

        print(f"\nDataset {d} (ω={omega:.1f}):")
        print(f"  {status_p} p^2 coeff: expected {a_p_expected:.2f}, found {a_p_found:.4f} (err: {p_error:.4f})")
        print(f"  {status_q} q^2 coeff: expected {a_q_expected:.2f}, found {a_q_found:.4f} (err: {q_error:.4f})")

        if p_error >= 0.1 or q_error >= 0.1:
            all_correct = False

    if all_correct:
        print("\n✓ SUCCESS: All datasets have correct coefficients!")
    else:
        print("\n✗ WARNING: Some coefficients deviate from expected values")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
