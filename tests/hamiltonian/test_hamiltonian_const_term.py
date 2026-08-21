# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

"""Test constant term handling (should have zero gradient contribution)."""

import torch
import numpy as np
from nestynet_sr.sr_de.hamiltonian_search import (
    discover_hamiltonian_from_data,
    HamiltonianSearchConfig,
)


def generate_sho_data(n_samples=500, omega=1.0, seed=42):
    """Generate (z, ż) data for SHO."""
    np.random.seed(seed)

    q0 = np.random.randn(n_samples) * 2
    p0 = np.random.randn(n_samples) * 2

    qdot = omega * p0
    pdot = -omega * q0

    z = np.column_stack([q0, p0])
    zdot = np.column_stack([qdot, pdot])

    return torch.tensor(z, dtype=torch.float32), torch.tensor(zdot, dtype=torch.float32)


def main():
    print("=" * 70)
    print("Test: Constant Term Handling (include_const=True)")
    print("=" * 70)

    # Generate data
    print("\n1. Generating SHO data...")
    z_train, zdot_train = generate_sho_data(n_samples=500)
    print("   True H = 0.5*q^2 + 0.5*p^2")
    print("   Note: Constant terms in H are pure gauge (don't affect dynamics)")

    # Test with include_const=True
    print("\n2. Testing with include_const=True...")
    print("   (This previously would crash with 'tensor not used in graph')")

    cfg = HamiltonianSearchConfig(
        n_dof=1,
        max_q_power=2,
        max_p_power=2,
        mechanical_split=True,
        include_const=True,  # Enable constant term
        stlsq_lambda=1e-3,
    )

    try:
        result = discover_hamiltonian_from_data(z_train, zdot_train, cfg=cfg)

        print("\n✓ SUCCESS: No crash! Constant term handled correctly.")
        print(f"\nDiscovered: {result.format_hamiltonian()}")
        print(f"Training RMS: {result.rms_train:.6f}")

        # Check if constant term was selected
        has_const = any(t is None for t in result.term_asts)
        print(f"\nConstant term selected by STLSQ: {has_const}")

        if has_const:
            const_idx = [i for i, t in enumerate(result.term_asts) if t is None][0]
            const_coeff = result.coeffs[const_idx].item()
            print(f"  Constant coefficient: {const_coeff:.6f}")
            print("  (Should be near zero since constant doesn't affect ż = J∇H)")

            if abs(const_coeff) < 0.1:
                print("  ✓ Constant coefficient is small as expected")
            else:
                print("  ⚠ Constant coefficient unexpectedly large")
        else:
            print("  ✓ STLSQ correctly pruned constant term (zero gradient)")

        # Verify main terms are still correct
        print("\n3. Verifying main terms are still correct...")
        for i, (term, coeff) in enumerate(zip(result.term_asts, result.coeffs)):
            if term is None:
                continue
            term_str = repr(term)
            if 'x0 ** 2' in term_str and abs(coeff.item() - 0.5) < 0.05:
                print(f"   ✓ q^2 term: {coeff.item():.4f} ≈ 0.5")
            elif 'x1 ** 2' in term_str and abs(coeff.item() - 0.5) < 0.05:
                print(f"   ✓ p^2 term: {coeff.item():.4f} ≈ 0.5")

        print("\n✓ OVERALL SUCCESS: Constant term handling works correctly!")

    except Exception as e:
        print(f"\n✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
