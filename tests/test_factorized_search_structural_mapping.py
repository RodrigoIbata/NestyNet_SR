# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

from nestynet_sr.sr_search.factorized_search.explorer import mapping_is_structural


def test_mapping_is_structural_accepts_identity_and_monomial():
    assert mapping_is_structural({})
    assert mapping_is_structural({"kind": "identity"})
    assert mapping_is_structural({"kind": "monomial"})
    assert mapping_is_structural({"kind": "mono"})
    assert mapping_is_structural({"kind": "affine"})


def test_mapping_is_structural_rejects_nonstructural_curve_fitters():
    assert mapping_is_structural({"kind": "poly", "coeffs": [0.0, 1.0]})
    assert not mapping_is_structural({"kind": "poly", "coeffs": [0.0, 1.0, 0.0]})
    assert not mapping_is_structural(
        {"kind": "pade", "numer": [0.0, 1.0], "denom": [1.0, 0.0]}
    )
