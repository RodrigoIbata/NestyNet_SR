# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

from __future__ import annotations

from nestynet_sr.sr_search.factorized_search.basis_compile import (
    basis_expr_key,
    basis_structure_signature_key,
    canonicalize_basis_expr,
)
from nestynet_sr.sr_search.factorized_search.basis_state import BasisState, FeatureBlock, admit_basis_state_to_beam


def test_canonicalize_basis_expr_flattens_and_sorts_commutative_forms():
    left = ("mul", ("mul", ("var", 0), ("var", 1)), ("var", 2))
    right = ("mul", ("var", 2), ("mul", ("var", 1), ("var", 0)))

    left_canon = canonicalize_basis_expr(left)
    right_canon = canonicalize_basis_expr(right)

    assert basis_expr_key(left_canon) == basis_expr_key(right_canon)
    assert basis_structure_signature_key(left) == basis_structure_signature_key(right)


def test_admit_basis_state_to_beam_dedupes_associative_compiled_expr_variants():
    block = FeatureBlock(
        family="quadratic",
        atoms=(("var", 0), ("var", 1), ("var", 2)),
        head_type="linear",
    )
    state_a = BasisState(
        blocks=(block,),
        fit_loss=1.0e-4,
        probe_loss=1.0e-4,
        complexity=3.0,
        compiled_expr=("mul", ("mul", ("var", 0), ("var", 1)), ("var", 2)),
    )
    state_b = BasisState(
        blocks=(block,),
        fit_loss=2.0e-4,
        probe_loss=2.0e-4,
        complexity=3.0,
        compiled_expr=("mul", ("var", 2), ("mul", ("var", 1), ("var", 0))),
    )

    beam = admit_basis_state_to_beam([state_a], state_b, beam_width=3)

    assert len(beam) == 1
    assert float(beam[0].probe_loss) == 1.0e-4


def test_canonicalize_basis_expr_accepts_inverse_trig_nodes():
    expr = ("asin", ("mul", ("var", 0), ("sin", ("var", 1))))

    canon = canonicalize_basis_expr(expr)

    assert basis_expr_key(canon) == "asin((sin(x1)*x0))"
    assert basis_structure_signature_key(expr) == "('asin', ('mul', (('sin', ('var', 1)), ('var', 0))))"
