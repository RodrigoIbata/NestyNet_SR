# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import json
from fractions import Fraction

import pytest

from nestynet_sr.sr_core.coefficient_units import (
    coefficient_dimensions_for_support,
    monomial_dimension,
    required_coefficient_dimension,
    solve_rational_coefficient_gauge,
    solve_rational_gauge,
    term_dimension,
)
from nestynet_sr.sr_core.units import UnitSystem, sub_dim
from nestynet_sr.sr_search.ratpoly_degree_probe import probe_rational_degrees


def _dims():
    unit_system = UnitSystem(("L", "T"))
    return (
        unit_system,
        unit_system.dimless(),
        unit_system.dim({"L": 1}),
        unit_system.dim({"T": 1}),
    )


def test_monomial_dimension_uses_exact_rational_exponents():
    _unit_system, _zero, length, time = _dims()

    result = monomial_dimension((Fraction(1, 2), -2), (length, time))

    assert result == (Fraction(1, 2), Fraction(-2))


def test_required_coefficient_dimension_closes_the_term_equation():
    unit_system, _zero, length, time = _dims()
    block = unit_system.dim({"L": 2, "T": -1})
    exponent = (1, 1)

    coefficient = required_coefficient_dimension(
        block,
        exponent,
        (length, time),
    )

    assert coefficient == unit_system.dim({"L": 1, "T": -2})
    assert term_dimension(coefficient, exponent, (length, time)) == block


def test_coefficient_dimensions_are_aligned_with_the_active_support():
    unit_system, zero, length, _time = _dims()
    block = unit_system.dim({"L": 2})

    result = coefficient_dimensions_for_support(
        block,
        ((0,), (1,), (2,)),
        (length,),
    )

    assert result == (block, length, zero)


def test_low_level_algebra_rejects_rank_and_arity_mismatches():
    _unit_system, _zero, length, time = _dims()

    with pytest.raises(ValueError, match="arity"):
        monomial_dimension((1,), (length, time))
    with pytest.raises(ValueError, match="rank"):
        term_dimension((1,), (1,), (length,))


def test_dimensionless_rational_support_uses_the_same_solver_path():
    _unit_system, zero, _length, _time = _dims()

    result = solve_rational_gauge(
        target_dim=zero,
        input_dims=(zero,),
        numerator_exponents=((0,), (1,), (2,)),
        denominator_exponents=((0,), (1,)),
    )

    assert result.ok is True
    assert result.numerator_block_dim == zero
    assert result.denominator_block_dim == zero
    assert all(term.required_dim == zero for term in result.numerator)
    assert all(term.required_dim == zero for term in result.denominator)


def test_implicit_and_explicit_unitless_coefficients_are_identical():
    _unit_system, zero, length, _time = _dims()
    implicit = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((1,),),
        denominator_exponents=((0,),),
    )
    explicit = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((1,),),
        denominator_exponents=((0,),),
        numerator_coefficient_dims=(zero,),
        denominator_coefficient_dims=(zero,),
    )

    assert implicit.ok is True
    assert explicit.ok is True
    assert implicit.numerator_block_dim == explicit.numerator_block_dim
    assert implicit.denominator_block_dim == explicit.denominator_block_dim
    assert [term.required_dim for term in implicit.numerator] == [
        term.required_dim for term in explicit.numerator
    ]


def test_pb111_style_homogeneous_rational_support_is_valid():
    unit_system, zero, length, _time = _dims()
    target = unit_system.dim({"L": -2})
    numerator = ((2, 0), (1, 1), (0, 2))
    denominator = ((4, 0), (3, 1), (2, 2), (1, 3), (0, 4))

    result = solve_rational_coefficient_gauge(
        target_dim=target,
        input_dims=(length, length),
        numerator_exponents=numerator,
        denominator_exponents=denominator,
    )

    assert result.ok is True
    assert result.numerator_block_dim == unit_system.dim({"L": 2})
    assert result.denominator_block_dim == unit_system.dim({"L": 4})
    assert all(term.required_dim == zero for term in result.numerator)
    assert all(term.required_dim == zero for term in result.denominator)
    assert sub_dim(result.numerator_block_dim, result.denominator_block_dim) == target


def test_mixed_degree_anonymous_support_fails_free_const_only():
    _unit_system, zero, length, _time = _dims()

    result = solve_rational_gauge(
        target_dim=zero,
        input_dims=(length,),
        numerator_exponents=((0,), (1,)),
        denominator_exponents=((0,),),
    )

    assert result.ok is False
    assert result.code == "numerator_block_inconsistent"
    assert result.failure_side == "numerator"
    assert result.failure_index == 1
    assert result.expected_dim == zero
    assert result.actual_dim == length


def test_declared_unitful_coefficient_makes_inhomogeneous_support_valid():
    _unit_system, zero, length, _time = _dims()

    result = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((0,), (1,)),
        denominator_exponents=((0,),),
        numerator_coefficient_dims=(length, zero),
        denominator_coefficient_dims=(zero,),
    )

    assert result.ok is True
    assert result.numerator_block_dim == length
    assert result.numerator[0].required_dim == length
    assert result.numerator[0].constraint_source == "declared"
    assert result.numerator[1].required_dim == zero
    assert result.numerator[1].constraint_source == "declared"


def test_wrong_declared_unitful_coefficient_fails_the_same_block_equation():
    _unit_system, zero, length, time = _dims()

    result = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((0,), (1,)),
        denominator_exponents=((0,),),
        numerator_coefficient_dims=(time, zero),
        denominator_coefficient_dims=(zero,),
    )

    assert result.ok is False
    assert result.code == "numerator_block_inconsistent"
    assert result.expected_dim == time
    assert result.actual_dim == length


def test_monic_numerator_pivot_pins_the_rational_gauge():
    unit_system, zero, length, _time = _dims()
    inverse_length = unit_system.dim({"L": -1})

    result = solve_rational_gauge(
        target_dim=inverse_length,
        input_dims=(length,),
        numerator_exponents=((0,),),
        denominator_exponents=((1,),),
        numerator_pivot=0,
        coefficient_policy="infer",
    )

    assert result.ok is True
    assert result.gauge_status == "pinned_by_numerator"
    assert result.gauge_free is False
    assert result.numerator_block_dim == zero
    assert result.denominator_block_dim == length
    assert result.numerator[0].constraint_source == "pivot_dimensionless"
    assert result.denominator[0].required_dim == zero


def test_denominator_pivot_can_pin_a_nonzero_denominator_block():
    unit_system, zero, length, _time = _dims()
    length_squared = unit_system.dim({"L": 2})

    result = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((2,),),
        denominator_exponents=((1,),),
        denominator_pivot=0,
        coefficient_policy="infer",
    )

    assert result.ok is True
    assert result.gauge_status == "pinned_by_denominator"
    assert result.denominator_block_dim == length
    assert result.numerator_block_dim == length_squared
    assert result.numerator[0].required_dim == zero


def test_reduced_support_exposes_required_unitful_nonpivot_coefficient():
    unit_system, zero, length, _time = _dims()
    length_squared = unit_system.dim({"L": 2})

    rejected = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((2,), (0,)),
        denominator_exponents=((1,),),
        numerator_pivot=0,
    )
    accepted = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((2,), (0,)),
        denominator_exponents=((1,),),
        numerator_coefficient_dims=(zero, length_squared),
        numerator_pivot=0,
        coefficient_policy="infer",
    )

    assert rejected.ok is False
    assert rejected.code == "numerator_block_inconsistent"
    assert accepted.ok is True
    assert accepted.numerator[1].required_dim == length_squared
    assert accepted.denominator[0].required_dim == zero


def test_pivot_rejects_a_nonzero_declared_coefficient_dimension():
    _unit_system, _zero, length, _time = _dims()

    result = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((1,),),
        denominator_exponents=((0,),),
        numerator_coefficient_dims=(length,),
        numerator_pivot=0,
        coefficient_policy="infer",
    )

    assert result.ok is False
    assert result.code == "pivot_dimension_conflict"
    assert result.failure_side == "numerator"
    assert result.failure_index == 0


def test_two_dimensionless_pivots_can_conflict_with_the_target():
    _unit_system, _zero, length, _time = _dims()

    result = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((0,),),
        denominator_exponents=((0,),),
        numerator_pivot=0,
        denominator_pivot=0,
        coefficient_policy="infer",
    )

    assert result.ok is False
    assert result.code == "rational_target_mismatch"
    assert result.failure_side == "rational"


def test_infer_policy_returns_a_canonical_free_gauge():
    unit_system, zero, length, _time = _dims()
    negative_length = unit_system.dim({"L": -1})

    denominator_gauge = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((0,),),
        denominator_exponents=((0,),),
        coefficient_policy="infer",
    )
    numerator_gauge = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((0,),),
        denominator_exponents=((0,),),
        coefficient_policy="infer",
        canonical_gauge="numerator_dimensionless",
    )

    assert denominator_gauge.ok is True
    assert denominator_gauge.gauge_free is True
    assert denominator_gauge.numerator_block_dim == length
    assert denominator_gauge.denominator_block_dim == zero
    assert denominator_gauge.numerator[0].required_dim == length
    assert numerator_gauge.ok is True
    assert numerator_gauge.gauge_free is True
    assert numerator_gauge.numerator_block_dim == zero
    assert numerator_gauge.denominator_block_dim == negative_length
    assert numerator_gauge.denominator[0].required_dim == negative_length
    assert sub_dim(
        numerator_gauge.numerator_block_dim,
        numerator_gauge.denominator_block_dim,
    ) == length


def test_one_declared_coefficient_pins_both_rational_blocks():
    unit_system, _zero, length, _time = _dims()
    length_squared = unit_system.dim({"L": 2})

    result = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((0,),),
        denominator_exponents=((0,),),
        numerator_coefficient_dims=(length_squared,),
        coefficient_policy="infer",
    )

    assert result.ok is True
    assert result.gauge_status == "pinned_by_numerator"
    assert result.numerator_block_dim == length_squared
    assert result.denominator_block_dim == length
    assert result.denominator[0].required_dim == length


def test_target_outside_the_active_dimensionless_support_is_rejected():
    _unit_system, zero, length, time = _dims()

    result = solve_rational_gauge(
        target_dim=time,
        input_dims=(length,),
        numerator_exponents=((1,),),
        denominator_exponents=((0,),),
    )

    assert result.ok is False
    assert result.code == "rational_target_mismatch"
    assert result.expected_dim == time
    assert result.actual_dim == length
    assert result.actual_dim != zero


def test_every_degree_probe_pair_satisfies_the_exact_coefficient_solver():
    unit_system, _zero, length, time = _dims()
    target = unit_system.dim({"L": 1, "T": -1})
    probe = probe_rational_degrees(
        target_dim=target,
        x_dims=(length, time),
        max_total_degree=4,
    )

    assert probe.valid_pairs
    for pair in probe.valid_pairs:
        numerator = tuple(
            exponent
            for degree in sorted(pair.monomials_num)
            for exponent in pair.monomials_num[degree]
        )
        denominator = tuple(
            exponent
            for degree in sorted(pair.monomials_den)
            for exponent in pair.monomials_den[degree]
        )
        result = solve_rational_gauge(
            target_dim=target,
            input_dims=(length, time),
            numerator_exponents=numerator,
            denominator_exponents=denominator,
        )

        assert result.ok, (pair.dim_num, pair.dim_den, result.to_dict())
        assert result.numerator_block_dim == pair.dim_num
        assert result.denominator_block_dim == pair.dim_den


@pytest.mark.parametrize("bad_exponent", [0.999999999, 1.000000001, -1e-12])
def test_solver_does_not_round_near_integer_polynomial_exponents(bad_exponent):
    _unit_system, zero, length, _time = _dims()

    result = solve_rational_gauge(
        target_dim=zero,
        input_dims=(length,),
        numerator_exponents=((bad_exponent,),),
        denominator_exponents=((0,),),
    )

    assert result.ok is False
    assert result.code == "invalid_input"
    assert "exact integer" in result.reason


def test_solver_accepts_exact_integral_float_support_metadata():
    _unit_system, _zero, length, _time = _dims()

    result = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((1.0,),),
        denominator_exponents=((0.0,),),
    )

    assert result.ok is True


@pytest.mark.parametrize("bad_value", [False, 0, [], {}, ""])
@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("coefficient_policy", "invalid_coefficient_policy"),
        ("canonical_gauge", "invalid_canonical_gauge"),
    ],
)
def test_solver_does_not_treat_falsey_metadata_as_omitted(field, code, bad_value):
    _unit_system, zero, length, _time = _dims()

    result = solve_rational_gauge(
        target_dim=zero,
        input_dims=(length,),
        numerator_exponents=((0,),),
        denominator_exponents=((0,),),
        **{field: bad_value},
    )

    assert result.ok is False
    assert result.code == code


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        (
            {
                "numerator_exponents": (),
                "denominator_exponents": ((0,),),
            },
            "empty_support",
        ),
        (
            {
                "numerator_exponents": ((0,), (0,)),
                "denominator_exponents": ((0,),),
            },
            "duplicate_exponent",
        ),
        (
            {
                "numerator_exponents": ((Fraction(1, 2),),),
                "denominator_exponents": ((0,),),
            },
            "invalid_input",
        ),
        (
            {
                "numerator_exponents": ((1,),),
                "denominator_exponents": ((0,),),
                "numerator_coefficient_dims": (),
            },
            "coefficient_count_mismatch",
        ),
        (
            {
                "numerator_exponents": ((1,),),
                "denominator_exponents": ((0,),),
                "numerator_pivot": 2,
            },
            "invalid_pivot",
        ),
        (
            {
                "numerator_exponents": ((1,),),
                "denominator_exponents": ((0,),),
                "numerator_pivot": 0.0,
            },
            "invalid_pivot",
        ),
        (
            {
                "numerator_exponents": ((1,),),
                "denominator_exponents": ((0,),),
                "canonical_gauge": "some_other_gauge",
            },
            "invalid_canonical_gauge",
        ),
    ],
)
def test_solver_fails_closed_on_malformed_support_metadata(kwargs, code):
    _unit_system, zero, length, _time = _dims()

    result = solve_rational_gauge(
        target_dim=zero,
        input_dims=(length,),
        **kwargs,
    )

    assert result.ok is False
    assert result.code == code


def test_solution_certificate_is_json_ready_and_exact():
    _unit_system, zero, length, _time = _dims()

    result = solve_rational_gauge(
        target_dim=length,
        input_dims=(length,),
        numerator_exponents=((1,),),
        denominator_exponents=((0,),),
    )
    payload = result.to_dict()
    encoded = json.dumps(payload)

    assert payload["valid"] is True
    assert payload["solver"] == "coefficient_units_v1"
    assert payload["target_dim"] == ["1", "0"]
    assert payload["numerator"][0]["required_dim"] == ["0", "0"]
    assert payload["numerator"][0]["dimensionless"] is True
    assert payload["anchors"][0]["source"] == "anonymous_dimensionless"
    assert '"solver": "coefficient_units_v1"' in encoded
