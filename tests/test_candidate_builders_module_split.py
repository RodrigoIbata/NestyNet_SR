import inspect
from typing import get_type_hints

import torch

from nestynet_sr.sr_core.bridges import AtomNode
from nestynet_sr.sr_search import candidate_builders as facade


EXPECTED_GROUPS = {
    "_common": (
        "_unwrap_leaf_core",
        "_atom_inputs_match",
        "_find_matching_core",
        "_single_power_coordinate_inputs",
        "_support_is_valid",
        "_max_total_degree_from_exps",
        "_exps_override_from_tensor",
        "_exps_key",
        "_select_clear_rratpoly_pivot",
        "_move_sparse_pivot_to_end",
        "_select_sign_region",
        "_parse_pure_difference_expr",
        "_eval_input_expr_value",
        "_build_atom_input_tensor",
        "_gather_atom_teacher_data",
        "_replace_node",
    ),
    "_multivariate": (
        "_make_power_exp_ratpoly_rewrite",
        "_make_power_exp_poly_rewrite",
        "_build_quadratic_poly_candidate",
        "_build_trig_diff_affine_envelope_candidate",
        "_build_sqrt_ratpoly_candidate",
        "_build_log_ratpoly_candidate",
        "_build_ratpoly_candidates",
        "_build_ratpoly_candidate",
        "_dim_tuple_is_zero",
        "_stable_last_hard_ratio_sig",
        "_build_last_hard_ratio_candidates",
        "_build_nonlinear_sub_candidate",
        "_build_log_poly_candidate",
        "_build_sqrt_poly_candidate",
        "_build_inv_poly_candidates",
        "_build_inv_poly_candidate",
        "_build_poly_split_from_subtree_separability",
        "_build_additive_poly_split_candidate",
        "_build_power_exp_rat_candidate",
        "_build_pure_exp_rat_candidate",
    ),
    "_univariate": (
        "_build_power_exp_1d_candidate",
        "_planck_power_label",
        "_build_planck_1d_candidates",
        "_build_planck_1d_candidate",
        "_build_planck_full_1d_candidate",
        "_build_expm1_1d_candidate",
        "_build_symexp_denom_1d_candidate",
        "_make_scaling_based_rewrite",
        "_make_trig_based_rewrite",
        "_make_tanh_based_rewrite",
        "_make_affine_trig_rewrite",
        "_make_multid_trig_rewrite",
        "_make_multid_trig_pair_rewrite",
        "_build_trig_affine_envelope_candidate",
        "_make_exp_poly_rewrite",
        "_make_exp_ratpoly_rewrite",
        "_build_ratpoly_1d_candidates",
        "_build_ratpoly_1d_candidate",
        "_build_sqrt_ratpoly_1d_candidates",
        "_build_power_1d_candidate",
    ),
    "_structural": (
        "_build_ratio_invariance_candidate",
        "_build_homogeneity_peel_candidate",
        "_build_product_homogeneity_candidate",
        "_build_coupled_ratio_candidate",
        "_estimate_trig_params_on_compound",
        "_estimate_univariate_trig_amplitude",
        "_build_planck_derived_feature_candidate",
        "_build_affine_decomp_candidate",
    ),
}


def _type_hint_outcome(function):
    try:
        return "ok", get_type_hints(function)
    except Exception as exc:  # Preserve legacy unresolved nested Node aliases too.
        return "error", type(exc), str(exc)


def test_candidate_builder_facade_preserves_grouped_function_contract():
    seen = set()
    for module_name, expected_names in EXPECTED_GROUPS.items():
        module = getattr(facade, module_name)
        facade_names = getattr(facade, f"{module_name.upper()}_FUNCTIONS")
        assert facade_names == expected_names
        assert seen.isdisjoint(expected_names)
        seen.update(expected_names)

        for name in expected_names:
            forwarded = getattr(facade, name)
            implementation = getattr(module, name)
            assert inspect.unwrap(forwarded) is implementation
            assert inspect.signature(forwarded) == inspect.signature(implementation)
            assert _type_hint_outcome(forwarded) == _type_hint_outcome(implementation)
            assert forwarded.__module__ == "nestynet_sr.sr_search.candidate_builders"
            assert forwarded.__qualname__ == name

    assert len(seen) == 64


def test_facade_monkeypatch_points_are_forwarded(monkeypatch):
    sentinels = {name: object() for name in facade._PATCHABLE_GLOBAL_NAMES}
    with monkeypatch.context() as context:
        for name, sentinel in sentinels.items():
            context.setattr(facade, name, sentinel)
        facade._sync_patchable_globals()
        for module in facade._IMPLEMENTATION_MODULES:
            for name, sentinel in sentinels.items():
                if hasattr(module, name):
                    assert getattr(module, name) is sentinel

    facade._sync_patchable_globals()


def test_facade_monkeypatch_sync_is_scoped_to_the_forwarded_call(monkeypatch):
    original = facade._multivariate._fit_rational_coeffs_nd
    fake = lambda *args, **kwargs: None
    monkeypatch.setattr(facade, "_fit_rational_coeffs_nd", fake)
    target = AtomNode(kind="poly", var_idxs=(0,), kwargs={"degree": 1})

    assert facade._build_ratpoly_candidates(
        root=target,
        target=target,
        reuse={},
        train_loader=[],
        device=torch.device("cpu"),
        dtype=torch.float64,
    ) == []

    assert facade._multivariate._fit_rational_coeffs_nd is original
