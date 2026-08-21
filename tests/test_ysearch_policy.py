# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import unittest
from types import SimpleNamespace

from nestynet_sr.run_SR import (
    _append_final_simplification_path_state,
    _compound_coordinate_variants,
    _decorate_simplification_path_y_space,
    _format_simplification_path,
    _identity_outer_affine_units_ok,
    _probe_compound_outer_affine_variants,
    _protect_exact_stageB_seed_in_final_polish,
    _stageA_status_message,
    _stageB_adjudication_key,
    _stageB_candidate_metrics,
    _stageB_generic_approximant_signature,
    _stageB_portfolio_can_stop_early,
    _stageB_shadow_rescue_reason,
    _stageB_shortlist_names,
    _stageB_shortlist_source_map,
)

from nestynet_sr.sr_search.ysearch_controller import (
    YSearchControllerConfig,
    YSearchState,
    run_depth1_ysearch,
    run_ysearch_beam,
)
from nestynet_sr.sr_search.ysearch_ranker import VirtualProbeHint, rank_virtual_hints, select_virtual_indices
from nestynet_sr.sr_search.ysearch_signals import structural_progress
from nestynet_sr.sr_search.search import (
    _stageA_identity_target_good,
    _stageA_initial_fit_restart_allowed,
)
from nestynet_sr.sr_core import AtomNode


class YSearchPolicyTests(unittest.TestCase):
    def _trigger(self, payload):
        ok, reasons = structural_progress(
            payload.get("parent_stagea_signals", {}),
            payload.get("stagea_signals", {}),
        )
        payload["reasons"] = reasons
        return ok

    def test_full_compound_is_provisional_not_confirmed(self):
        ok, reasons = structural_progress({}, {"full_compound_solved": 1.0})
        self.assertFalse(ok)
        self.assertIn("provisional:full_compound_compressed", reasons)

    def test_proxy_hints_do_not_confirm_by_themselves(self):
        proxies = {
            "simplicity_hint_ok": 1.0,
            "trig_affine_conf": 0.99,
            "sep_score": 0.95,
            "best_split_score": 0.99,
            "logquad_ok": 1.0,
        }
        ok, reasons = structural_progress({}, proxies)
        self.assertFalse(ok)
        self.assertTrue(all(str(r).startswith("provisional:") for r in reasons))

    def test_outer_affine_certificate_confirms(self):
        ok, reasons = structural_progress({}, {"outer_affine_confirmed": 1.0})
        self.assertTrue(ok)
        self.assertIn("outer_affine_confirmed", reasons)

    def test_ranker_prefers_outer_affine_certificate_over_proxy_score(self):
        hints = [
            VirtualProbeHint(
                idx=1,
                name="cos",
                domain_ok_frac=1.0,
                candidate_flag=True,
                sep_has_split=False,
                sep_proposals=0,
                trig_strength=50.0,
                virtual_mse=1.0e-12,
            ),
            VirtualProbeHint(
                idx=2,
                name="sin",
                domain_ok_frac=1.0,
                candidate_flag=False,
                sep_has_split=False,
                sep_proposals=0,
                trig_strength=0.0,
                virtual_mse=1.0e-3,
                outer_affine_confirmed=True,
                outer_affine_rms_rel=1.0e-10,
                outer_affine_domain_ok_frac=1.0,
            ),
        ]
        ranked = rank_virtual_hints(hints)
        self.assertEqual(ranked[0].name, "sin")

    def test_controller_best_trial_prefers_confirmed_branch(self):
        payloads = {
            ("cos",): {
                "val_loss_base": 1.0e-12,
                "stagea_signals": {"full_compound_compressed": 1.0},
                "branch_confirmation": "provisional",
            },
            ("sin",): {
                "val_loss_base": 1.0e-4,
                "stagea_signals": {"outer_affine_confirmed": 1.0},
                "branch_confirmation": "outer_affine_confirmed",
                "parent_stagea_signals": {},
            },
        }

        res = run_ysearch_beam(
            parent_state=YSearchState(y_stack=tuple()),
            candidate_names=["cos", "sin"],
            evaluate_state=lambda stack: dict(payloads[tuple(stack)]),
            parent_val_loss_base=1.0,
            cfg=YSearchControllerConfig(max_depth=1, beam=2, expand_k=2),
            strong_structure_trigger_fn=self._trigger,
        )
        self.assertIsNotNone(res.best_trial)
        self.assertEqual(res.best_trial.name, "sin")
        self.assertIn("outer_affine_confirmed", res.best_trial.payload.get("reasons", []))

    def test_depth1_controller_does_not_treat_compression_as_trigger(self):
        def eval_candidate(name):
            return {
                "val_loss_base": 1.0e-12,
                "stagea_signals": {"full_compound_compressed": 1.0},
                "branch_confirmation": "provisional",
                "parent_stagea_signals": {},
            }

        res = run_depth1_ysearch(
            parent_state=YSearchState(y_stack=tuple()),
            candidate_names=["identity_like"],
            evaluate_candidate=eval_candidate,
            parent_val_loss_base=1.0e-13,
            cfg=YSearchControllerConfig(max_depth=1, beam=1, expand_k=1, eps_parent_loss=1.0e-12),
            strong_structure_trigger_fn=self._trigger,
        )
        self.assertFalse(res.accepted_trials)
        self.assertFalse(res.best_trial.strong_structure_trigger)

    def test_portfolio_selection_keeps_close_proxy_ties(self):
        hints = [
            VirtualProbeHint(0, "cos", 1.0, True, False, 0, virtual_mse=1.0),
            VirtualProbeHint(1, "exp", 1.0, True, False, 0, virtual_mse=1.12),
            VirtualProbeHint(2, "log", 1.0, True, False, 0, virtual_mse=10.0),
        ]
        selected = select_virtual_indices(hints, 1, margin_decades=0.15, max_k=3)
        self.assertEqual(selected, [0, 1])

    def test_pb025_outer_affine_probe_identifies_sin_of_y(self):
        import torch
        from nestynet_sr.sr_search.outer_peel import probe_affine_outer_peels_on_z

        x0 = torch.linspace(-0.95, 0.95, 64, dtype=torch.float64)
        x1 = torch.linspace(-2.5, 2.5, 64, dtype=torch.float64)
        X0, X1 = torch.meshgrid(x0, x1, indexing="ij")
        z = (X0.reshape(-1) * torch.sin(X1.reshape(-1))).clamp(-0.999, 0.999)
        y = torch.asin(z)

        ranked = probe_affine_outer_peels_on_z(
            y=y,
            z=z,
            transform_names=["cos", "sin", "exp", "log"],
            min_points=256,
            min_domain_frac=0.20,
        )
        by_name = {r.name: r for r in ranked}
        self.assertEqual(ranked[0].name, "sin")
        self.assertLess(by_name["sin"].rms_rel, 1.0e-10)
        self.assertGreaterEqual(by_name["sin"].domain_ok_frac, 0.995)

    def test_pb030_outer_affine_probe_identifies_sin_on_reciprocal_coordinate(self):
        import torch
        from nestynet_sr.sr_core.bridges import MulNode, PowNode, Var

        x0 = torch.linspace(1.0, 2.0, 24, dtype=torch.float64)
        x1 = torch.linspace(2.0, 5.0, 24, dtype=torch.float64)
        x2 = torch.linspace(1.0, 5.0, 24, dtype=torch.float64)
        X0, X1, X2 = torch.meshgrid(x0, x1, x2, indexing="ij")
        X = torch.stack([X0.reshape(-1), X1.reshape(-1), X2.reshape(-1)], dim=1)
        u = X0.reshape(-1) / (X1.reshape(-1) * X2.reshape(-1))
        y = torch.asin(u)
        z_expr = MulNode(Var(1), MulNode(Var(2), PowNode(Var(0), -1.0)))

        entries, by_name, _identity_confirmed = _probe_compound_outer_affine_variants(
            y_values=y,
            z_expr=z_expr,
            x_values=X,
            Nxvars=3,
            transform_names=["sin", "cos", "reciprocal"],
            rms_thr=1.0e-6,
            dom_thr=0.995,
            min_points=256,
            min_domain_frac=0.20,
        )
        self.assertEqual(entries[0]["name"], "sin")
        self.assertEqual(by_name["sin"]["coordinate"], "z_inv")
        self.assertTrue(by_name["sin"]["confirmed"])
        self.assertLess(by_name["sin"]["rms_rel"], 1.0e-10)
        self.assertGreaterEqual(by_name["sin"]["domain_ok_frac"], 0.995)

        variants = _compound_coordinate_variants(z_expr)
        self.assertEqual([v.name for v in variants], ["z", "z_inv"])

    def test_pb010_outer_affine_probe_identifies_identity_of_y(self):
        import torch
        from nestynet_sr.sr_search.outer_peel import probe_affine_outer_peels_on_z

        x0 = torch.linspace(1.0, 5.0, 64, dtype=torch.float64)
        x1 = torch.linspace(1.0, 5.0, 64, dtype=torch.float64)
        X0, X1 = torch.meshgrid(x0, x1, indexing="ij")
        z = X0.reshape(-1) * X1.reshape(-1)
        y = z

        ranked = probe_affine_outer_peels_on_z(
            y=y,
            z=z,
            transform_names=["identity", "square", "sqrt", "reciprocal"],
            min_points=256,
            min_domain_frac=0.20,
        )
        by_name = {r.name: r for r in ranked}
        self.assertIn("identity", by_name)
        self.assertLess(by_name["identity"].rms_rel, 1.0e-12)
        self.assertLess(abs(by_name["identity"].b) / float(torch.median(y.abs())), 1.0e-12)

    def test_identity_outer_affine_shortcut_requires_unit_safe_z(self):
        from nestynet_sr.sr_core.bridges import MulNode, Var
        from nestynet_sr.sr_core.units import UnitSystem

        us = UnitSystem(("L", "T"))
        z_expr = MulNode(Var(0), Var(1))
        payload = {
            "unit_system": us,
            "x_dims": (us.dim({"L": 1}), us.dim({"T": 1})),
            "y_dim": us.dim({"L": 1, "T": 1}),
        }
        ok, _reason = _identity_outer_affine_units_ok(
            z_expr,
            payload,
            intercept=1.0e-12,
            y_scale=1.0,
        )
        self.assertTrue(ok)

        bad_dim = dict(payload)
        bad_dim["y_dim"] = us.dim({"L": 1})
        ok, reason = _identity_outer_affine_units_ok(
            z_expr,
            bad_dim,
            intercept=1.0e-12,
            y_scale=1.0,
        )
        self.assertFalse(ok)
        self.assertIn("compound dim", reason)

        ok, reason = _identity_outer_affine_units_ok(
            z_expr,
            payload,
            intercept=0.1,
            y_scale=1.0,
        )
        self.assertFalse(ok)
        self.assertIn("offset too large", reason)

    def test_simplification_path_reports_original_y_space_for_y_transform(self):
        path = [
            {
                "step": 0,
                "stage": "C",
                "action": "SymPy simplification",
                "expression": "x0*sin(x1)",
                "mse_raw": 1.0e-30,
                "n_params": 1,
                "detail": "ops=2",
            }
        ]
        decorated = _decorate_simplification_path_y_space(
            path,
            y_transform_name="sin",
            phi_expr_str="x0*sin(x1)",
            y_expr_str="asin(x0*sin(x1))",
        )
        self.assertEqual(decorated[-1]["expression"], "asin(x0*sin(x1))")
        self.assertEqual(decorated[-1]["phi_expression"], "x0*sin(x1)")
        self.assertIn("sin(y) = x0*sin(x1)", decorated[-1]["recipe"])

        rendered = _format_simplification_path(decorated)
        self.assertIn("phi(y): x0*sin(x1)", rendered)
        self.assertIn("y: asin(x0*sin(x1))", rendered)
        self.assertIn("=== Final: asin(x0*sin(x1)) ===", rendered)

    def test_simplification_path_appends_unresolved_final_scaffold(self):
        path = [
            {
                "step": 0,
                "stage": "A",
                "action": "pure NN surrogate",
                "expression": "NN[x0, x1, x2]",
                "mse_raw": 1.0,
                "n_params": 96,
            }
        ]
        final_expr = "(NN[x0, x1] + NN[x2])"
        updated = _append_final_simplification_path_state(
            path,
            final_expr=final_expr,
            val_loss=0.25,
            n_params=64,
            num_nn_atoms=2,
        )
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[-1]["expression"], final_expr)
        self.assertEqual(updated[-1]["stage"], "Final")
        self.assertEqual(updated[-1]["num_nn_atoms"], 2)

        rendered = _format_simplification_path(updated)
        self.assertIn("reported final state", rendered)
        self.assertIn("nn_atoms=2", rendered)
        self.assertIn(f"=== Final: {final_expr} ===", rendered)

    def test_simplification_path_does_not_duplicate_matching_final_state(self):
        path = [
            {
                "step": 0,
                "stage": "B",
                "action": "rewrite",
                "expression": "(NN[x0, x1] + NN[x2])",
                "mse_raw": 0.25,
                "n_params": 64,
            }
        ]
        updated = _append_final_simplification_path_state(
            path,
            final_expr="(NN[x0, x1] + NN[x2])",
            val_loss=0.25,
            n_params=64,
            num_nn_atoms=2,
        )
        self.assertEqual(len(updated), 1)

    def test_stageA_status_message_distinguishes_compound_from_no_separability(self):
        self.assertEqual(
            _stageA_status_message("compound_outer_confirmed", False),
            "Stage A found full-variable compound compression; outer map confirmed.",
        )
        self.assertEqual(
            _stageA_status_message("compound_unresolved", False),
            "Stage A found full-variable compound compression; outer map unresolved.",
        )
        self.assertEqual(
            _stageA_status_message("unresolved", False),
            "No Stage A separability found.",
        )
        self.assertIsNone(_stageA_status_message("split_confirmed", True))

    def test_stageA_identity_target_good_uses_validation_or_training_loss(self):
        ok, reason = _stageA_identity_target_good(
            val_loss=5.0e-8,
            train_loss=1.0e-5,
            loss_target_eff=1.0e-7,
        )
        self.assertTrue(ok)
        self.assertIn("validation-loss", reason)

        ok, reason = _stageA_identity_target_good(
            val_loss=5.0e-6,
            train_loss=5.0e-8,
            loss_target_eff=1.0e-7,
        )
        self.assertTrue(ok)
        self.assertIn("training-loss", reason)

        ok, reason = _stageA_identity_target_good(
            val_loss=5.0e-6,
            train_loss=2.0e-6,
            loss_target_eff=1.0e-7,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "")

    def test_stageB_shortlist_runs_identity_before_proxy_candidates(self):
        ranked = {
            "ranked": [
                {"name": "sqrt"},
                {"name": "square"},
                {"name": "cos"},
                {"name": "identity"},
            ]
        }
        names = _stageB_shortlist_names(
            final_y_op_name="identity",
            outer_peel_ranked=ranked,
            available_y_names=["identity", "sqrt", "square", "cos"],
            virtual_top_names=["square", "cos", "sqrt"],
            top_k=3,
        )
        self.assertEqual(names, ["identity", "square", "cos", "sqrt"])

    def test_stageB_shortlist_adds_best_scatter_to_virtual_top3(self):
        ranked = {
            "ranked": [
                {"name": "sqrt"},
                {"name": "identity"},
                {"name": "square"},
                {"name": "cos"},
            ]
        }
        names = _stageB_shortlist_names(
            final_y_op_name="identity",
            outer_peel_ranked=ranked,
            available_y_names=["identity", "sqrt", "square", "cos", "sqrt1p"],
            virtual_top_names=["square", "sqrt1p", "cos"],
            top_k=3,
        )
        self.assertEqual(names, ["identity", "square", "sqrt1p", "cos", "sqrt"])

    def test_stageB_shortlist_preserves_one_structural_reserve_after_top3(self):
        names = _stageB_shortlist_names(
            final_y_op_name="identity",
            outer_peel_ranked=None,
            available_y_names=["identity", "expneg", "arctan", "cos", "square"],
            virtual_top_names=["expneg", "arctan", "cos", "square"],
            virtual_reserved_names=["square"],
            top_k=3,
        )
        self.assertEqual(names, ["identity", "expneg", "arctan", "cos", "square"])

        sources = _stageB_shortlist_source_map(
            names=names,
            final_y_op_name="identity",
            outer_peel_ranked=None,
            virtual_top_names=["expneg", "arctan", "cos", "square"],
            virtual_reserved_names=["square"],
        )
        self.assertEqual(
            sources["square"],
            ["virtual", "virtual_structural_reserve"],
        )

    def test_stageB_shortlist_adds_confirmed_compound_affine_candidate(self):
        ranked = {
            "compound_z_affine": [{"name": "sin", "confirmed": True}],
            "ranked": [{"name": "sqrt"}, {"name": "identity"}, {"name": "cos"}],
        }
        names = _stageB_shortlist_names(
            final_y_op_name="identity",
            outer_peel_ranked=ranked,
            available_y_names=["identity", "sin", "sqrt", "cos", "exp", "log"],
            virtual_top_names=["cos", "exp", "log"],
            top_k=3,
        )
        self.assertEqual(names, ["sin", "identity", "cos", "exp", "log", "sqrt"])

    def test_stageB_shortlist_ignores_unconfirmed_compound_affine_candidate(self):
        ranked = {
            "compound_z_affine": [{"name": "sin", "confirmed": False}],
            "ranked": [{"name": "sqrt"}],
        }
        names = _stageB_shortlist_names(
            final_y_op_name="identity",
            outer_peel_ranked=ranked,
            available_y_names=["identity", "sin", "sqrt", "cos"],
            virtual_top_names=["cos"],
            top_k=3,
        )
        self.assertEqual(names, ["identity", "cos", "sqrt"])

    def test_stageB_shortlist_source_map_records_independent_channels(self):
        ranked = {
            "compound_z_affine": [{"name": "sin", "confirmed": True}],
            "ranked": [{"name": "sqrt"}],
        }
        names = ["cos", "sqrt", "sin", "identity"]
        sources = _stageB_shortlist_source_map(
            names=names,
            final_y_op_name="identity",
            outer_peel_ranked=ranked,
            virtual_top_names=["cos"],
        )
        self.assertEqual(sources["cos"], ["virtual"])
        self.assertEqual(sources["sqrt"], ["scatter"])
        self.assertEqual(sources["sin"], ["outer_affine_confirmed"])
        self.assertEqual(sources["identity"], ["baseline"])

    def test_stageB_adjudication_prefers_identity_over_unreduced_proxy(self):
        class Model:
            def num_parameters(self):
                return 640

        identity = SimpleNamespace(
            model=Model(),
            val_loss=1.0e-12,
            loss_good_enough_eff=1.0e-13,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=[],
            num_nn_atoms=1,
            num_multivar_nn_atoms=1,
            max_nn_arity=2,
        )
        cos_proxy = SimpleNamespace(
            model=Model(),
            val_loss=0.5e-12,
            loss_good_enough_eff=1.0e-13,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=[],
            num_nn_atoms=1,
            num_multivar_nn_atoms=1,
            max_nn_arity=2,
        )
        self.assertLess(
            _stageB_adjudication_key(identity, y_name="identity", rank=1),
            _stageB_adjudication_key(cos_proxy, y_name="cos", rank=0),
        )

    def test_stageB_adjudication_prefers_structural_rewrite_over_identity(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        identity = SimpleNamespace(
            model=Model(640),
            val_loss=1.0e-14,
            loss_good_enough_eff=1.0e-15,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=[],
            num_nn_atoms=1,
            num_multivar_nn_atoms=1,
            max_nn_arity=2,
        )
        sqrt_rewrite = SimpleNamespace(
            model=Model(3),
            val_loss=1.0e-10,
            loss_good_enough_eff=1.0e-15,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["homogeneity_peel", "exp"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        self.assertLess(
            _stageB_adjudication_key(sqrt_rewrite, y_name="sqrt", rank=0),
            _stageB_adjudication_key(identity, y_name="identity", rank=1),
        )

    def test_stageB_adjudication_prefers_confirmed_exact_branch_over_approx_chain(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        sin_confirmed = SimpleNamespace(
            model=Model(1),
            val_loss=4.0e-33,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["monomial_deg1"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        arctan_approx = SimpleNamespace(
            model=Model(5),
            val_loss=5.0e-8,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["ratpoly_1d[4]", "ratpoly_1d[5]"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        self.assertLess(
            _stageB_adjudication_key(
                sin_confirmed,
                y_name="sin",
                rank=0,
                y_sources=["virtual", "outer_affine_confirmed", "stageA_selected"],
            ),
            _stageB_adjudication_key(arctan_approx, y_name="arctan", rank=2, y_sources=["virtual"]),
        )

    def test_stageB_adjudication_does_not_reward_more_rewrites_over_loss(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        exact_one_rewrite = SimpleNamespace(
            model=Model(3),
            val_loss=1.0e-30,
            loss_good_enough_eff=1.0e-12,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["sqrt_poly"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        approximate_two_rewrites = SimpleNamespace(
            model=Model(3),
            val_loss=1.0e-7,
            loss_good_enough_eff=1.0e-12,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["ratpoly_1d[4]", "ratpoly_1d[5]"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        self.assertLess(
            _stageB_adjudication_key(exact_one_rewrite, y_name="cos", rank=1),
            _stageB_adjudication_key(approximate_two_rewrites, y_name="arctan", rank=2),
        )

    def test_stageB_adjudication_treats_exact_losses_as_equivalent(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        simple_exact = SimpleNamespace(
            model=Model(1),
            val_loss=1.0e-12,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["monomial_deg1[z_inv]"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        complex_exact = SimpleNamespace(
            model=Model(9),
            val_loss=1.0e-30,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["mono_peel_nn_resid", "ratpoly_1d[9]"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        self.assertLess(
            _stageB_adjudication_key(simple_exact, y_name="sin", rank=3),
            _stageB_adjudication_key(complex_exact, y_name="identity", rank=4),
        )

    def test_stageB_adjudication_treats_noise_floor_losses_as_equivalent(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        simple_identity = SimpleNamespace(
            model=Model(3),
            val_loss=1.1159e-8,
            acceptance_noise_floor_raw=1.114e-8,
            loss_good_enough_eff=1.0e-10,
            loss_acceptable_eff=1.0e-5,
            enabled_patterns=["exp_square"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 11.0},
        )
        complex_transform = SimpleNamespace(
            model=Model(4),
            val_loss=1.0947e-8,
            acceptance_noise_floor_raw=1.114e-8,
            loss_good_enough_eff=1.0e-10,
            loss_acceptable_eff=1.0e-5,
            enabled_patterns=["exp_poly"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 44.7},
        )

        self.assertLess(
            _stageB_adjudication_key(simple_identity, y_name="identity", rank=0),
            _stageB_adjudication_key(complex_transform, y_name="arctan", rank=1),
        )

    def test_stageB_adjudication_prefers_clean_branch_for_noise_tied_full_rewrites(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        clean_identity = SimpleNamespace(
            model=Model(1),
            val_loss=5.3413e-6,
            acceptance_noise_floor_raw=5.31e-6,
            acceptance_noise_n_eff=2000,
            loss_good_enough_eff=1.0e-7,
            loss_acceptable_eff=1.6e-5,
            enabled_patterns=["metric_distance"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 15.6},
            y_expr_str="sqrt((x0-x1)**2 + (x2-x3)**2)",
        )
        generic_sqrt = SimpleNamespace(
            model=Model(8),
            val_loss=5.1149e-6,
            acceptance_noise_floor_raw=5.31e-6,
            acceptance_noise_n_eff=2000,
            loss_good_enough_eff=1.0e-7,
            loss_acceptable_eff=1.6e-5,
            enabled_patterns=["rratpoly"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 1120.0},
            y_expr_str="sqrt(rratpoly0(x0,x1,x2,x3))",
        )

        self.assertLess(
            _stageB_adjudication_key(clean_identity, y_name="identity", rank=0),
            _stageB_adjudication_key(generic_sqrt, y_name="sqrt", rank=2),
        )

    def test_stageB_adjudication_keeps_large_loss_differences_before_generic_penalty(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        weak_clean = SimpleNamespace(
            model=Model(1),
            val_loss=1.5e-5,
            acceptance_noise_floor_raw=5.31e-6,
            acceptance_noise_n_eff=2000,
            loss_good_enough_eff=1.0e-7,
            loss_acceptable_eff=1.6e-5,
            enabled_patterns=["metric_distance"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 15.6},
            y_expr_str="sqrt((x0-x1)**2 + (x2-x3)**2)",
        )
        much_better_generic = SimpleNamespace(
            model=Model(8),
            val_loss=5.1149e-6,
            acceptance_noise_floor_raw=5.31e-6,
            acceptance_noise_n_eff=2000,
            loss_good_enough_eff=1.0e-7,
            loss_acceptable_eff=1.6e-5,
            enabled_patterns=["rratpoly"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 1120.0},
            y_expr_str="sqrt(rratpoly0(x0,x1,x2,x3))",
        )

        self.assertLess(
            _stageB_adjudication_key(much_better_generic, y_name="sqrt", rank=2),
            _stageB_adjudication_key(weak_clean, y_name="identity", rank=0),
        )

    def test_stageB_adjudication_pb030_exact_sin_beats_identity_rational(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        sin_exact = SimpleNamespace(
            model=Model(1),
            val_loss=3.2830e-33,
            loss_good_enough_eff=2.76e-10,
            loss_acceptable_eff=2.76e-6,
            enabled_patterns=["monomial_deg1[z_inv]"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        identity_rational = SimpleNamespace(
            model=Model(9),
            val_loss=7.3212e-10,
            loss_good_enough_eff=2.82e-10,
            loss_acceptable_eff=2.82e-6,
            enabled_patterns=["mono_peel_nn_resid", "ratpoly_1d[9]"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        self.assertLess(
            _stageB_adjudication_key(sin_exact, y_name="sin", rank=3, y_sources=["scatter"]),
            _stageB_adjudication_key(identity_rational, y_name="identity", rank=4, y_sources=["baseline"]),
        )

    def test_stageB_adjudication_prefers_non_generic_exact_over_exact_ratpoly(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        factorized_search_exact = SimpleNamespace(
            model=Model(4),
            val_loss=2.0e-32,
            loss_good_enough_eff=1.0e-8,
            loss_acceptable_eff=1.0e-5,
            enabled_patterns=["factorized_search_mono(1/x0 + (1-cos(x3))/z)"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        identity_ratpoly = SimpleNamespace(
            model=Model(8),
            val_loss=1.0e-10,
            loss_good_enough_eff=1.0e-8,
            loss_acceptable_eff=1.0e-5,
            enabled_patterns=["ratpoly[2]"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        self.assertLess(
            _stageB_adjudication_key(factorized_search_exact, y_name="reciprocal", rank=1, y_sources=["virtual"]),
            _stageB_adjudication_key(identity_ratpoly, y_name="identity", rank=0, y_sources=["baseline"]),
        )

    def test_stageB_adjudication_pb085_planck_beats_generic_transformed_branch(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        identity_planck = SimpleNamespace(
            model=Model(4),
            val_loss=1.0e-27,
            loss_good_enough_eff=1.0e-8,
            loss_acceptable_eff=1.0e-5,
            enabled_patterns=["planck"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        sqrt1p_ratpoly = SimpleNamespace(
            model=Model(6),
            val_loss=1.0e-30,
            loss_good_enough_eff=1.0e-8,
            loss_acceptable_eff=1.0e-5,
            enabled_patterns=["rratpoly"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        self.assertLess(
            _stageB_adjudication_key(
                identity_planck,
                y_name="identity",
                rank=4,
                y_sources=["baseline"],
            ),
            _stageB_adjudication_key(
                sqrt1p_ratpoly,
                y_name="sqrt1p",
                rank=0,
                y_sources=["virtual"],
            ),
        )

    def test_stageB_adjudication_pb001_prefers_decisively_exact_analytic_branch(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        identity_rational = SimpleNamespace(
            model=Model(34),
            val_loss=1.6932111582072403e-13,
            acceptance_noise_floor_raw=0.0,
            loss_good_enough_eff=7.5044e-11,
            loss_acceptable_eff=7.5044e-7,
            enabled_patterns=["last_ratpoly[39]"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 623.5568},
            y_expr_str=(
                "(0.060973*x0**4*x1**3 + 34.780273*x0**3 + 2.160016*x1**2)"
                "/(8.449203*x0**4*x1**2 + 87.167892*x0**4 + 0.714632)"
            ),
        )
        square_analytic = SimpleNamespace(
            model=Model(0),
            val_loss=8.501837632640743e-35,
            acceptance_noise_floor_raw=0.0,
            loss_good_enough_eff=3.5781e-12,
            loss_acceptable_eff=3.5781e-8,
            original_y_val_loss=1.4436609115104047e-33,
            original_y_loss_good_enough_eff=7.5044e-11,
            original_y_loss_acceptable_eff=7.5044e-7,
            original_y_noise_floor_raw=0.0,
            enabled_patterns=["homogeneity_peel", "exp"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 16.91},
            y_expr_str="sqrt(2)*sqrt(exp(-x1**2/x0**2)/x0**2)/(2*sqrt(pi))",
        )

        square_metrics = _stageB_candidate_metrics(square_analytic, y_name="square")
        identity_metrics = _stageB_candidate_metrics(identity_rational, y_name="identity")
        self.assertTrue(square_metrics["decisively_exact_structural"])
        self.assertFalse(identity_metrics["decisively_exact_structural"])
        self.assertLess(
            _stageB_adjudication_key(
                square_analytic,
                y_name="square",
                rank=2,
                y_sources=["virtual"],
            ),
            _stageB_adjudication_key(
                identity_rational,
                y_name="identity",
                rank=0,
                y_sources=["baseline"],
            ),
        )

    def test_stageB_adjudication_pb064_prefers_visible_identity_rational(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        identity_ratpoly = SimpleNamespace(
            model=Model(0),
            val_loss=3.6e-32,
            loss_good_enough_eff=2.5e-9,
            loss_acceptable_eff=2.5e-5,
            enabled_patterns=["ratpoly_1d", "stageB_polish:snap_symbolic_constants"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 7.49},
            phi_expr_str="(-2*x0*x1 - 3)/(x0*x1 - 3)",
        )
        arctan_proxy = SimpleNamespace(
            model=Model(4),
            val_loss=2.6e-12,
            loss_good_enough_eff=2.5e-9,
            loss_acceptable_eff=2.5e-5,
            original_y_val_loss=5.3e-11,
            original_y_loss_good_enough_eff=2.5e-9,
            original_y_loss_acceptable_eff=2.5e-5,
            enabled_patterns=["leaftr_sqrt_poly3"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": False, "complexity_score": 6.4},
            phi_expr_str=(
                "(0.8862256668398598 + 0.2821507583847325*x0*x1 "
                "- 0.0924033722843827*x0**2*x1**2 "
                "+ 0.01504672160191483*x0**3*x1**3)**2"
            ),
            y_expr_str=(
                "tan((0.01504672160191483*x0**3*x1**3 "
                "- 0.0924033722843827*x0**2*x1**2 "
                "+ 0.2821507583847325*x0*x1 + 0.8862256668398598)**2)"
            ),
        )

        identity_metrics = _stageB_candidate_metrics(identity_ratpoly)
        arctan_metrics = _stageB_candidate_metrics(arctan_proxy)
        self.assertFalse(identity_metrics["generic_approximant"])
        self.assertGreater(arctan_metrics["complexity_score"], identity_metrics["complexity_score"])
        self.assertLess(
            _stageB_adjudication_key(
                identity_ratpoly,
                y_name="identity",
                rank=0,
                y_sources=["baseline"],
            ),
            _stageB_adjudication_key(
                arctan_proxy,
                y_name="arctan",
                rank=1,
                y_sources=["virtual"],
            ),
        )

    def test_stageB_metrics_use_original_y_guard_for_transformed_branch(self):
        class Model:
            def num_parameters(self):
                return 6

        transformed = SimpleNamespace(
            model=Model(),
            val_loss=1.0e-30,
            loss_good_enough_eff=1.0e-8,
            loss_acceptable_eff=1.0e-5,
            original_y_val_loss=1.0e-3,
            original_y_loss_good_enough_eff=1.0e-8,
            original_y_loss_acceptable_eff=1.0e-5,
            enabled_patterns=["monomial_deg1"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        metrics = _stageB_candidate_metrics(transformed)
        self.assertTrue(metrics["original_y_bad_loss"])
        self.assertTrue(metrics["bad_loss"])
        self.assertFalse(metrics["exact_loss"])

    def test_stageB_metrics_noisy_original_y_acceptance_rescues_proxy_bad_loss(self):
        class Model:
            def num_parameters(self):
                return 6

        transformed = SimpleNamespace(
            model=Model(),
            val_loss=1.0e-3,
            acceptance_noise_floor_raw=1.0e-6,
            loss_good_enough_eff=1.0e-8,
            loss_acceptable_eff=1.0e-5,
            original_y_val_loss=5.0e-6,
            original_y_loss_good_enough_eff=1.0e-8,
            original_y_loss_acceptable_eff=1.0e-5,
            enabled_patterns=["metric_distance"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        metrics = _stageB_candidate_metrics(transformed)
        self.assertTrue(metrics["has_original_y_validation"])
        self.assertFalse(metrics["original_y_bad_loss"])
        self.assertFalse(metrics["bad_loss"])
        self.assertAlmostEqual(metrics["loss_acceptable_ratio"], 4.0 / 9.0)

    def test_stageB_metrics_noiseless_proxy_bad_loss_is_not_rescued(self):
        class Model:
            def num_parameters(self):
                return 6

        transformed = SimpleNamespace(
            model=Model(),
            val_loss=1.0e-3,
            acceptance_noise_floor_raw=0.0,
            loss_good_enough_eff=1.0e-8,
            loss_acceptable_eff=1.0e-5,
            original_y_val_loss=5.0e-6,
            original_y_loss_good_enough_eff=1.0e-8,
            original_y_loss_acceptable_eff=1.0e-5,
            enabled_patterns=["metric_distance"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        metrics = _stageB_candidate_metrics(transformed)
        self.assertTrue(metrics["has_original_y_validation"])
        self.assertFalse(metrics["original_y_bad_loss"])
        self.assertTrue(metrics["bad_loss"])

    def test_stageB_decisive_exactness_is_measured_in_original_y_space(self):
        class Model:
            def num_parameters(self):
                return 2

        transformed = SimpleNamespace(
            model=Model(),
            val_loss=1.0e-30,
            acceptance_noise_floor_raw=0.0,
            loss_good_enough_eff=1.0e-8,
            loss_acceptable_eff=1.0e-5,
            original_y_val_loss=1.0e-9,
            original_y_loss_good_enough_eff=1.0e-8,
            original_y_loss_acceptable_eff=1.0e-5,
            original_y_noise_floor_raw=0.0,
            enabled_patterns=["exp"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )

        metrics = _stageB_candidate_metrics(transformed, y_name="square")

        self.assertTrue(metrics["exact_loss"])
        self.assertAlmostEqual(metrics["decisive_exact_ratio"], 0.1)
        self.assertFalse(metrics["decisively_exact_structural"])

    def test_stageB_adjudication_prefers_original_y_valid_clean_branch_over_proxy_loss(self):
        class Model:
            def __init__(self, n):
                self.n = n

            def num_parameters(self):
                return self.n

        clean_identity = SimpleNamespace(
            model=Model(0),
            val_loss=5.6093e-4,
            acceptance_noise_floor_raw=6.82e-5,
            acceptance_noise_n_eff=2000,
            loss_good_enough_eff=1.0e-8,
            loss_acceptable_eff=2.1e-4,
            original_y_val_loss=5.6093e-4,
            original_y_loss_good_enough_eff=5.6223e-8,
            original_y_loss_acceptable_eff=5.6223e-4,
            enabled_patterns=["sqrt_poly"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 15.6},
            y_expr_str="sqrt((x0 - x1)**2 + (x2 - x3)**2)",
        )
        generic_sqrt = SimpleNamespace(
            model=Model(8),
            val_loss=3.6293e-5,
            acceptance_noise_floor_raw=6.82e-5,
            acceptance_noise_n_eff=2000,
            loss_good_enough_eff=1.0e-8,
            loss_acceptable_eff=2.1e-4,
            original_y_val_loss=8.8706e-4,
            original_y_loss_good_enough_eff=5.6223e-8,
            original_y_loss_acceptable_eff=5.6223e-4,
            enabled_patterns=["last_ratpoly_1d"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": False, "complexity_score": 500.0},
            y_expr_str="sqrt(scale0() * rratpoly0((x0-x1)**2 + (x2-x3)**2))",
        )

        clean_metrics = _stageB_candidate_metrics(clean_identity)
        generic_metrics = _stageB_candidate_metrics(generic_sqrt)
        self.assertFalse(clean_metrics["bad_loss"])
        self.assertTrue(generic_metrics["bad_loss"])
        self.assertLess(
            _stageB_adjudication_key(clean_identity, y_name="identity", rank=0),
            _stageB_adjudication_key(generic_sqrt, y_name="sqrt", rank=2),
        )

    def test_final_polish_keeps_exact_stageB_seed_over_worse_recommendation(self):
        seed = SimpleNamespace(
            label="seed",
            val_mse=1.0e-30,
            val_mse_se=0.0,
            is_recommended=False,
        )
        worse = SimpleNamespace(
            label="snap_symbolic_constants",
            val_mse=1.0e-8,
            val_mse_se=0.0,
            is_recommended=True,
        )
        result = SimpleNamespace(
            recommended=worse,
            all_candidates=[seed, worse],
            warnings=[],
        )
        protected, reason = _protect_exact_stageB_seed_in_final_polish(
            result,
            SimpleNamespace(epsilon_pareto_k=1.0, loss_equiv_abs_floor=0.0),
            {
                "candidate_metrics": {
                    "exact_loss": True,
                    "full_rewrite": True,
                    "generic_approximant": False,
                    "accepted_patterns": 1,
                }
            },
        )
        self.assertIs(protected.recommended, seed)
        self.assertTrue(seed.is_recommended)
        self.assertFalse(worse.is_recommended)
        self.assertIn("exact non-generic", reason)

    def test_stageB_generic_approximant_detection_from_pattern_label(self):
        is_generic, reason = _stageB_generic_approximant_signature(
            None,
            accepted_labels=("ratpoly[2]",),
        )
        self.assertTrue(is_generic)
        self.assertIn("ratpoly[2]", reason)

    def test_stageB_generic_label_is_provenance_after_visible_simplification(self):
        is_generic, reason = _stageB_generic_approximant_signature(
            None,
            accepted_labels=("ratpoly_1d",),
            explicit_simplified_expr=True,
        )
        self.assertFalse(is_generic)
        self.assertEqual(reason, "")

        is_generic, reason = _stageB_generic_approximant_signature(
            None,
            accepted_labels=("exp_rat",),
            explicit_simplified_expr=True,
        )
        self.assertTrue(is_generic)
        self.assertIn("exp_rat", reason)

    def test_stageB_generic_ratpoly_atom_is_provenance_after_small_integer_simplification(self):
        root = AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={"deg_num": 1, "deg_den": 1})
        is_generic, reason = _stageB_generic_approximant_signature(
            root,
            accepted_labels=("ratpoly[1]",),
            explicit_simplified_expr=True,
            simple_integer_rational_expr=True,
        )
        self.assertFalse(is_generic)
        self.assertEqual(reason, "")

    def test_stageB_generic_ratpoly_atom_not_upgraded_before_simplification(self):
        root = AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={"deg_num": 1, "deg_den": 1})
        is_generic, reason = _stageB_generic_approximant_signature(
            root,
            accepted_labels=("ratpoly[1]",),
            explicit_simplified_expr=False,
            simple_integer_rational_expr=True,
        )
        self.assertTrue(is_generic)
        self.assertIn("ratpoly[1]", reason)

    def test_stageB_generic_approximant_detection_from_ast_atom(self):
        root = AtomNode(kind="rratpoly", var_idxs=(0, 1, 2), kwargs={"deg_num": 5, "deg_den": 4})
        is_generic, reason = _stageB_generic_approximant_signature(root, accepted_labels=())
        self.assertTrue(is_generic)
        self.assertIn("rratpoly/arity3", reason)

    def test_stageB_metrics_upgrade_small_integer_rational_only_after_stagec_accepts(self):
        class Model:
            def num_parameters(self):
                return 2

        root = AtomNode(kind="rratpoly", var_idxs=(0,), kwargs={"deg_num": 1, "deg_den": 1})
        accepted = SimpleNamespace(
            root=root,
            model=Model(),
            val_loss=1.0e-30,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["ratpoly[1]"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": True, "complexity_score": 7.0},
            phi_expr_str="(-2*x0 - 3)/(x0 - 3)",
        )
        pending = SimpleNamespace(
            root=root,
            model=Model(),
            val_loss=1.0e-30,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["ratpoly[1]"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
            sympy_meta={"accepted": False, "complexity_score": 7.0},
            phi_expr_raw_str="(-2*x0 - 3)/(x0 - 3)",
        )

        accepted_metrics = _stageB_candidate_metrics(accepted)
        pending_metrics = _stageB_candidate_metrics(pending)
        self.assertTrue(accepted_metrics["simple_integer_rational_expr"])
        self.assertFalse(accepted_metrics["generic_approximant"])
        self.assertFalse(pending_metrics["simple_integer_rational_expr"])
        self.assertTrue(pending_metrics["generic_approximant"])

    def test_stageB_portfolio_early_stop_requires_branch_safe_validation_good_full_rewrite(self):
        class Model:
            def num_parameters(self):
                return 1

        exact = SimpleNamespace(
            model=Model(),
            val_loss=1.0e-30,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["monomial_deg1"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        approximate = SimpleNamespace(
            model=Model(),
            val_loss=1.0e-5,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["ratpoly_1d"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        exact_generic_ratpoly = SimpleNamespace(
            model=Model(),
            val_loss=1.0e-30,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["rratpoly"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        self.assertTrue(
            _stageB_portfolio_can_stop_early(
                exact,
                y_sources=["outer_affine_confirmed"],
            )
        )
        self.assertTrue(
            _stageB_portfolio_can_stop_early(
                exact,
                y_sources=["baseline"],
            )
        )
        self.assertFalse(
            _stageB_portfolio_can_stop_early(
                exact,
                y_sources=["virtual"],
            )
        )
        self.assertFalse(
            _stageB_portfolio_can_stop_early(
                approximate,
                y_sources=["outer_affine_confirmed"],
            )
        )
        self.assertFalse(
            _stageB_portfolio_can_stop_early(
                exact_generic_ratpoly,
                y_sources=["baseline"],
            )
        )

    def test_stageB_shadow_rescue_reason_flags_generic_approximant(self):
        class Model:
            def num_parameters(self):
                return 8

        generic = SimpleNamespace(
            model=Model(),
            val_loss=1.0e-30,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["sqrt_ratpoly"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        clean = SimpleNamespace(
            model=Model(),
            val_loss=1.0e-30,
            loss_good_enough_eff=1.0e-9,
            loss_acceptable_eff=1.0e-6,
            enabled_patterns=["planck_compound_prefactor"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        self.assertIn("generic approximant", _stageB_shadow_rescue_reason(generic))
        self.assertEqual(_stageB_shadow_rescue_reason(clean), "")

    def test_stageB_fully_analytic_polish_failure_keeps_accepted_state(self):
        import torch

        from nestynet_sr.sr_search.stageB.engine import StageBContext, StageBState
        import nestynet_sr.sr_search.stageB.polish as polish

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)

            def num_parameters(self):
                return 0

        state = StageBState(
            root=AtomNode(kind="planck", var_idxs=(0,), tag="leaf0"),
            model=Model(),
            reuse={},
            val_loss=1.0e-30,
            enabled_patterns=["planck"],
            num_nn_atoms=0,
            num_multivar_nn_atoms=0,
            max_nn_arity=0,
        )
        ctx = StageBContext(
            state=state,
            train_loader=[],
            val_loader=[],
            lm_hp=SimpleNamespace(
                stageB_polish=True,
                stageB_polish_commit=True,
                stageB_polish_max_candidates=1,
                loss_acceptable=1.0e-6,
                select_stageB_max_decades_over_floor=1.0,
            ),
            device=torch.device("cpu"),
            dtype=torch.float64,
            epochs_stageB=1,
            loss_scale=1.0,
            loss_good_enough_raw=1.0e-9,
            score_tol=0.0,
            scale_specs=[],
            scaling_by_axis={},
            trig_by_axis={},
            verbose=False,
        )

        original = polish.build_fully_analytic_polish_candidate

        def raise_index_error(*_args, **_kwargs):
            raise IndexError("index 0 is out of range")

        polish.build_fully_analytic_polish_candidate = raise_index_error
        try:
            self.assertFalse(ctx.maybe_polish_after_accept())
            self.assertIs(ctx.state, state)
            self.assertEqual(ctx.state.num_nn_atoms, 0)
            self.assertEqual(ctx.state.enabled_patterns, ["planck"])
        finally:
            polish.build_fully_analytic_polish_candidate = original

    def test_stageA_initial_random_restart_is_identity_baseline_only(self):
        base = dict(
            y_op_is_identity=True,
            is_multi=False,
            skip_initial_fit=False,
            restart_used=False,
            has_previous_model=False,
            fit_y_link_active=False,
        )
        self.assertTrue(_stageA_initial_fit_restart_allowed(**base))

        for key in (
            "y_op_is_identity",
            "is_multi",
            "skip_initial_fit",
            "restart_used",
            "has_previous_model",
            "fit_y_link_active",
        ):
            case = dict(base)
            case[key] = not case[key]
            self.assertFalse(_stageA_initial_fit_restart_allowed(**case), key)


if __name__ == "__main__":
    unittest.main()
