import numpy as np
import pytest
import torch

from nestynet_sr.sr_core.bridges import AtomNode, MulNode, PowNode, Var
from nestynet_sr.sr_gs import GeneralizedSymmetryConfig, suppress_shadowed_stagea_proposals
from nestynet_sr.sr_gs.reporting import build_gs_payload, format_gs_markdown, reset_gs_reporter
from nestynet_sr.sr_search.search import _detect_compound_variable_for_atom


def _monomial_ast():
    return MulNode(PowNode(Var(0), 2.0), PowNode(Var(1), -1.0))


def _promoted_gs_monomial_proposal():
    """A promoted GS proposal shaped like the stagea_bridge output."""

    return (
        (1, 1),
        _monomial_ast(),
        1.0,
        None,
        {
            "kind": "gs_promoted_reduction",
            "source": "generalized_symmetry",
            "gs_source_family": "general_affine",
            "gs_promotion_state": "promoted",
            "gs_coordinate_kind": "monomial",
            "gs_chart": "log",
            "gs_monomial_exponents": (2, -1),
            "gs_monomial_exponents_key": (2, -1),
            "gs_confidence": 1.0,
            "z_human": "((x0)**2 * (x1)**-1)",
        },
    )


def _replace_cfg(policy="replace-shadowed"):
    return GeneralizedSymmetryConfig(
        enabled=True,
        mode="propose",
        policy=policy,
        known_generators=False,
        known_lie=False,
        general_affine=True,
        residual_tol=1.0e-8,
        general_affine_promotion_residual_tol=1.0e-8,
        general_affine_charts=("identity", "log"),
    )


# ---------------------------------------------------------------------------
# Helper-level contracts
# ---------------------------------------------------------------------------


def test_only_bare_matching_monomial_is_suppressed():
    gs = _promoted_gs_monomial_proposal()
    bare_monomial_3tuple = ((2, -1), _monomial_ast(), 0.9)
    monomial_with_extra_override = ((2, -1), _monomial_ast(), 0.9, [0])
    monomial_with_prefactor = (
        (2, -1),
        _monomial_ast(),
        0.9,
        None,
        {"kind": "monomial", "prefactor_exponents": (1, 0)},
    )
    monomial_other_ray = ((1, -2), _monomial_ast(), 0.7, None, {"kind": "monomial"})
    linear_unmatched = ((1, -1), _monomial_ast(), 0.8, None, {"kind": "linear"})
    radial_mask = ((1, 1), _monomial_ast(), 0.8, None, {"kind": "radial"})
    legacy = [
        bare_monomial_3tuple,
        monomial_with_extra_override,
        monomial_with_prefactor,
        monomial_other_ray,
        linear_unmatched,
        radial_mask,
    ]

    reset_gs_reporter()
    filtered, events = suppress_shadowed_stagea_proposals(
        legacy, [gs], cols=(0, 1), cfg=_replace_cfg()
    )

    assert len(events) == 1
    assert events[0]["legacy_kind"] == "monomial"
    assert events[0]["legacy_pattern"] == [2, -1]
    assert events[0]["gs_chart"] == "log"
    assert bare_monomial_3tuple not in filtered
    assert monomial_with_extra_override in filtered
    assert monomial_with_prefactor in filtered
    assert monomial_other_ray in filtered
    assert linear_unmatched in filtered
    assert radial_mask in filtered

    payload = build_gs_payload()
    assert len(payload["policy_events"]) == 1
    event = payload["policy_events"][0]
    assert event["action"] == "stagea_replace_shadowed_suppression"
    assert event["policy"] == "replace-shadowed"
    markdown = format_gs_markdown(payload)
    assert "stagea_replace_shadowed_suppression" in markdown


def test_projective_key_matching_handles_sign_and_scale():
    gs = _promoted_gs_monomial_proposal()
    # (-2, 1) and (4, -2) describe the same monomial ray as (2, -1).
    flipped = ((-2, 1), _monomial_ast(), 0.9)
    doubled = ((4, -2), _monomial_ast(), 0.9)

    filtered, events = suppress_shadowed_stagea_proposals(
        [flipped, doubled], [gs], cols=(0, 1), cfg=_replace_cfg()
    )
    assert filtered == []
    assert len(events) == 2


def test_support_mismatch_never_suppresses():
    gs = _promoted_gs_monomial_proposal()
    wider_support = ((2, -1, 1), MulNode(_monomial_ast(), Var(2)), 0.9)

    filtered, events = suppress_shadowed_stagea_proposals(
        [wider_support], [gs], cols=(0, 1, 2), cfg=_replace_cfg()
    )
    assert filtered == [wider_support]
    assert events == []


def test_linear_suppression_requires_covector_direction_match():
    linear_gs = (
        (1, 1),
        _monomial_ast(),
        1.0,
        None,
        {
            "kind": "gs_promoted_reduction",
            "source": "generalized_symmetry",
            "gs_promotion_state": "promoted",
            "gs_coordinate_kind": "linear_projection",
            "gs_chart": "identity",
            "gs_linear_covector": (0.7071067811865476, -0.7071067811865476),
            "gs_confidence": 1.0,
            "z_human": "(x0 - x1)",
        },
    )
    matching_linear = ((1, -1), _monomial_ast(), 0.8, None, {"kind": "linear"})
    scaled_matching_linear = ((2, -2), _monomial_ast(), 0.8, None, {"kind": "linear"})
    non_matching_linear = ((1, 1), _monomial_ast(), 0.8, None, {"kind": "linear"})
    no_covector_gs = (
        (1, 1),
        _monomial_ast(),
        1.0,
        None,
        {
            "kind": "gs_promoted_reduction",
            "gs_promotion_state": "promoted",
            "gs_coordinate_kind": "linear_projection",
            "gs_confidence": 1.0,
        },
    )

    filtered, events = suppress_shadowed_stagea_proposals(
        [matching_linear, scaled_matching_linear, non_matching_linear],
        [linear_gs],
        cols=(0, 1),
        cfg=_replace_cfg(),
    )
    assert matching_linear not in filtered
    assert scaled_matching_linear not in filtered
    assert non_matching_linear in filtered
    assert len(events) == 2

    # Covector absent: never suppress.
    filtered2, events2 = suppress_shadowed_stagea_proposals(
        [matching_linear], [no_covector_gs], cols=(0, 1), cfg=_replace_cfg()
    )
    assert filtered2 == [matching_linear]
    assert events2 == []


def test_non_promoted_gs_proposals_never_suppress():
    audit_gs = (
        (1, 1),
        _monomial_ast(),
        0.9,
        None,
        {
            "kind": "gs_promoted_reduction",
            "gs_promotion_state": "audit",
            "gs_monomial_exponents_key": (2, -1),
        },
    )
    named_gs = (
        (1, 1),
        _monomial_ast(),
        0.9,
        None,
        {"kind": "gs_scaling", "source": "generalized_symmetry"},
    )
    bare = ((2, -1), _monomial_ast(), 0.9)

    filtered, events = suppress_shadowed_stagea_proposals(
        [bare], [audit_gs, named_gs], cols=(0, 1), cfg=_replace_cfg()
    )
    assert filtered == [bare]
    assert events == []


def test_no_promoted_gs_is_strict_passthrough():
    bare = ((2, -1), _monomial_ast(), 0.9)
    filtered, events = suppress_shadowed_stagea_proposals(
        [bare], [], cols=(0, 1), cfg=_replace_cfg()
    )
    assert filtered == [bare]
    assert events == []


# ---------------------------------------------------------------------------
# Merge-level contracts through _detect_compound_variable_for_atom
# ---------------------------------------------------------------------------


class MonomialLeaf(torch.nn.Module):
    """g(x0**2/x1) with generic g and analytic input gradients."""

    def forward(self, x):
        z = x[:, 0:1] ** 2 / x[:, 1:2]
        return torch.sin(z) + 0.17 * z**3

    def grad(self, cache):
        x = cache["x"]
        z = x[:, 0:1] ** 2 / x[:, 1:2]
        gz = torch.cos(z) + 0.51 * z**2
        d0 = gz * 2.0 * x[:, 0:1] / x[:, 1:2]
        d1 = -gz * x[:, 0:1] ** 2 / x[:, 1:2] ** 2
        return torch.cat([d0, d1], dim=1).unsqueeze(1)


def _run_detector(gs_cfg):
    rng = np.random.default_rng(2469)
    x_raw = torch.tensor(rng.uniform(0.5, 2.0, size=(256, 2)), dtype=torch.float64)
    y_dummy = torch.zeros((x_raw.shape[0], 1), dtype=torch.float64)
    atom = AtomNode(
        kind="nn",
        var_idxs=(0, 1),
        kwargs={"num_segments": 8, "dual_layer": False},
        tag="nn_gs_monomial",
    )
    proposals, _ = _detect_compound_variable_for_atom(
        model=object(),
        atom=atom,
        leaf=MonomialLeaf(),
        datagen_train=[(x_raw, y_dummy)],
        device=torch.device("cpu"),
        max_exponent=2,
        precision=0.05,
        max_batches=1,
        enable_linear=False,
        enable_radial=False,
        enable_shift=False,
        enable_mixed_compound=False,
        trig_axis_specs=None,
        scaling_features=None,
        gs_cfg=gs_cfg,
    )
    return proposals


def _bare_legacy_monomials(proposals):
    """Legacy bare-monomial proposals for the (2, -1) ray on support {0, 1}."""

    from nestynet_sr.sr_gs.unit_torus import projective_exponent_key

    out = []
    for p in proposals:
        meta = p[4] if len(p) >= 5 and isinstance(p[4], dict) else {}
        if str(meta.get("source", "")) == "generalized_symmetry":
            continue
        if str(meta.get("kind", "monomial")) != "monomial":
            continue
        extra = p[3] if len(p) >= 4 else None
        if extra:
            continue
        if any(meta.get(k) for k in ("prefactor_exponents", "retained_axis_wrapper", "compound_subset")):
            continue
        pattern = p[0]
        if not isinstance(pattern, (tuple, list)) or len(pattern) != 2:
            continue
        support = tuple(i for i, v in enumerate(pattern) if float(v) != 0.0)
        if support != (0, 1):
            continue
        if tuple(projective_exponent_key(pattern)) == (2, -1):
            out.append(p)
    return out


def _gs_promoted(proposals):
    return [
        p
        for p in proposals
        if len(p) >= 5 and isinstance(p[4], dict) and p[4].get("kind") == "gs_promoted_reduction"
    ]


def test_merge_level_replace_shadowed_preserves_ordinary_stagea_lane():
    reset_gs_reporter()
    proposals = _run_detector(_replace_cfg("replace-shadowed"))
    assert _gs_promoted(proposals), "promoted GS reduction expected in the proposal stream"
    assert _bare_legacy_monomials(proposals), "GS must not displace an ordinary proposal"
    payload = build_gs_payload()
    assert payload["policy_events"] == []


def test_merge_level_augment_keeps_legacy_monomial():
    reset_gs_reporter()
    proposals = _run_detector(_replace_cfg("augment"))
    assert _bare_legacy_monomials(proposals), "legacy monomial must survive under augment"
    payload = build_gs_payload()
    assert payload["policy_events"] == []


def test_merge_level_gs_off_baseline_unchanged():
    """gs_cfg=None and enabled=False must produce identical proposal streams."""

    def summary(proposals):
        out = []
        for p in proposals:
            meta = p[4] if len(p) >= 5 and isinstance(p[4], dict) else {}
            out.append((tuple(p[0]), repr(p[1]), round(float(p[2]), 12), str(meta.get("kind", ""))))
        return out

    none_props = _run_detector(None)
    disabled_props = _run_detector(GeneralizedSymmetryConfig(enabled=False))
    assert summary(none_props) == summary(disabled_props)
    assert not _gs_promoted(none_props)
    assert _bare_legacy_monomials(none_props), "baseline monomial detector must fire"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
