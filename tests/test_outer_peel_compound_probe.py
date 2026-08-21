# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata

import torch

from nestynet_sr.sr_search.outer_peel import probe_affine_outer_peels_on_z


def test_compound_z_probe_prefers_quadratic_when_needed():
    """sin(y) should recover a quadratic-in-z inner exactly."""
    torch.manual_seed(0)
    z = torch.linspace(-0.8, 0.8, 1200, dtype=torch.float64)
    inner = 0.35 * z * z + 0.55 * z - 0.05
    y = torch.asin(inner)

    ranked = probe_affine_outer_peels_on_z(
        y=y,
        z=z,
        transform_names=("sin", "cos"),
        min_points=200,
        min_domain_frac=0.90,
    )

    assert ranked, "Expected at least one probe result"
    best = ranked[0]
    assert best.name == "sin", f"Expected sin peel, got {best.name}"
    assert best.rms_rel < 1.0e-8, f"Expected near-perfect fit, got {best.rms_rel:.3g}"
    assert best.details.get("fit_kind") == "quadratic"


def test_compound_z_probe_still_solves_affine_case():
    """Quadratic fallback must not break affine compound-z links."""
    z = torch.linspace(-0.9, 0.9, 1000, dtype=torch.float64)
    inner = 0.72 * z + 0.03
    y = torch.asin(inner)

    ranked = probe_affine_outer_peels_on_z(
        y=y,
        z=z,
        transform_names=("sin",),
        min_points=200,
        min_domain_frac=0.90,
    )

    assert ranked, "Expected at least one probe result"
    best = ranked[0]
    assert best.name == "sin"
    assert best.rms_rel < 1.0e-8
    assert abs(float(best.details.get("q2", 0.0))) < 1.0e-8
