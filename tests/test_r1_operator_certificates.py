# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2023-2026 Rodrigo Ibata
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

import math

import torch

from nestynet_sr.sr_core.bridges import AsinNode, AtomNode, PowNode, Var
from nestynet_sr.sr_search.r1_operator_certificates import (
    build_r1_certificate_replacement,
    scan_r1_operator_certificates,
)


def _cert_by_label(certs, label):
    for cert in certs:
        if cert.label == label:
            return cert
    return None


def test_square_link_detects_sqrt_one_refine_two_z():
    z = torch.linspace(0.0, 20.0, 2000, dtype=torch.float64)
    y = torch.sqrt(1.0 + 2.0 * z)

    certs = scan_r1_operator_certificates(z, y, rel_rms_max=1.0e-10)
    cert = _cert_by_label(certs, "r1_square_sqrt")

    assert cert is not None
    assert cert.inverse_kind == "sqrt"
    assert abs(cert.affine_a - 2.0) < 1.0e-10
    assert abs(cert.affine_b - 1.0) < 1.0e-10


def test_square_link_also_tests_reciprocal_coordinate():
    z = torch.linspace(1.0, 20.0, 2000, dtype=torch.float64)
    y = torch.sqrt(1.0 + 2.0 / z)

    certs = scan_r1_operator_certificates(z, y, max_results=20, rel_rms_max=1.0e-10)
    cert = _cert_by_label(certs, "r1_square_sqrt_zinv")

    assert cert is not None
    assert cert.inverse_kind == "sqrt"
    assert abs(cert.psi_power + 1.0) < 1.0e-12
    assert abs(cert.affine_a - 2.0) < 1.0e-10
    assert abs(cert.affine_b - 1.0) < 1.0e-10


def test_inverse_trig_certificate_requires_principal_asin_branch():
    z = torch.linspace(-0.8, 0.8, 2000, dtype=torch.float64)
    principal = torch.asin(0.75 * z)
    off_branch = math.pi - principal

    certs_principal = scan_r1_operator_certificates(z, principal, rel_rms_max=1.0e-10)
    certs_off = scan_r1_operator_certificates(z, off_branch, rel_rms_max=1.0e-10)

    assert _cert_by_label(certs_principal, "r1_outer_asin") is not None
    assert _cert_by_label(certs_off, "r1_outer_asin") is None


def test_build_r1_square_certificate_replaces_nn_with_visible_sqrt_poly():
    z = torch.linspace(0.0, 10.0, 512, dtype=torch.float64)
    y = torch.sqrt(1.0 + 2.0 * z)
    cert = _cert_by_label(scan_r1_operator_certificates(z, y, rel_rms_max=1.0e-10), "r1_square_sqrt")
    assert cert is not None

    target = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0", inputs=(Var(0),))
    root, arg_tag = build_r1_certificate_replacement(
        target,
        target,
        Var(0),
        cert,
        tag_prefix="leaf0",
    )

    assert arg_tag == "leaf0_r1_square_sqrt_arg"
    assert isinstance(root, PowNode)
    assert abs(float(root.exponent) - 0.5) < 1.0e-12
    assert isinstance(root.base, AtomNode)
    assert root.base.kind == "poly"


def test_build_r1_asin_certificate_replaces_nn_with_visible_asin_poly():
    z = torch.linspace(-0.8, 0.8, 512, dtype=torch.float64)
    y = torch.asin(0.5 * z)
    cert = _cert_by_label(scan_r1_operator_certificates(z, y, rel_rms_max=1.0e-10), "r1_outer_asin")
    assert cert is not None

    target = AtomNode(kind="nn", var_idxs=(0,), tag="leaf0", inputs=(Var(0),))
    root, arg_tag = build_r1_certificate_replacement(
        target,
        target,
        Var(0),
        cert,
        tag_prefix="leaf0",
    )

    assert arg_tag == "leaf0_r1_outer_asin_arg"
    assert isinstance(root, AsinNode)
    assert isinstance(root.arg, AtomNode)
    assert root.arg.kind == "poly"
