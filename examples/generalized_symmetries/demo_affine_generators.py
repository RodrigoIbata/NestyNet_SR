#!/usr/bin/env python3
"""Tiny audit of the GS affine-generator layer on analytic toy functions."""

from __future__ import annotations

import json
import numpy as np

from nestynet_sr.sr_gs import GeneralizedSymmetryConfig, discover_generator_specs, summarize_specs


def run_case(name, X, y, G):
    cfg = GeneralizedSymmetryConfig(
        enabled=True,
        translations=True,
        diagonal_translations=True,
        scalings=True,
        rotations=True,
        lorentz_boosts=True,
        min_confidence=0.5,
        mode="auto",
        report_rejected=True,
    )
    specs = discover_generator_specs(X, y, G, cols=tuple(range(X.shape[1])), cfg=cfg, include_rejected=True)
    print(f"\n{name}")
    print(json.dumps(summarize_specs(specs[:6]), indent=2))


def main():
    rng = np.random.default_rng(123)

    X = rng.normal(size=(512, 2))
    y = X[:, 0] ** 2 + X[:, 1] ** 2
    G = np.stack([2 * X[:, 0], 2 * X[:, 1]], axis=1)
    run_case("radial y=x0^2+x1^2; expect SO(2) rotation invariant", X, y, G)

    X = rng.uniform(0.5, 2.0, size=(512, 2))
    z = X[:, 0] / X[:, 1]
    y = np.sin(z)
    G = np.stack([np.cos(z) / X[:, 1], -X[:, 0] * np.cos(z) / (X[:, 1] ** 2)], axis=1)
    run_case("ratio y=sin(x0/x1); expect common scaling invariant", X, y, G)

    X = rng.normal(size=(512, 2))
    z = X[:, 0] - X[:, 1]
    y = z ** 3
    G = np.stack([3 * z ** 2, -3 * z ** 2], axis=1)
    run_case("difference y=(x0-x1)^3; expect diagonal translation invariant", X, y, G)

    X = rng.uniform(-2.0, 2.0, size=(512, 2))
    # Avoid a tiny band where the gradient of u^2-x^2 is too small.
    X = X[np.abs(X[:, 0]) + np.abs(X[:, 1]) > 0.2]
    y = X[:, 0] ** 2 - X[:, 1] ** 2
    G = np.stack([2 * X[:, 0], -2 * X[:, 1]], axis=1)
    run_case("Lorentz interval y=x0^2-x1^2; expect boost invariant", X, y, G)


if __name__ == "__main__":
    main()
