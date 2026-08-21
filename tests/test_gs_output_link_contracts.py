import numpy as np


def _log_link_fixture(n=160):
    rng = np.random.default_rng(301)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    f = np.exp(X[:, 0] ** 2 + np.sin(X[:, 1]))
    f_x = f * (2.0 * X[:, 0])
    f_y = f * np.cos(X[:, 1])
    f_xx = f * (4.0 * X[:, 0] ** 2 + 2.0)
    f_yy = f * (np.cos(X[:, 1]) ** 2 - np.sin(X[:, 1]))
    f_xy = f * (2.0 * X[:, 0]) * np.cos(X[:, 1])
    grad = np.stack([f_x, f_y], axis=1)
    hess = np.zeros((len(X), 2, 2))
    hess[:, 0, 0] = f_xx
    hess[:, 1, 1] = f_yy
    hess[:, 0, 1] = f_xy
    hess[:, 1, 0] = f_xy
    return X, f, grad, hess


def _additive_link_fixture(n=128):
    rng = np.random.default_rng(302)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    f = X[:, 0] ** 2 + np.sin(X[:, 1])
    f_x = 2.0 * X[:, 0]
    f_y = np.cos(X[:, 1])
    f_xx = np.full(len(X), 2.0)
    f_yy = -np.sin(X[:, 1])
    grad = np.stack([f_x, f_y], axis=1)
    hess = np.zeros((len(X), 2, 2))
    hess[:, 0, 0] = f_xx
    hess[:, 1, 1] = f_yy
    return X, f, grad, hess


def _power_link_fixture(power=2.0, n=144):
    rng = np.random.default_rng(303)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    u = 2.5 + X[:, 0] ** 2 + 0.3 * np.sin(X[:, 1])
    u_x = 2.0 * X[:, 0]
    u_y = 0.3 * np.cos(X[:, 1])
    u_xx = np.full(len(X), 2.0)
    u_yy = -0.3 * np.sin(X[:, 1])
    q = 1.0 / float(power)
    f = u**q
    f_x = q * u ** (q - 1.0) * u_x
    f_y = q * u ** (q - 1.0) * u_y
    f_xx = q * (q - 1.0) * u ** (q - 2.0) * u_x**2 + q * u ** (q - 1.0) * u_xx
    f_yy = q * (q - 1.0) * u ** (q - 2.0) * u_y**2 + q * u ** (q - 1.0) * u_yy
    f_xy = q * (q - 1.0) * u ** (q - 2.0) * u_x * u_y
    grad = np.stack([f_x, f_y], axis=1)
    hess = np.zeros((len(X), 2, 2))
    hess[:, 0, 0] = f_xx
    hess[:, 1, 1] = f_yy
    hess[:, 0, 1] = f_xy
    hess[:, 1, 0] = f_xy
    return X, f, grad, hess


def test_output_link_witness_fits_log_link_by_undivided_residual():
    from nestynet_sr.sr_gs.output_link import discover_output_link_separability

    X, y, grad, hess = _log_link_fixture()
    witness = discover_output_link_separability(X, y, grad, hess, blocks=((0,), (1,)))

    assert witness.accepted
    assert witness.link_family == "log"
    assert witness.uses_implicit_residual
    assert witness.max_cross_pair_residual < 1.0e-8
    assert witness.psi_prime_nonzero
    assert witness.gauge == "psi(1)=0, psi_prime(1)=1"


def test_output_link_witness_handles_stationary_directions_without_ratio_blowup():
    from nestynet_sr.sr_gs.output_link import discover_output_link_separability

    X, y, grad, hess = _log_link_fixture()
    X[0] = 0.0
    grad[0] = 0.0

    witness = discover_output_link_separability(X, y, grad, hess, blocks=((0,), (1,)))

    assert witness.accepted
    assert np.isfinite(witness.max_cross_pair_residual)
    assert not witness.computed_raw_ratio


def test_output_link_witness_recognizes_additive_as_zero_curvature_link():
    from nestynet_sr.sr_gs.output_link import discover_output_link_separability

    X, y, grad, hess = _additive_link_fixture()
    witness = discover_output_link_separability(X, y, grad, hess, blocks=((0,), (1,)))

    assert witness.accepted
    assert witness.link_family == "additive"
    assert witness.max_cross_pair_residual < 1.0e-10
    assert witness.uses_implicit_residual


def test_output_link_witness_recognizes_power_and_reciprocal_links():
    from nestynet_sr.sr_gs.output_link import discover_output_link_separability

    X, y, grad, hess = _power_link_fixture(power=2.0)
    witness = discover_output_link_separability(X, y, grad, hess, blocks=((0,), (1,)))

    assert witness.accepted
    assert witness.link_family == "power"
    assert abs(float(witness.power_exponent) - 2.0) < 1.0e-8
    assert witness.max_cross_pair_residual < 1.0e-10

    X, y, grad, hess = _power_link_fixture(power=-1.0)
    witness = discover_output_link_separability(X, y, grad, hess, blocks=((0,), (1,)))

    assert witness.accepted
    assert witness.link_family == "reciprocal"
    assert witness.max_cross_pair_residual < 1.0e-10
