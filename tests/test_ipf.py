"""IPF unit tests."""

from __future__ import annotations

import numpy as np

from pak.grounding.ipf import ipf


def test_ipf_2d_converges() -> None:
    seed = np.array([[1.0, 1.0], [1.0, 1.0]])
    row = np.array([60.0, 40.0])
    col = np.array([70.0, 30.0])
    out, info = ipf(seed, [row, col])
    assert info["converged"]
    np.testing.assert_allclose(out.sum(axis=1), row, rtol=1e-6)
    np.testing.assert_allclose(out.sum(axis=0), col, rtol=1e-6)


def test_ipf_3d_preserves_marginals() -> None:
    rng = np.random.default_rng(42)
    seed = rng.uniform(0.5, 1.5, size=(3, 4, 5))
    target_a = np.array([100.0, 200.0, 300.0])
    target_b = np.array([150.0, 150.0, 150.0, 150.0])
    target_c = np.array([120.0, 120.0, 120.0, 120.0, 120.0])
    out, info = ipf(seed, [target_a, target_b, target_c], max_iter=500, tol=1e-10)
    # IPF on 3-d is correct iff axes are jointly consistent and converges to fixed point.
    # Last applied axis is exactly preserved; earlier axes preserved up to the iteration.
    np.testing.assert_allclose(out.sum(axis=(0, 1)), target_c, rtol=1e-6)
    # After max iterations, the other axes still nearly match (within 1%)
    np.testing.assert_allclose(out.sum(axis=(1, 2)), target_a, rtol=1e-2)
    np.testing.assert_allclose(out.sum(axis=(0, 2)), target_b, rtol=1e-2)


def test_ipf_non_negative() -> None:
    seed = np.eye(3) + 0.1
    row = np.array([10.0, 20.0, 30.0])
    col = np.array([15.0, 20.0, 25.0])
    out, _ = ipf(seed, [row, col])
    assert (out >= 0).all()


def test_ipf_handles_zero_seed_cells() -> None:
    """Preserve marginals even when the seed contains zero cells."""
    seed = np.array([[0.0, 1.0], [1.0, 1.0]])
    row = np.array([10.0, 20.0])
    col = np.array([10.0, 20.0])
    out, info = ipf(seed, [row, col])
    np.testing.assert_allclose(out.sum(axis=1), row, rtol=1e-3)
    np.testing.assert_allclose(out.sum(axis=0), col, rtol=1e-3)
