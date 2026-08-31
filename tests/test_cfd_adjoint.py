# SPDX-License-Identifier: Apache-2.0
"""Morph transpose consistency (fast) and the drag adjoint vs central FD (slow)."""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

MESH_DIR = Path(__file__).resolve().parents[1] / "tesseracts/cfd/mesh"
sys.path.insert(0, str(MESH_DIR))

import morph  # noqa: E402


def _ellipse(n, a=0.5, b=0.08, cx=0.5):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + a * np.cos(t), b * np.sin(t)])


def test_projection_weights_reconstruct_displacement():
    ref = _ellipse(90)
    target = _ellipse(220, b=0.11)
    w = morph.projection_weights(ref, target)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose(w @ target - ref, morph.surface_displacement(ref, target))


def test_operator_transpose_identity():
    centers = _ellipse(70)
    mesh_xy = np.vstack([centers, _ellipse(70, a=0.6, b=0.3), _ellipse(40, a=8, b=8)])
    op = morph.build_operator(mesh_xy, centers, radius=0.7)

    rng = np.random.default_rng(0)
    u = rng.standard_normal((centers.shape[0], 2))
    v = rng.standard_normal((mesh_xy.shape[0], 2))
    assert np.isclose(np.sum((op @ u) * v), np.sum(u * (op.T @ v)), rtol=1e-9)


_slow = os.environ.get("AEROSTEALTH_SLOW_TESTS") == "1"
_have_of = shutil.which("adjointOptimisationFoam") is not None


@pytest.mark.skipif(
    not (_slow and _have_of),
    reason="set AEROSTEALTH_SLOW_TESTS=1 with OpenFOAM for the adjoint vs FD check",
)
@pytest.mark.parametrize("coefficient", ["Cd", "Cl"])
def test_adjoint_vs_fd(coefficient):
    import jax

    jax.config.update("jax_enable_x64", True)

    import sensitivity

    from driver.tesseracts import local_tesseract

    geom = local_tesseract("geom")
    w = np.array([0.171, 0.154, 0.162, 0.135, 0.146, 0.144])
    theta = np.concatenate([w, -w])
    alpha, re = 3.7306, 6.0e6

    def xsurf(th):
        return np.asarray(geom.apply({"theta": th, "n_surface": 161})["x_surf"])

    tmp = Path(os.environ.get("PYTEST_TMPDIR", "/tmp")) / "cfd_adjoint_fd"
    sens = sensitivity.shape_sensitivity(xsurf(theta), alpha, re, tmp / "adj")
    dtheta = np.asarray(
        geom.vector_jacobian_product(
            {"theta": theta, "n_surface": 161}, ["theta"], ["x_surf"],
            {"x_surf": sens[f"d{coefficient}_dx_surf"]},
        )["theta"]
    )

    h, k = 0.006, 2
    tp, tm = theta.copy(), theta.copy()
    tp[k] += h
    tm[k] -= h
    plus = sensitivity.runner.run_primal(xsurf(tp), alpha, re, tmp / "p")[coefficient]
    minus = sensitivity.runner.run_primal(xsurf(tm), alpha, re, tmp / "m")[coefficient]
    fd = (plus - minus) / (2 * h)
    assert abs(dtheta[k] - fd) / (abs(fd) + 1e-9) < 0.2
