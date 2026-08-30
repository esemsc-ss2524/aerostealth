# SPDX-License-Identifier: Apache-2.0
"""em jacobian (reverse-mode autodiff through the MoM EFIE) vs central FD."""

import numpy as np
import pytest

from driver.tesseracts import local_tesseract

OVERRIDES = {"n_surface": 60, "incidence_count": 7}


@pytest.fixture(scope="module")
def setup():
    geom = local_tesseract("geom")
    em = local_tesseract("em")
    w = np.array([0.17, 0.15, 0.16, 0.14, 0.15, 0.14])
    theta = np.concatenate([w, -w])
    x_surf = np.asarray(
        geom.apply({"theta": theta, "n_surface": OVERRIDES["n_surface"]})["x_surf"]
    )
    return em, x_surf


@pytest.mark.parametrize("output", ["sigma_agg", "sigma_by_angle"])
def test_jacobian_vs_fd(setup, output):
    em, x_surf = setup
    analytic = np.asarray(
        em.jacobian({"x_surf": x_surf, **{k: v for k, v in OVERRIDES.items() if k != "n_surface"}},
                    ["x_surf"], [output])[output]["x_surf"]
    )
    h = 1e-7
    rng = np.random.default_rng(0)
    for _ in range(5):
        i, j = rng.integers(1, x_surf.shape[0] - 1), rng.integers(0, 2)
        plus, minus = x_surf.copy(), x_surf.copy()
        plus[i, j] += h
        minus[i, j] -= h
        op = np.asarray(em.apply({"x_surf": plus, "incidence_count": 7})[output])
        om = np.asarray(em.apply({"x_surf": minus, "incidence_count": 7})[output])
        fd = (op - om) / (2 * h)
        assert np.allclose(analytic[..., i, j], fd, rtol=1e-4, atol=1e-8)
