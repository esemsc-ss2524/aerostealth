# SPDX-License-Identifier: Apache-2.0
"""geom jacobian endpoint agrees with central finite differences on x_surf w.r.t. theta."""

import numpy as np

from driver.tesseracts import local_tesseract


def _x_surf(geom, theta):
    return np.asarray(geom.apply({"theta": theta})["x_surf"])


def test_geom_jacobian_vs_fd():
    geom = local_tesseract("geom")
    rng = np.random.default_rng(0)
    theta = 0.05 * rng.standard_normal(8)

    jac = geom.jacobian({"theta": theta}, ["theta"], ["x_surf"])
    jac_analytic = np.asarray(jac["x_surf"]["theta"])

    h = 1e-6
    jac_fd = np.empty_like(jac_analytic)
    for k in range(theta.size):
        step = np.zeros_like(theta)
        step[k] = h
        jac_fd[..., k] = (_x_surf(geom, theta + step) - _x_surf(geom, theta - step)) / (2 * h)

    assert np.allclose(jac_analytic, jac_fd, rtol=1e-5, atol=1e-7)
