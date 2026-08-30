# SPDX-License-Identifier: Apache-2.0
"""geom jacobian endpoint agrees with central finite differences for every output."""

import numpy as np
import pytest

from driver.tesseracts import local_tesseract

OVERRIDES = {"n_surface": 60, "raster_nx": 48, "raster_ny": 48}


@pytest.fixture(scope="module")
def geom():
    return local_tesseract("geom")


@pytest.fixture(scope="module")
def theta():
    rng = np.random.default_rng(1)
    return np.concatenate(
        [0.18 + 0.03 * rng.standard_normal(4), -0.15 + 0.03 * rng.standard_normal(4)]
    )


def _fd_jacobian(geom, theta, output, h=1e-6):
    cols = []
    for k in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[k] += h
        minus[k] -= h
        op = np.asarray(geom.apply({**OVERRIDES, "theta": plus})[output])
        om = np.asarray(geom.apply({**OVERRIDES, "theta": minus})[output])
        cols.append((op - om) / (2 * h))
    return np.stack(cols, axis=-1)


@pytest.mark.parametrize("output", ["x_surf", "g_geom", "level_set"])
def test_jacobian_vs_fd(geom, theta, output):
    analytic = np.asarray(
        geom.jacobian({**OVERRIDES, "theta": theta}, ["theta"], [output])[output]["theta"]
    )
    numeric = _fd_jacobian(geom, theta, output)
    scale = np.abs(numeric).max() + 1e-12
    assert np.max(np.abs(analytic - numeric)) / scale < 1e-6
