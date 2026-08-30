# SPDX-License-Identifier: Apache-2.0
"""geom shape correctness: ordering, closure, symmetry, level-set sign, constraints."""

import numpy as np
import pytest

from driver.tesseracts import local_tesseract

OVERRIDES = {"n_surface": 80, "raster_nx": 64, "raster_ny": 64}
BBOX = (-0.5, 1.5, -0.75, 0.75)


@pytest.fixture(scope="module")
def geom():
    return local_tesseract("geom")


@pytest.fixture(scope="module")
def symmetric_theta():
    w = np.array([0.20, 0.24, 0.19, 0.15])
    return np.concatenate([w, -w])


def _sample(level_set, x, y):
    ix = round((x - BBOX[0]) / (BBOX[1] - BBOX[0]) * (OVERRIDES["raster_nx"] - 1))
    iy = round((y - BBOX[2]) / (BBOX[3] - BBOX[2]) * (OVERRIDES["raster_ny"] - 1))
    return level_set[iy, ix]


def test_loop_ordering_and_closure(geom, symmetric_theta):
    x_surf = np.asarray(geom.apply({**OVERRIDES, "theta": symmetric_theta})["x_surf"])
    assert x_surf.shape == (2 * OVERRIDES["n_surface"] - 1, 2)
    assert np.allclose(x_surf[0], [1.0, 0.0])
    assert np.allclose(x_surf[-1], [1.0, 0.0])
    le = x_surf[x_surf[:, 0].argmin()]
    assert np.allclose(le, [0.0, 0.0], atol=1e-9)


def test_symmetric_airfoil_has_symmetric_field(geom, symmetric_theta):
    out = geom.apply({**OVERRIDES, "theta": symmetric_theta})
    x_surf = np.asarray(out["x_surf"])
    n = OVERRIDES["n_surface"]
    upper, lower = x_surf[:n], x_surf[n - 1 :][::-1]
    assert np.allclose(upper[:, 1], -lower[:, 1], atol=1e-12)

    level_set = np.asarray(out["level_set"])
    assert np.allclose(level_set, level_set[::-1, :], atol=1e-9)


def test_level_set_sign(geom, symmetric_theta):
    level_set = np.asarray(geom.apply({**OVERRIDES, "theta": symmetric_theta})["level_set"])
    assert _sample(level_set, 0.3, 0.0) < 0.0
    assert _sample(level_set, 1.0, 0.6) > 0.0
    assert _sample(level_set, -0.4, 0.0) > 0.0


def test_constraints_and_area(geom, symmetric_theta):
    out = geom.apply(
        {**OVERRIDES, "theta": symmetric_theta, "thickness_min": 0.02, "thickness_max": 0.3}
    )
    g_geom = np.asarray(out["g_geom"])
    assert g_geom.shape == (4,)
    assert g_geom[0] < 0.0 and g_geom[1] < 0.0
    assert g_geom[2] < 0.0
