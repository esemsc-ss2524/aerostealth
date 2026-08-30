# SPDX-License-Identifier: Apache-2.0
"""The three Tesseracts load through the local dev loop and their apply endpoints run."""

import numpy as np
import pytest

from driver.tesseracts import load_config, local_tesseract

OVERRIDES = {"n_surface": 40, "raster_nx": 32, "raster_ny": 32}


@pytest.fixture(scope="module")
def theta():
    return np.array([0.2, 0.24, 0.19, 0.15, -0.2, -0.24, -0.19, -0.15])


@pytest.fixture(scope="module")
def geom_out(theta):
    return local_tesseract("geom").apply({**OVERRIDES, "theta": theta})


def test_geom_apply(geom_out):
    x_surf = np.asarray(geom_out["x_surf"])
    assert x_surf.shape == (2 * OVERRIDES["n_surface"] - 1, 2)
    assert np.isfinite(x_surf).all()
    assert np.asarray(geom_out["level_set"]).shape == (
        OVERRIDES["raster_ny"],
        OVERRIDES["raster_nx"],
    )
    assert np.asarray(geom_out["g_geom"]).shape == (4,)


def test_cfd_loads():
    # apply shells out to OpenFOAM (see tests/test_cfd_primal.py); here just
    # check the Tesseract loads and exposes the expected interface.
    cfd = local_tesseract("cfd")
    assert "apply" in cfd.available_endpoints


def test_em_apply(geom_out):
    out = local_tesseract("em").apply({"level_set": np.asarray(geom_out["level_set"])})
    assert np.isfinite(float(out["sigma_agg"]))
    assert np.asarray(out["sigma_by_angle"]).ndim == 1


def test_baseline_config_frozen():
    baseline = load_config("configs/baseline.yaml")
    assert baseline["aero"]["cl_target"] == 0.4
    assert baseline["em"]["incidence_count"] == 13
