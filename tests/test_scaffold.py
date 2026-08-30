# SPDX-License-Identifier: Apache-2.0
"""M0: the three Tesseracts load through the local dev loop and their apply endpoints run."""

import numpy as np
import pytest

from driver.tesseracts import load_config, local_tesseract


@pytest.fixture(scope="module")
def theta():
    return np.array([0.05, -0.02, 0.03, 0.0, 0.01, 0.0, 0.0, 0.0])


def test_geom_apply(theta):
    out = local_tesseract("geom").apply({"theta": theta})
    x_surf = np.asarray(out["x_surf"])
    assert x_surf.shape == (2 * theta.size, 2)
    assert np.isfinite(x_surf).all()
    assert np.asarray(out["g_geom"]).shape == (2,)


def test_cfd_apply(theta):
    x_surf = np.asarray(local_tesseract("geom").apply({"theta": theta})["x_surf"])
    out = local_tesseract("cfd").apply({"x_surf": x_surf})
    assert {"Cd", "Cl", "Cm", "alpha_deg"} <= set(out)
    assert np.isfinite(float(out["Cd"]))


def test_em_apply(theta):
    x_surf = np.asarray(local_tesseract("geom").apply({"theta": theta})["x_surf"])
    level_set = np.tile(x_surf[:, 1], (x_surf.shape[0], 1))
    out = local_tesseract("em").apply({"level_set": level_set})
    assert np.isfinite(float(out["sigma_agg"]))
    assert np.asarray(out["sigma_by_angle"]).ndim == 1


def test_baseline_config_frozen():
    baseline = load_config("configs/baseline.yaml")
    assert baseline["aero"]["cl_target"] == 0.4
    assert baseline["em"]["incidence_count"] == 13
