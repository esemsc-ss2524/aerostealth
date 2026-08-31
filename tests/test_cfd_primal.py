# SPDX-License-Identifier: Apache-2.0
"""End-to-end cfd primal: geom -> morph -> simpleFoam -> Cd/Cl at fixed alpha.

Slow (shells out to OpenFOAM, minutes). Opt in with AEROSTEALTH_SLOW_TESTS=1.
"""

import os
import shutil

import numpy as np
import pytest

from driver.tesseracts import local_tesseract

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("AEROSTEALTH_SLOW_TESTS") != "1",
        reason="set AEROSTEALTH_SLOW_TESTS=1 to run",
    ),
    pytest.mark.skipif(shutil.which("blockMesh") is None, reason="OpenFOAM not on PATH"),
]


def test_primal_coefficients():
    geom = local_tesseract("geom")
    cfd = local_tesseract("cfd")

    theta = np.array([0.17, 0.15, 0.16, 0.14, 0.15, 0.14])
    theta = np.concatenate([theta, -theta])
    x_surf = np.asarray(geom.apply({"theta": theta, "n_surface": 161})["x_surf"])

    out = cfd.apply({"x_surf": x_surf, "alpha_deg": 3.7306})
    assert 0.3 < float(out["Cl"]) < 0.5
    assert 0.005 < float(out["Cd"]) < 0.05
