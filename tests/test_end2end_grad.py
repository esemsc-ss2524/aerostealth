# SPDX-License-Identifier: Apache-2.0
"""The composed map theta -> objectives differentiates end to end, checked vs FD.

The fast checks run the all-JAX path (geom -> em plus the geometric
constraints). The aero check adds the OpenFOAM leg and needs
AEROSTEALTH_SLOW_TESTS=1.
"""

import os
import shutil
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.forward import baseline_theta, build_forward  # noqa: E402
from driver.tesseracts import load_config  # noqa: E402

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def config():
    return load_config(ROOT / "configs/baseline.yaml")


def _central_diff(fn, theta, k, h):
    return (fn(theta.at[k].add(h)) - fn(theta.at[k].add(-h))) / (2.0 * h)


def test_forward_runs_and_is_feasible(config):
    out = build_forward(config, aero=False)(baseline_theta(config))
    assert np.isfinite(out["sigma_agg"]) and out["sigma_agg"] > 0
    assert np.all(np.asarray(out["g_geom"]) <= 0)


def test_sigma_gradient_vs_fd(config):
    forward = build_forward(config, aero=False)
    theta = baseline_theta(config)

    def sigma(th):
        return forward(th)["sigma_agg"]

    grad = jax.grad(sigma)(theta)
    for k in (0, 2, 5):
        fd = _central_diff(sigma, theta, k, 1e-6)
        assert abs(grad[k] - fd) <= 1e-6 * abs(fd)


def test_constraint_gradient_vs_fd(config):
    forward = build_forward(config, aero=False)
    theta = baseline_theta(config)

    def thickness(th):
        return forward(th)["g_geom"][1]

    grad = jax.grad(thickness)(theta)
    fd = _central_diff(thickness, theta, 0, 1e-6)
    assert abs(grad[0] - fd) <= 1e-6 * abs(fd)


@pytest.mark.skipif(
    not (os.environ.get("AEROSTEALTH_SLOW_TESTS") == "1"
         and shutil.which("simpleFoam") is not None),
    reason="set AEROSTEALTH_SLOW_TESTS=1 with OpenFOAM for the aero leg",
)
def test_drag_gradient_vs_fd(config):
    forward = build_forward(config, aero=True)
    theta = baseline_theta(config)

    def cd(th):
        return forward(th)["Cd"]

    grad = jax.grad(cd)(theta)
    fd = _central_diff(cd, theta, 0, 5e-3)
    assert np.sign(grad[0]) == np.sign(fd)
    assert abs(grad[0] - fd) <= 0.6 * abs(fd)
