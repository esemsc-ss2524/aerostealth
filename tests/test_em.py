# SPDX-License-Identifier: Apache-2.0
"""EM primal: Bessel fits vs scipy, MoM echo width vs the PEC cylinder series."""

import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy import special

jax.config.update("jax_enable_x64", True)

EM_DIR = Path(__file__).resolve().parents[1] / "tesseracts/em"
sys.path.insert(0, str(EM_DIR))

import bessel  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mom  # noqa: E402

from driver.tesseracts import local_tesseract  # noqa: E402


@pytest.mark.parametrize(
    "name,jf,sf",
    [("j0", bessel.j0, special.j0), ("j1", bessel.j1, special.j1),
     ("y0", bessel.y0, special.y0), ("y1", bessel.y1, special.y1)],
)
def test_bessel_matches_scipy(name, jf, sf):
    x = np.concatenate([np.linspace(0.05, 2.99, 300), np.linspace(3.0, 50.0, 500)])
    assert np.max(np.abs(np.asarray(jf(jnp.asarray(x))) - sf(x))) < 5e-6


def _cylinder_sigma(ka, k):
    n = np.arange(int(ka + 20) + 1)
    eps = np.where(n == 0, 1.0, 2.0)
    term = eps * (-1.0) ** n * special.jv(n, ka) / special.hankel2(n, ka)
    return (4.0 / k) * np.abs(term.sum()) ** 2


@pytest.mark.parametrize("ka", [3.0, 6.0, 10.0])
def test_mom_circle_vs_series(ka):
    k = ka
    t = jnp.linspace(0.0, 2.0 * jnp.pi, max(200, int(24 * ka)), endpoint=False)
    verts = jnp.stack([jnp.cos(t), jnp.sin(t)], axis=-1)
    sigma = float(mom.echo_width(verts, k, jnp.array([[1.0, 0.0]]))[0])
    assert abs(sigma - _cylinder_sigma(ka, k)) / _cylinder_sigma(ka, k) < 0.01


def test_em_apply_and_symmetry():
    geom = local_tesseract("geom")
    em = local_tesseract("em")
    w = np.array([0.17, 0.15, 0.16, 0.14, 0.15, 0.14])
    x_surf = np.asarray(geom.apply({"theta": np.concatenate([w, -w]), "n_surface": 81})["x_surf"])

    out = em.apply({"x_surf": x_surf})
    sigma = np.asarray(out["sigma_by_angle"])
    assert sigma.shape == (13,)
    assert (sigma > 0).all()
    assert float(out["sigma_agg"]) >= sigma.max()
    assert np.allclose(sigma, sigma[::-1], rtol=1e-6)
