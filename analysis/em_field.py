# SPDX-License-Identifier: Apache-2.0
"""Scattered and total near field for the 2D PEC scattering problem.

Same discretization as the `em` Tesseract, in numpy rather than JAX because
nothing here is differentiated. Lengths are in chord units, so the wavenumber
passed in is $k c$.
"""

import numpy as np
from scipy.special import hankel2

EULER_GAMMA = 1.7810724179901979852


def _segments(verts):
    p0 = np.asarray(verts, dtype=float)
    p1 = np.roll(p0, -1, axis=0)
    return 0.5 * (p0 + p1), np.linalg.norm(p1 - p0, axis=1)


def impedance(mid, seglen, k):
    n = len(mid)
    diff = mid[:, None, :] - mid[None, :, :]
    r = np.sqrt((diff**2).sum(-1) + np.eye(n))
    z = (k / 4.0) * seglen[None, :] * hankel2(0, k * r)
    diag = (k * seglen / 4.0) * (
        1.0 - 1j * (2.0 / np.pi) * np.log(EULER_GAMMA * k * seglen / (4.0 * np.e))
    )
    return z * (1.0 - np.eye(n)) + np.diag(diag)


def currents(verts, k, direction):
    """Surface current for a plane wave travelling along `direction`."""
    mid, seglen = _segments(verts)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    rhs = np.exp(-1j * k * (mid @ d))
    return np.linalg.solve(impedance(mid, seglen, k), rhs), mid, seglen


def near_field(verts, k, direction, gx, gy, chunk=20000):
    """Total field on a grid: incident plane wave plus the scattered field."""
    j, mid, seglen = currents(verts, k, direction)
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)

    pts = np.column_stack([gx.ravel(), gy.ravel()])
    scattered = np.zeros(len(pts), dtype=complex)
    weight = (k / 4.0) * seglen * j
    for i in range(0, len(pts), chunk):
        block = pts[i : i + chunk]
        r = np.linalg.norm(block[:, None, :] - mid[None, :, :], axis=-1)
        scattered[i : i + chunk] = -(hankel2(0, k * np.maximum(r, 1e-9)) @ weight)

    incident = np.exp(-1j * k * (pts @ d))
    return (incident + scattered).reshape(gx.shape)


def echo_width(verts, k, directions):
    """Monostatic echo width per incidence direction, for the polar plot."""
    out = []
    for d in np.atleast_2d(directions):
        j, mid, seglen = currents(verts, k, d)
        d = np.asarray(d, dtype=float) / np.linalg.norm(d)
        amp = np.sum(j * seglen * np.exp(-1j * k * (mid @ d)))
        out.append((k / 4.0) * abs(amp) ** 2)
    return np.array(out)
