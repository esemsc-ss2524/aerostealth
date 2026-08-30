# SPDX-License-Identifier: Apache-2.0
"""2D method-of-moments EFIE for a PEC contour, TM-z polarization.

Pulse-basis, point-matched EFIE with the standard log self term. The scatterer is
the closed polyline `verts` (metres); `k` is the free-space wavenumber. Returns
the monostatic echo width per incidence direction. Fully JAX-differentiable, so
the EM shape sensitivity is plain reverse-mode autodiff through this file.
"""

import jax.numpy as jnp
from bessel import hankel2

EULER_GAMMA = 1.7810724179901979852


def _segments(verts):
    p0 = verts
    p1 = jnp.roll(verts, -1, axis=0)
    return 0.5 * (p0 + p1), jnp.linalg.norm(p1 - p0, axis=1)


def impedance_matrix(mid, seglen, k):
    """(k / 4) * integral of H_0^(2)(k R) over each segment, midpoint rule off the
    diagonal and the small-argument log form on it. Drops the free-space eta;
    echo_width restores the matching factor."""
    n = mid.shape[0]
    diff = mid[:, None, :] - mid[None, :, :]
    r = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + jnp.eye(n))
    z_off = (k / 4.0) * seglen[None, :] * hankel2(k * r, 0)
    z_diag = (k * seglen / 4.0) * (
        1.0 - 1j * (2.0 / jnp.pi) * jnp.log(EULER_GAMMA * k * seglen / (4.0 * jnp.e))
    )
    return z_off * (1.0 - jnp.eye(n)) + jnp.diag(z_diag)


def echo_width(verts, k, prop_dirs):
    """Monostatic 2D radar cross section (echo width, metres) for each row of
    prop_dirs (unit incident propagation directions). verts: closed loop (N, 2)."""
    mid, seglen = _segments(verts)
    z = impedance_matrix(mid, seglen, k)
    phase = jnp.exp(-1j * k * (mid @ prop_dirs.T))          # (N, A)
    currents = jnp.linalg.solve(z, phase)                   # (N, A)
    amp = jnp.sum(currents * seglen[:, None] * phase, axis=0)  # (A,)  backscatter
    return (k / 4.0) * jnp.abs(amp) ** 2


def ks_aggregate(values, rho):
    """Kreisselmeier-Steinhauser smooth maximum in the log domain: an upper bound
    on max(values), scale invariant so rho has the same meaning at any RCS level."""
    logv = jnp.log(values)
    shift = jnp.max(logv)
    return jnp.exp(shift + jnp.log(jnp.sum(jnp.exp(rho * (logv - shift)))) / rho)


def incidence_dirs(angle_min_deg, angle_max_deg, count):
    theta = jnp.linspace(jnp.radians(angle_min_deg), jnp.radians(angle_max_deg), count)
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
