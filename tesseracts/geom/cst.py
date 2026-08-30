# SPDX-License-Identifier: Apache-2.0
"""CST (Kulfan) airfoil parameterization and geometric quantities, in JAX."""

import math

import jax.numpy as jnp

LE_EXPONENT = 0.5
TE_EXPONENT = 1.0


def cosine_psi(n):
    """Chordwise stations in [0, 1] clustered at the leading and trailing edges."""
    beta = jnp.linspace(0.0, jnp.pi, n)
    return 0.5 * (1.0 - jnp.cos(beta))


def _bernstein_matrix(psi, degree):
    k = jnp.arange(degree + 1)
    binom = jnp.array([math.comb(degree, int(kk)) for kk in range(degree + 1)], dtype=float)
    return binom * psi[:, None] ** k[None, :] * (1.0 - psi[:, None]) ** (degree - k)[None, :]


def _class_fn(psi):
    return psi**LE_EXPONENT * (1.0 - psi) ** TE_EXPONENT


def surface_y(psi, weights, te_offset):
    """One surface height at stations psi, with x = psi and chord 1."""
    bern = _bernstein_matrix(psi, weights.shape[0] - 1)
    return _class_fn(psi) * (bern @ weights) + psi * te_offset


def evaluate(theta, n_surface, te_thickness):
    """Split theta into upper/lower weights and return (psi, y_upper, y_lower)."""
    n = theta.shape[0] // 2
    psi = cosine_psi(n_surface)
    te_half = 0.5 * te_thickness
    y_upper = surface_y(psi, theta[:n], te_half)
    y_lower = surface_y(psi, theta[n:], -te_half)
    return psi, y_upper, y_lower


def boundary_loop(psi, y_upper, y_lower):
    """Closed boundary in Selig order: trailing edge over the upper surface to the
    leading edge, then along the lower surface back to the trailing edge."""
    upper = jnp.stack([psi[::-1], y_upper[::-1]], axis=-1)
    lower = jnp.stack([psi[1:], y_lower[1:]], axis=-1)
    return jnp.concatenate([upper, lower], axis=0)


def polygon_area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y))


def geometric_constraints(y_upper, y_lower, loop, bounds):
    """Constraint vector in g <= 0 form: max thickness bracketed by
    (thickness_min, thickness_max), trailing-edge gap, enclosed area."""
    thickness_min, thickness_max, te_gap_max, area_min = bounds
    max_thickness = jnp.max(y_upper - y_lower)
    te_gap = y_upper[-1] - y_lower[-1]
    return jnp.array(
        [
            thickness_min - max_thickness,
            max_thickness - thickness_max,
            jnp.abs(te_gap) - te_gap_max,
            area_min - polygon_area(loop),
        ]
    )
