# SPDX-License-Identifier: Apache-2.0
"""Real-argument Bessel functions J0, J1, Y0, Y1 in JAX (Abramowitz and Stegun 9.4).

Polynomial and asymptotic fits, accurate to about 1e-7, smooth for x > 0 so they
differentiate cleanly. Enough to build the 2D Hankel function for the EFIE.
"""

import jax.numpy as jnp

_J0_SMALL = (1.0, -2.2499997, 1.2656208, -0.3163866, 0.0444479, -0.0039444, 0.0002100)
_J0_F = (0.79788456, -0.00000077, -0.00552740, -0.00009512,
         0.00137237, -0.00072805, 0.00014476)
_J0_THETA = (-0.78539816, -0.04166397, -0.00003954, 0.00262573,
             -0.00054125, -0.00029333, 0.00013558)
_Y0_SMALL = (0.36746691, 0.60559366, -0.74350384, 0.25300117,
             -0.04261214, 0.00427916, -0.00024846)

_J1_SMALL = (0.5, -0.56249985, 0.21093573, -0.03954289,
             0.00443319, -0.00031761, 0.00001109)
_J1_F = (0.79788456, 0.00000156, 0.01659667, 0.00017105,
         -0.00249511, 0.00113653, -0.00020033)
_J1_THETA = (-2.35619449, 0.12499612, 0.00005650, -0.00637879,
             0.00074348, 0.00079824, -0.00029166)
_Y1_SMALL = (-0.6366198, 0.2212091, 2.1682709, -1.3164827,
             0.3123951, -0.0400976, 0.0027873)


def _poly(coeffs, x):
    acc = coeffs[-1]
    for c in coeffs[-2::-1]:
        acc = acc * x + c
    return acc


def j0(x):
    x = jnp.abs(x)
    small = _poly(_J0_SMALL, (x / 3.0) ** 2)
    t = 3.0 / jnp.where(x < 3.0, 3.0, x)
    f = _poly(_J0_F, t)
    theta = x + _poly(_J0_THETA, t)
    large = f * jnp.cos(theta) / jnp.sqrt(jnp.where(x < 3.0, 1.0, x))
    return jnp.where(x < 3.0, small, large)


def j1(x):
    s = jnp.sign(x)
    x = jnp.abs(x)
    small = x * _poly(_J1_SMALL, (x / 3.0) ** 2)
    t = 3.0 / jnp.where(x < 3.0, 3.0, x)
    f = _poly(_J1_F, t)
    theta = x + _poly(_J1_THETA, t)
    large = f * jnp.cos(theta) / jnp.sqrt(jnp.where(x < 3.0, 1.0, x))
    return s * jnp.where(x < 3.0, small, large)


def y0(x):
    z = (x / 3.0) ** 2
    small = (2.0 / jnp.pi) * jnp.log(x / 2.0) * j0(x) + _poly(_Y0_SMALL, z)
    t = 3.0 / jnp.where(x < 3.0, 3.0, x)
    f = _poly(_J0_F, t)
    theta = x + _poly(_J0_THETA, t)
    large = f * jnp.sin(theta) / jnp.sqrt(jnp.where(x < 3.0, 1.0, x))
    return jnp.where(x < 3.0, small, large)


def y1(x):
    z = (x / 3.0) ** 2
    small = ((2.0 / jnp.pi) * (x * jnp.log(x / 2.0) * j1(x)) + _poly(_Y1_SMALL, z)) / jnp.where(
        x < 3.0, x, 1.0
    )
    t = 3.0 / jnp.where(x < 3.0, 3.0, x)
    f = _poly(_J1_F, t)
    theta = x + _poly(_J1_THETA, t)
    large = f * jnp.sin(theta) / jnp.sqrt(jnp.where(x < 3.0, 1.0, x))
    return jnp.where(x < 3.0, small, large)


def hankel2(x, order=0):
    """H_order^(2)(x) = J_order(x) - i Y_order(x), for order 0 or 1."""
    if order == 0:
        return j0(x) - 1j * y0(x)
    return j1(x) - 1j * y1(x)
