# SPDX-License-Identifier: Apache-2.0
"""Differentiable rasterization of a closed polygon to a signed-distance field."""

import jax.numpy as jnp
from jax import lax


def _safe_sqrt(x):
    return jnp.sqrt(jnp.maximum(x, 1e-24))


def signed_distance_field(loop, bbox, nx, ny):
    """Signed distance on an (ny, nx) grid spanning bbox = (xmin, xmax, ymin, ymax).

    Negative inside the polygon, positive outside. Distance is exact to the polygon
    edges; the sign comes from the winding number so it is smooth in the vertex
    positions everywhere except on the boundary itself.
    """
    xmin, xmax, ymin, ymax = bbox
    gx, gy = jnp.meshgrid(
        jnp.linspace(xmin, xmax, nx), jnp.linspace(ymin, ymax, ny), indexing="xy"
    )
    a = loop
    b = jnp.roll(loop, -1, axis=0)
    segments = jnp.stack([a[:, 0], a[:, 1], b[:, 0], b[:, 1]], axis=-1)

    def accumulate(carry, segment):
        min_d2, angle = carry
        ax, ay, bx, by = segment
        ex, ey = bx - ax, by - ay
        wx, wy = gx - ax, gy - ay
        t = jnp.clip((wx * ex + wy * ey) / (ex * ex + ey * ey + 1e-30), 0.0, 1.0)
        rx, ry = wx - t * ex, wy - t * ey
        d2 = rx * rx + ry * ry

        vax, vay = ax - gx, ay - gy
        vbx, vby = bx - gx, by - gy
        cross = vax * vby - vay * vbx
        dot = vax * vbx + vay * vby
        return (jnp.minimum(min_d2, d2), angle + jnp.arctan2(cross, dot)), None

    init = (jnp.full((ny, nx), jnp.inf), jnp.zeros((ny, nx)))
    (min_d2, angle), _ = lax.scan(accumulate, init, segments)
    sign = jnp.where(jnp.abs(angle) > jnp.pi, -1.0, 1.0)
    return sign * _safe_sqrt(min_d2)
