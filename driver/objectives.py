# SPDX-License-Identifier: Apache-2.0
"""Objective transforms shared by the driver: KS aggregation and baseline normalization.

Cd sits near 1e-2 and the echo width near 1e-2 m, but their shape sensitivities
differ by orders of magnitude, so both objectives are divided by their baseline
values before the optimizer sees them. Everything downstream is dimensionless
and of order one.
"""

import jax.numpy as jnp


def ks_aggregate(values, rho):
    """Kreisselmeier-Steinhauser smooth maximum. Approaches max(values) as rho grows."""
    v = jnp.asarray(values)
    shift = jnp.max(v)
    return shift + jnp.log(jnp.sum(jnp.exp(rho * (v - shift)))) / rho


def normalize(value, baseline):
    return value / baseline


def references(config):
    n = config["normalization"]
    return float(n["cd_ref"]), float(n["sigma_ref"])


def epsilon_problem(forward, config, eps):
    """The epsilon-constraint subproblem as (objective, constraints).

    objective(theta) -> Cd / cd_ref
    constraints(theta) -> vector that must stay <= 0: the normalized echo width
    against eps, then the geometric constraints.
    """
    cd_ref, sigma_ref = references(config)

    def objective(theta):
        return forward(theta)["Cd"] / cd_ref

    def constraints(theta):
        out = forward(theta)
        return jnp.concatenate([
            jnp.atleast_1d(out["sigma_agg"] / sigma_ref - eps),
            out["g_geom"],
        ])

    return objective, constraints
