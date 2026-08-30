# SPDX-License-Identifier: Apache-2.0
"""Objective transforms shared by the driver: KS aggregation and baseline normalization."""

import jax.numpy as jnp


def ks_aggregate(values, rho):
    """Kreisselmeier-Steinhauser smooth maximum. Approaches max(values) as rho grows."""
    v = jnp.asarray(values)
    shift = jnp.max(v)
    return shift + jnp.log(jnp.sum(jnp.exp(rho * (v - shift)))) / rho


def normalize(value, baseline):
    return value / baseline
