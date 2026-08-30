# SPDX-License-Identifier: Apache-2.0
"""2D monostatic echo width over a frontal incidence sector, via a MoM EFIE.

The PEC contour is x_surf from geom, scaled to a physical chord. TM-z
polarization. The whole map x_surf -> sigma_agg is JAX, so the EM shape
sensitivity is reverse-mode autodiff (no separate adjoint)."""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import mom
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)

C_LIGHT = 299792458.0


class InputSchema(BaseModel):
    x_surf: Differentiable[Array[(None, 2), Float64]]
    frequency_hz: float = 10.0e9
    chord_m: float = 0.05
    incidence_deg_min: float = -30.0
    incidence_deg_max: float = 30.0
    incidence_count: int = 13
    ks_rho: float = 50.0


class OutputSchema(BaseModel):
    sigma_agg: Differentiable[Float64]
    sigma_by_angle: Differentiable[Array[(None,), Float64]]


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    verts = inputs["x_surf"][:-1] * inputs["chord_m"]
    k = 2.0 * jnp.pi * inputs["frequency_hz"] / C_LIGHT
    dirs = mom.incidence_dirs(
        inputs["incidence_deg_min"], inputs["incidence_deg_max"], inputs["incidence_count"]
    )
    sigma = mom.echo_width(verts, k, dirs)
    return {"sigma_agg": mom.ks_aggregate(sigma, inputs["ks_rho"]), "sigma_by_angle": sigma}


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
