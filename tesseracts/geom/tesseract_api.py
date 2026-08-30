# SPDX-License-Identifier: Apache-2.0
"""Shared geometry spine: design vector to boundary curve and geometric constraints."""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)


class InputSchema(BaseModel):
    theta: Differentiable[Array[(None,), Float64]]


class OutputSchema(BaseModel):
    x_surf: Differentiable[Array[(None, 2), Float64]]
    g_geom: Differentiable[Array[(None,), Float64]]


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    theta = inputs["theta"]
    n = theta.shape[0]
    s = jnp.linspace(0.0, 1.0, 2 * n)
    modes = jnp.sin(jnp.pi * jnp.outer(s, jnp.arange(1, n + 1)))
    offset = modes @ theta
    x_surf = jnp.stack([s, offset], axis=-1)
    g_geom = jnp.array(
        [
            jnp.max(jnp.abs(offset)) - 0.5,
            0.01 - jnp.mean(jnp.abs(offset)),
        ]
    )
    return {"x_surf": x_surf, "g_geom": g_geom}


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
