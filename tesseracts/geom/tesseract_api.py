# SPDX-License-Identifier: Apache-2.0
"""Shared geometry spine: CST design vector to boundary curve, level set, and constraints."""

from typing import Any

import cst
import equinox as eqx
import raster
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
    n_surface: int = 120
    te_thickness: float = 0.0
    raster_bbox: tuple[float, float, float, float] = (-0.5, 1.5, -0.75, 0.75)
    raster_nx: int = 96
    raster_ny: int = 96
    thickness_min: float = 0.02
    thickness_max: float = 0.16
    trailing_edge_gap_max: float = 1.0e-3
    enclosed_area_min: float = 0.05


class OutputSchema(BaseModel):
    x_surf: Differentiable[Array[(None, 2), Float64]]
    level_set: Differentiable[Array[(None, None), Float64]]
    g_geom: Differentiable[Array[(None,), Float64]]


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    psi, y_upper, y_lower = cst.evaluate(
        inputs["theta"], inputs["n_surface"], inputs["te_thickness"]
    )
    loop = cst.boundary_loop(psi, y_upper, y_lower)
    level_set = raster.signed_distance_field(
        loop, inputs["raster_bbox"], inputs["raster_nx"], inputs["raster_ny"]
    )
    g_geom = cst.geometric_constraints(
        y_upper,
        y_lower,
        loop,
        (
            inputs["thickness_min"],
            inputs["thickness_max"],
            inputs["trailing_edge_gap_max"],
            inputs["enclosed_area_min"],
        ),
    )
    return {"x_surf": loop, "level_set": level_set, "g_geom": g_geom}


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
