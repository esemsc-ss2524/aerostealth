# SPDX-License-Identifier: Apache-2.0
"""Incompressible RANS drag and lift on a morphed reference C-mesh at a fixed
angle of attack, with drag and lift adjoint VJPs."""

import hashlib
from typing import Any

import numpy as np
import runner
import sensitivity
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64

RUN_ROOT = runner.CFD_DIR / "_run"


class InputSchema(BaseModel):
    x_surf: Differentiable[Array[(None, 2), Float64]]
    alpha_deg: float = 3.7306
    reynolds: float = 6.0e6


class OutputSchema(BaseModel):
    Cd: Differentiable[Float64]
    Cl: Differentiable[Float64]
    Cm: Differentiable[Float64]


def _workdir(x_surf, alpha_deg, reynolds):
    key = hashlib.sha1(
        np.ascontiguousarray(x_surf, dtype=np.float64).tobytes()
        + np.float64([alpha_deg, reynolds]).tobytes()
    ).hexdigest()[:16]
    return RUN_ROOT / key


def abstract_eval(abstract_inputs):
    """Output shapes without running OpenFOAM: the three coefficients are scalars."""
    return {name: {"shape": (), "dtype": "float64"} for name in ("Cd", "Cl", "Cm")}


def apply(inputs: InputSchema) -> OutputSchema:
    x_surf = np.asarray(inputs.x_surf, dtype=np.float64)
    result = runner.run_primal(
        x_surf,
        inputs.alpha_deg,
        inputs.reynolds,
        _workdir(x_surf, inputs.alpha_deg, inputs.reynolds),
    )
    return OutputSchema(Cd=result["Cd"], Cl=result["Cl"], Cm=result["Cm"])


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    x_surf = np.asarray(inputs.x_surf, dtype=np.float64)
    sens = sensitivity.shape_sensitivity(
        x_surf, inputs.alpha_deg, inputs.reynolds,
        _workdir(x_surf, inputs.alpha_deg, inputs.reynolds),
    )
    grad = np.zeros_like(x_surf)
    if "Cd" in cotangent_vector:
        grad += float(cotangent_vector["Cd"]) * sens["dCd_dx_surf"]
    if "Cl" in cotangent_vector:
        grad += float(cotangent_vector["Cl"]) * sens["dCl_dx_surf"]
    return {"x_surf": grad}
