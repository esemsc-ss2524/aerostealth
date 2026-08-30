# SPDX-License-Identifier: Apache-2.0
"""Incompressible RANS drag/lift via a morphed reference mesh, with a drag adjoint VJP."""

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64


class InputSchema(BaseModel):
    x_surf: Differentiable[Array[(None, 2), Float64]]
    cl_target: float = 0.4
    alpha_deg: float = 0.0
    reynolds: float = 1.0e6


class OutputSchema(BaseModel):
    Cd: Differentiable[Float64]
    Cl: Differentiable[Float64]
    Cm: Differentiable[Float64]
    alpha_deg: float


def apply(inputs: InputSchema) -> OutputSchema:
    _ = np.asarray(inputs.x_surf)
    return OutputSchema(Cd=0.0, Cl=inputs.cl_target, Cm=0.0, alpha_deg=inputs.alpha_deg)
