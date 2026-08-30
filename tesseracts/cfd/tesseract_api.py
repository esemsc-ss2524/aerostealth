# SPDX-License-Identifier: Apache-2.0
"""Incompressible RANS drag/lift on a morphed reference C-mesh, trimmed to a target
lift by an inner angle-of-attack Newton solve. Drag adjoint VJP to follow."""

import hashlib

import numpy as np
import runner
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64

RUN_ROOT = runner.CFD_DIR / "_run"


class InputSchema(BaseModel):
    x_surf: Differentiable[Array[(None, 2), Float64]]
    cl_target: float = 0.4
    alpha_deg: float = 3.0
    reynolds: float = 1.0e6


class OutputSchema(BaseModel):
    Cd: Differentiable[Float64]
    Cl: Differentiable[Float64]
    Cm: Differentiable[Float64]
    alpha_deg: float
    trim_iterations: int


def _workdir(x_surf, reynolds):
    key = hashlib.sha1(
        np.ascontiguousarray(x_surf, dtype=np.float64).tobytes()
        + np.float64([reynolds]).tobytes()
    ).hexdigest()[:16]
    return RUN_ROOT / key


def apply(inputs: InputSchema) -> OutputSchema:
    x_surf = np.asarray(inputs.x_surf, dtype=np.float64)
    result = runner.run_trim(
        x_surf,
        inputs.cl_target,
        inputs.reynolds,
        _workdir(x_surf, inputs.reynolds),
        alpha0=inputs.alpha_deg,
    )
    return OutputSchema(
        Cd=result["Cd"],
        Cl=result["Cl"],
        Cm=result["Cm"],
        alpha_deg=result["alpha_deg"],
        trim_iterations=result["trim_iterations"],
    )
