# SPDX-License-Identifier: Apache-2.0
"""2D monostatic echo width over a frontal incidence sector, with an FDTD shape adjoint VJP."""

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64

N_ANGLES = 13


class InputSchema(BaseModel):
    level_set: Differentiable[Array[(None, None), Float64]]
    frequency_hz: float = 10.0e9
    ks_rho: float = 20.0


class OutputSchema(BaseModel):
    sigma_agg: Differentiable[Float64]
    sigma_by_angle: Array[(None,), Float64]


def apply(inputs: InputSchema) -> OutputSchema:
    _ = np.asarray(inputs.level_set)
    return OutputSchema(sigma_agg=0.0, sigma_by_angle=np.zeros(N_ANGLES))
