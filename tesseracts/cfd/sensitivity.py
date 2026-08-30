# SPDX-License-Identifier: Apache-2.0
"""Trimmed drag and lift shape sensitivities w.r.t. the geometry curve x_surf.

Combines the OpenFOAM drag and lift adjoints, the angle-of-attack derivatives,
and the RBF morph transpose. With the inner trim holding Cl at cl_target, the
gradient the optimizer sees is

    dCd/dx|_trim = dCd/dx - (dCd/dalpha / dCl/dalpha) dCl/dx
"""

import sys
from pathlib import Path

import numpy as np

CFD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CFD_DIR / "mesh"))
import morph  # noqa: E402
import runner  # noqa: E402

MORPH_RADIUS = 0.6


def trimmed_sensitivity(x_surf, cl_target, reynolds, workdir, alpha0=3.0):
    """Run the trim, both adjoints, and the angle-of-attack derivatives; return
    dCd/dx|_trim and dCl/dx (each (N, 2)) plus the trimmed operating point."""
    x_surf = np.asarray(x_surf, dtype=np.float64)
    workdir = Path(workdir)

    trim = runner.run_trim(x_surf, cl_target, reynolds, workdir / "trim", alpha0=alpha0)
    alpha = trim["alpha_deg"]
    primal_case = trim["case"]

    sens = runner.run_adjoint(primal_case, alpha, reynolds, workdir / "adj")
    dcda = runner.alpha_derivatives(primal_case, alpha, reynolds, workdir / "da")

    def project(point_vector):
        return morph.morph_vjp(primal_case, x_surf, point_vector, radius=MORPH_RADIUS)

    dCd_dx = project(sens["drag"])
    dCl_dx = project(sens["lift"])
    dCd_trim_dx = dCd_dx - (dcda["dCd_da"] / dcda["dCl_da"]) * dCl_dx

    return {
        "dCd_trim_dx_surf": dCd_trim_dx,
        "dCl_dx_surf": dCl_dx,
        "Cd": trim["Cd"],
        "Cl": trim["Cl"],
        "Cm": trim["Cm"],
        "alpha_deg": alpha,
        "dCd_dalpha": dcda["dCd_da"],
        "dCl_dalpha": dcda["dCl_da"],
        "trim_iterations": trim["trim_iterations"],
    }
