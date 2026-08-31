# SPDX-License-Identifier: Apache-2.0
"""Drag and lift shape sensitivities w.r.t. the geometry curve x_surf.

Both come straight from single-objective OpenFOAM adjoints at a fixed angle of
attack, projected through the RBF morph transpose. Lift is a constraint the
optimizer carries itself, so nothing here differentiates a trim.
"""

import sys
from pathlib import Path

import numpy as np

CFD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CFD_DIR / "mesh"))
import morph  # noqa: E402
import runner  # noqa: E402

MORPH_RADIUS = 0.6
SENS_CACHE = "sensitivity.npz"
SENS_VERSION = 3

ARRAYS = ("dCd_dx_surf", "dCl_dx_surf")
META = ("alpha_deg_key", "version")


def _read_cache(workdir, alpha_deg):
    """A restarted sweep should not repay for designs it already solved."""
    path = Path(workdir) / SENS_CACHE
    if not path.exists():
        return None
    data = np.load(path)
    if "version" not in data.files or int(data["version"]) != SENS_VERSION:
        return None
    if float(data["alpha_deg_key"]) != alpha_deg:
        return None
    return {k: (data[k] if k in ARRAYS else float(data[k])) for k in data.files
            if k not in META}


def _write_cache(workdir, alpha_deg, result):
    np.savez(Path(workdir) / SENS_CACHE, alpha_deg_key=alpha_deg, version=SENS_VERSION,
             **{k: v for k, v in result.items()})
    return result


def shape_sensitivity(x_surf, alpha_deg, reynolds, workdir):
    """dCd/dx_surf and dCl/dx_surf, each (N, 2), plus the operating point."""
    x_surf = np.asarray(x_surf, dtype=np.float64)
    workdir = Path(workdir)
    primal = runner.run_primal(x_surf, alpha_deg, reynolds, workdir)

    with runner.workdir_lock(workdir / "adj"):
        cached = _read_cache(workdir, alpha_deg)
        if cached is not None:
            return cached

        primal_case = primal["case"]
        sens = runner.run_adjoint(primal_case, alpha_deg, reynolds, workdir / "adj")
        ref_points = runner.reference_points(workdir)

        def project(point_vector):
            return morph.morph_vjp(primal_case, x_surf, point_vector, radius=MORPH_RADIUS,
                                   reference_points=ref_points)

        return _write_cache(workdir, alpha_deg, {
            "dCd_dx_surf": project(sens["drag"]),
            "dCl_dx_surf": project(sens["lift"]),
            "Cd": primal["Cd"],
            "Cl": primal["Cl"],
            "Cm": primal["Cm"],
            "alpha_deg": alpha_deg,
        })
