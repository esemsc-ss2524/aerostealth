# SPDX-License-Identifier: Apache-2.0
"""Drive an OpenFOAM incompressible RANS primal on the morphed reference mesh,
with an inner angle-of-attack Newton solve to trim to a target lift."""

import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

CFD_DIR = Path(__file__).resolve().parent
TEMPLATE = CFD_DIR / "case_template"
DEFAULT_BASHRC = Path.home() / "side-projects/openfoam/etc/bashrc"
UINF = 1.0
CHORD = 1.0
FIELDS = ("U", "p", "nut", "nuTilda")

sys.path.insert(0, str(CFD_DIR / "mesh"))
import morph  # noqa: E402


def _bashrc():
    return os.environ.get("AEROSTEALTH_OF_BASHRC", str(DEFAULT_BASHRC))


def _foam(args, case, log):
    cmd = f"source {_bashrc()} && {args} -case {case}"
    with open(log, "w") as fh:
        proc = subprocess.run(
            ["bash", "-lc", cmd], stdout=fh, stderr=subprocess.STDOUT, text=True
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{args} failed (see {log})")


def _reference_mesh(cache_root):
    ref = Path(cache_root) / "reference"
    if not (ref / "constant/polyMesh/points").exists():
        shutil.rmtree(ref, ignore_errors=True)
        shutil.copytree(TEMPLATE, ref)
        _foam("blockMesh", ref, ref / "log.blockMesh")
    return ref


def prepare_mesh(x_surf, workdir, morph_radius=0.6):
    """Morph the reference mesh onto x_surf once; return the prepared polyMesh dir."""
    workdir = Path(workdir)
    mesh = workdir / "mesh"
    if (mesh / "constant/polyMesh/points").exists():
        return mesh
    workdir.mkdir(parents=True, exist_ok=True)
    ref = _reference_mesh(workdir.parent)
    shutil.copytree(TEMPLATE, mesh)
    shutil.rmtree(mesh / "constant/polyMesh", ignore_errors=True)
    shutil.copytree(ref / "constant/polyMesh", mesh / "constant/polyMesh")
    morph.morph_case(mesh, np.asarray(x_surf), radius=morph_radius)
    _foam("checkMesh -constant", mesh, mesh / "log.checkMesh")
    _foam("renumberMesh -overwrite -constant", mesh, mesh / "log.renumber")
    return mesh


def _set_flow_conditions(case, alpha_deg, reynolds):
    a = math.radians(alpha_deg)
    ux, uy = UINF * math.cos(a), UINF * math.sin(a)
    lx, ly = -math.sin(a), math.cos(a)
    nu = UINF * CHORD / reynolds

    u = (case / "0/U").read_text()
    u = re.sub(r"uniform \([^)]*\)", f"uniform ({ux:.10g} {uy:.10g} 0)", u)
    (case / "0/U").write_text(u)

    tp = (case / "constant/transportProperties").read_text()
    (case / "constant/transportProperties").write_text(
        re.sub(r"nu\s+\S+;", f"nu              {nu:.10g};", tp)
    )

    cd = (case / "system/controlDict").read_text()
    cd = cd.replace("liftDir         (0 1 0)", f"liftDir         ({lx:.10g} {ly:.10g} 0)")
    cd = cd.replace("dragDir         (1 0 0)", f"dragDir         ({ux:.10g} {uy:.10g} 0)")
    (case / "system/controlDict").write_text(cd)


def _parse_forces(case):
    dat = case / "postProcessing/forceCoeffs/0/coefficient.dat"
    rows = [ln.split() for ln in dat.read_text().splitlines() if ln and not ln.startswith("#")]
    last = rows[-1]
    return {"Cd": float(last[1]), "Cl": float(last[4]), "Cm": float(last[7])}


def _converged(case):
    log = (case / "log.simpleFoam").read_text()
    return "SIMPLE solution converged" in log


def solve(mesh_dir, alpha_deg, reynolds, workdir, restart_from=None):
    """One primal at a fixed angle of attack on the prepared mesh."""
    case = Path(workdir)
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(mesh_dir, case)
    if restart_from is not None:
        src = max((Path(restart_from) / "0").parent.glob("[0-9]*"), key=lambda p: float(p.name))
        for f in FIELDS:
            shutil.copy(src / f, case / "0" / f)

    _set_flow_conditions(case, alpha_deg, reynolds)
    _foam("simpleFoam", case, case / "log.simpleFoam")
    out = _parse_forces(case)
    out["converged"] = _converged(case)
    out["case"] = case
    return out


def run_trim(x_surf, cl_target, reynolds, workdir, alpha0=0.0, tol=2e-3, max_iter=6):
    """Newton/secant trim: iterate the angle of attack until Cl matches cl_target."""
    workdir = Path(workdir)
    mesh = prepare_mesh(x_surf, workdir)
    history = []
    alpha, prev = alpha0, None
    for it in range(max_iter):
        r = solve(mesh, alpha, reynolds, workdir / f"it{it}",
                  restart_from=prev["case"] if prev else None)
        history.append({"alpha_deg": alpha, "Cl": r["Cl"], "Cd": r["Cd"]})
        if abs(r["Cl"] - cl_target) <= tol:
            return {"Cd": r["Cd"], "Cl": r["Cl"], "Cm": r["Cm"], "alpha_deg": alpha,
                    "trim_iterations": it + 1, "converged": r["converged"], "history": history}
        if prev is None:
            alpha_next = alpha + (cl_target - r["Cl"]) / 0.1
        else:
            slope = (r["Cl"] - prev["Cl"]) / (alpha - prev["alpha_deg"])
            alpha_next = alpha - (r["Cl"] - cl_target) / slope
        prev = {"case": r["case"], "Cl": r["Cl"], "alpha_deg": alpha}
        alpha = alpha_next
    return {"Cd": r["Cd"], "Cl": r["Cl"], "Cm": r["Cm"], "alpha_deg": alpha,
            "trim_iterations": max_iter, "converged": False, "history": history}


def run_primal(x_surf, alpha_deg, reynolds, workdir, morph_radius=0.6):
    """Single primal at a fixed angle of attack (no trim)."""
    mesh = prepare_mesh(x_surf, workdir, morph_radius)
    r = solve(mesh, alpha_deg, reynolds, Path(workdir) / "case")
    return {"Cd": r["Cd"], "Cl": r["Cl"], "Cm": r["Cm"],
            "alpha_deg": alpha_deg, "converged": r["converged"]}
