# SPDX-License-Identifier: Apache-2.0
"""Drive an OpenFOAM incompressible RANS primal on the morphed reference mesh,
with an inner angle-of-attack Newton solve to trim to a target lift."""

import gzip
import json
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
UINF = 60.0
CHORD = 1.0
TURB_INIT = 25.0
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
    """Uncompress the vendored reference polyMesh once per cache root."""
    ref = Path(cache_root) / "reference"
    if not (ref / "constant/polyMesh/points").exists():
        shutil.rmtree(ref, ignore_errors=True)
        shutil.copytree(TEMPLATE / "constant/polyMesh", ref / "constant/polyMesh")
        for gz in (ref / "constant/polyMesh").glob("*.gz"):
            gz.with_suffix("").write_bytes(gzip.decompress(gz.read_bytes()))
            gz.unlink()
    return ref


def prepare_mesh(x_surf, workdir, morph_radius=0.6):
    """Morph the reference mesh onto x_surf once; return the prepared polyMesh dir."""
    workdir = Path(workdir)
    mesh = workdir / "mesh"
    if (mesh / "constant/polyMesh/points").exists():
        return mesh
    workdir.mkdir(parents=True, exist_ok=True)
    ref = _reference_mesh(workdir.parent)
    shutil.copytree(TEMPLATE, mesh,
                    ignore=shutil.ignore_patterns("adjoint", "polyMesh"))
    shutil.copytree(ref / "constant/polyMesh", mesh / "constant/polyMesh")
    morph.morph_case(mesh, np.asarray(x_surf), radius=morph_radius)
    _foam("checkMesh -constant", mesh, mesh / "log.checkMesh")
    _foam("renumberMesh -overwrite -constant", mesh, mesh / "log.renumber")
    return mesh


def _set_flow_conditions(case, alpha_deg, reynolds):
    a = math.radians(alpha_deg)
    dx, dy = math.cos(a), math.sin(a)
    lx, ly = -math.sin(a), math.cos(a)
    nu = UINF * CHORD / reynolds

    u = (case / "0/U").read_text()
    u = re.sub(r"(internalField|freestreamValue)(\s+)uniform \([^)]*\)",
               rf"\1\2uniform ({UINF * dx:.10g} {UINF * dy:.10g} 0)", u)
    (case / "0/U").write_text(u)

    for field in ("nuTilda", "nut"):
        f = case / "0" / field
        f.write_text(re.sub(r"uniform 2\.5e-0?4", f"uniform {TURB_INIT * nu:.10g}",
                            f.read_text()))

    tp = (case / "constant/transportProperties").read_text()
    (case / "constant/transportProperties").write_text(
        re.sub(r"nu\s+\S+;", f"nu              {nu:.10g};", tp)
    )

    cd = (case / "system/controlDict").read_text()
    cd = cd.replace("liftDir         (0 1 0)", f"liftDir         ({lx:.10g} {ly:.10g} 0)")
    cd = cd.replace("dragDir         (1 0 0)", f"dragDir         ({dx:.10g} {dy:.10g} 0)")
    (case / "system/controlDict").write_text(cd)


def _coefficient_rows(case):
    dat = case / "postProcessing/forceCoeffs/0/coefficient.dat"
    return [ln.split() for ln in dat.read_text().splitlines() if ln and not ln.startswith("#")]


def _parse_forces(case):
    last = _coefficient_rows(case)[-1]
    return {"Cd": float(last[1]), "Cl": float(last[4]), "Cm": float(last[7])}


def _converged(case, window=200, tol=1e-4):
    """Accept a solve whose forces have stopped moving, not only one that tripped
    residualControl.

    A section far from the reference can hold Cd and Cl steady to six figures for
    hundreds of iterations while the residuals sit just above the threshold.
    Rejecting those loses good designs; what the objective needs is that the
    coefficients are stationary.
    """
    if "SIMPLE solution converged" in (case / "log.simpleFoam").read_text():
        return True
    rows = _coefficient_rows(case)
    if len(rows) < window:
        return False
    tail = np.array([[float(r[1]), float(r[4])] for r in rows[-window:]])
    scale = np.abs(tail).mean(axis=0) + 1e-12
    return bool(np.all(np.ptp(tail, axis=0) / scale < tol))


def _case_ignore(_dir, names):
    return [n for n in names if n == "postProcessing" or (n.isdigit() and n != "0")]


def solve(mesh_dir, alpha_deg, reynolds, workdir, restart_from=None):
    """One primal at a fixed angle of attack on the prepared mesh."""
    case = Path(workdir)
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(mesh_dir, case, ignore=_case_ignore)
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


class TrimFailure(RuntimeError):
    """The trim never reached the target lift on a converged primal.

    Raised rather than returned: a diverged primal produces a plausible-looking
    Cd and an adjoint built on top of it, and nothing downstream can tell the
    difference from a real one.
    """


TRIM_CACHE = "trim.json"


def _read_trim_cache(workdir, cl_target):
    path = Path(workdir) / TRIM_CACHE
    if not path.exists():
        return None
    cached = json.loads(path.read_text())
    if cached.get("cl_target") != cl_target or not Path(cached["case"]).exists():
        return None
    return cached


def _write_trim_cache(workdir, cl_target, result):
    (Path(workdir) / TRIM_CACHE).write_text(json.dumps({**result, "cl_target": cl_target}))
    return result


def run_trim(x_surf, cl_target, reynolds, workdir, alpha0=0.0, tol=2e-3, max_iter=6):
    """Newton/secant trim: iterate the angle of attack until Cl matches cl_target.

    The converged result is cached beside the run, since apply and the vjp are
    separate endpoint calls on the same design and would otherwise each pay for
    a full trim.
    """
    workdir = Path(workdir)
    cached = _read_trim_cache(workdir, cl_target)
    if cached is not None:
        return cached
    mesh = prepare_mesh(x_surf, workdir)
    history = []
    alpha, prev = alpha0, None
    for it in range(max_iter):
        r = solve(mesh, alpha, reynolds, workdir / f"it{it}",
                  restart_from=prev["case"] if prev else None)
        history.append({"alpha_deg": alpha, "Cl": r["Cl"], "Cd": r["Cd"]})
        if abs(r["Cl"] - cl_target) <= tol and r["converged"]:
            return _write_trim_cache(workdir, cl_target, {
                "Cd": r["Cd"], "Cl": r["Cl"], "Cm": r["Cm"], "alpha_deg": alpha,
                "trim_iterations": it + 1, "converged": r["converged"],
                "case": str(r["case"]), "history": history})
        if prev is None:
            alpha_next = alpha + (cl_target - r["Cl"]) / 0.1
        else:
            slope = (r["Cl"] - prev["Cl"]) / (alpha - prev["alpha_deg"])
            alpha_next = alpha - (r["Cl"] - cl_target) / slope
        prev = {"case": r["case"], "Cl": r["Cl"], "alpha_deg": alpha}
        alpha = alpha_next
    raise TrimFailure(
        f"no converged primal at Cl={cl_target} in {max_iter} iterations "
        f"(last Cl={r['Cl']:.4f}, converged={r['converged']}); history={history}"
    )


def run_primal(x_surf, alpha_deg, reynolds, workdir, morph_radius=0.6):
    """Single primal at a fixed angle of attack (no trim)."""
    mesh = prepare_mesh(x_surf, workdir, morph_radius)
    r = solve(mesh, alpha_deg, reynolds, Path(workdir) / "case")
    return {"Cd": r["Cd"], "Cl": r["Cl"], "Cm": r["Cm"],
            "alpha_deg": alpha_deg, "converged": r["converged"]}


ADJOINT_OVERLAY = TEMPLATE / "adjoint"
RESTART_FIELDS = ("U", "p", "phi", "nut", "nuTilda")


def _dir_tuple(vec):
    return f"({vec[0]:.12g} {vec[1]:.12g} 0)"


def _read_point_vector(path):
    """Parse a pointVectorField into (nPoints, 2), dropping the empty direction."""
    body = morph._foam_list(Path(path).read_text().split("internalField", 1)[1])
    rows = [ln.strip()[1:-1].split() for ln in body.splitlines() if ln.strip().startswith("(")]
    return np.array([[float(r[0]), float(r[1])] for r in rows])


def _adjoint_one(primal_case, obj_name, obj_dir, reynolds, case,
                 primal_iters=400, adj_iters=2000):
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(primal_case, case, ignore=shutil.ignore_patterns("adjoint", "postProcessing"))
    latest = max(case.glob("[0-9]*"), key=lambda p: float(p.name))
    for f in RESTART_FIELDS:
        if (latest / f).exists() and latest.name != "0":
            shutil.copy(latest / f, case / "0" / f)
    if latest.name != "0":
        shutil.rmtree(latest)

    for sub in ("0", "constant", "system"):
        for f in (ADJOINT_OVERLAY / sub).iterdir():
            shutil.copy(f, case / sub / f.name)

    opt = (case / "system/optimisationDict").read_text()
    opt = (opt.replace("__OBJ__", obj_name).replace("__OBJ_DIR__", _dir_tuple(obj_dir))
           .replace("__PRIMAL_ITERS__", str(primal_iters))
           .replace("__ADJ_ITERS__", str(adj_iters)))
    (case / "system/optimisationDict").write_text(opt)

    tp = (case / "constant/transportProperties").read_text()
    (case / "constant/transportProperties").write_text(
        re.sub(r"nu\s+\S+;", f"nu              {UINF * CHORD / reynolds:.10g};", tp)
    )
    _foam("adjointOptimisationFoam", case, case / "log.adjoint")

    sens = sorted(case.glob("[0-9]*/pointSensVecadj*"))
    if not sens:
        raise RuntimeError(f"no point sensitivity written (see {case / 'log.adjoint'})")
    return _read_point_vector(sens[-1])


def run_adjoint(primal_case, alpha_deg, reynolds, workdir, adj_iters=3000, primal_iters=200):
    """Drag and lift shape sensitivities dJ/dx per mesh point, from two
    single-objective adjointOptimisationFoam runs restarted off primal_case."""
    a = math.radians(alpha_deg)
    workdir = Path(workdir)
    return {
        "drag": _adjoint_one(Path(primal_case), "drag", (math.cos(a), math.sin(a)),
                             reynolds, workdir / "drag", primal_iters, adj_iters),
        "lift": _adjoint_one(Path(primal_case), "lift", (-math.sin(a), math.cos(a)),
                             reynolds, workdir / "lift", primal_iters, adj_iters),
    }


def alpha_derivatives(primal_case, alpha_deg, reynolds, workdir, dalpha=0.25):
    """Central-difference dCd/dalpha and dCl/dalpha (per degree), warm-restarted."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    plus = solve(primal_case, alpha_deg + dalpha, reynolds, workdir / "plus",
                 restart_from=primal_case)
    minus = solve(primal_case, alpha_deg - dalpha, reynolds, workdir / "minus",
                  restart_from=primal_case)
    return {"dCd_da": (plus["Cd"] - minus["Cd"]) / (2 * dalpha),
            "dCl_da": (plus["Cl"] - minus["Cl"]) / (2 * dalpha)}
