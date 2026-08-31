# SPDX-License-Identifier: Apache-2.0
"""Epsilon-constraint sweep: min Cd s.t. sigma_agg <= eps_k, Cl >= Cl*, g_geom <= 0.

The angle of attack is fixed and lift is a genuine constraint carried by the
optimizer, so every gradient in the problem is an adjoint and nothing is
finite-differenced.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import nlopt
import numpy as np

from driver.forward import baseline_theta, build_forward, em_inputs, geom_inputs
from driver.objectives import references
from driver.tesseracts import load_config, local_tesseract

jax.config.update("jax_enable_x64", True)

GEOM_KEYS = ("thickness_min", "thickness_max", "trailing_edge_gap_max", "enclosed_area_min")


class AeroFailure(RuntimeError):
    """The aero leg would not solve or differentiate at this design."""


def _geom_scales(config):
    c = config["geometry"]["constraints"]
    return np.array([abs(float(c[k])) for k in GEOM_KEYS])


def _bounds(config):
    lo, hi = config["geometry"]["cst_weight_bounds"]
    n = int(config["geometry"]["cst_coeffs"]) // 2
    return (np.concatenate([np.full(n, lo), np.full(n, -hi)]),
            np.concatenate([np.full(n, hi), np.full(n, -lo)]))


class Evaluator:
    """Values and gradients for one design, memoized across nlopt callbacks.

    Cd and Cl share one primal and one pair of adjoint runs, so asking for both
    costs what asking for either would. The echo width and the geometric
    constraints come from the JAX-only leg and never touch OpenFOAM.
    """

    def __init__(self, config, out_dir=None):
        self.config = config
        self.aero = build_forward(config, aero=True)
        self.cheap = build_forward(config, aero=False)
        self.cd_ref, self.sigma_ref = references(config)
        self.cl_target = float(config["aero"]["cl_target"])
        self.geom_scale = _geom_scales(config)
        self.cache = {}
        self.n_aero_calls = 0
        self.trace = (Path(out_dir) / "trajectory.jsonl") if out_dir else None
        self.level = None

        self.cd_and_grad = jax.value_and_grad(lambda t: self.aero(t)["Cd"])
        self.cl_and_grad = jax.value_and_grad(lambda t: self.aero(t)["Cl"])
        self.sigma_and_grad = jax.value_and_grad(lambda t: self.cheap(t)["sigma_agg"])
        self.geom_constraints = lambda t: self.cheap(t)["g_geom"]
        self.geom_jac = jax.jacrev(self.geom_constraints)

        self.geom = local_tesseract("geom")
        self.em = local_tesseract("em")
        self.geom_in, self.em_in = geom_inputs(config), em_inputs(config)

    def _polar(self, theta):
        """RCS by angle and the surface curve, recorded for the animation."""
        x_surf = self.geom.apply({"theta": np.asarray(theta), **self.geom_in})["x_surf"]
        out = self.em.apply({"x_surf": x_surf, **self.em_in})
        return np.asarray(x_surf), np.asarray(out["sigma_by_angle"])

    def __call__(self, theta):
        key = np.asarray(theta, dtype=np.float64).tobytes()
        if key in self.cache:
            return self.cache[key]

        sigma, dsigma = self.sigma_and_grad(theta)
        g_geom = np.asarray(self.geom_constraints(theta)) / self.geom_scale
        dg_geom = np.asarray(self.geom_jac(theta)) / self.geom_scale[:, None]

        self.n_aero_calls += 1
        try:
            cd, dcd = self.cd_and_grad(theta)
            cl, dcl = self.cl_and_grad(theta)
        except Exception as exc:
            print(f"    eval {self.n_aero_calls:3d}  AERO FAILED: "
                  f"{str(exc).splitlines()[-1][:140]}", flush=True)
            raise AeroFailure(str(exc)) from exc

        entry = {
            "obj": float(cd) / self.cd_ref,
            "dobj": np.asarray(dcd) / self.cd_ref,
            "Cd": float(cd),
            "Cl": float(cl),
            "sigma_agg": float(sigma),
            "g": np.concatenate([
                [float(sigma) / self.sigma_ref],
                [1.0 - float(cl) / self.cl_target],
                g_geom,
            ]),
            "dg": np.vstack([
                np.asarray(dsigma) / self.sigma_ref,
                -np.asarray(dcl) / self.cl_target,
                dg_geom,
            ]),
        }
        self.cache[key] = entry
        print(f"    eval {self.n_aero_calls:3d}  Cd={entry['Cd']:.6f}  Cl={entry['Cl']:.4f}  "
              f"sigma={entry['sigma_agg']:.6f}  gmax={entry['g'][1:].max():+.4f}", flush=True)
        self._record(theta, entry)
        return entry

    def _record(self, theta, entry):
        if self.trace is None:
            return
        x_surf, polar = self._polar(theta)
        row = {
            "eval": self.n_aero_calls,
            "level": self.level,
            "theta": np.asarray(theta, dtype=np.float64).tolist(),
            "Cd": entry["Cd"], "Cl": entry["Cl"], "sigma_agg": entry["sigma_agg"],
            "sigma_by_angle": polar.tolist(),
            "g": entry["g"].tolist(),
            "dCd_dtheta": entry["dobj"].tolist(),
            "x_surf": x_surf.tolist(),
            "case": self._case_dir(x_surf),
        }
        with open(self.trace, "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def _case_dir(self, x_surf):
        """The OpenFOAM run directory this design landed in, for later replay."""
        a = float(self.config["aero"]["alpha_deg"])
        re = float(self.config["aero"]["reynolds"])
        key = hashlib.sha1(
            np.ascontiguousarray(x_surf, dtype=np.float64).tobytes()
            + np.float64([a, re]).tobytes()
        ).hexdigest()[:16]
        return str(Path("tesseracts/cfd/_run") / key)


def _rows(ev, entry, eps):
    """Constraint rows in g <= 0 form; eps None drops the echo-width row."""
    rows, jac = entry["g"].copy(), entry["dg"]
    if eps is None:
        return rows[1:], jac[1:]
    rows[0] -= eps
    return rows, jac


def _mma(ev, opt_cfg, eps, x0, lb, ub):
    n = len(x0)
    m = len(_rows(ev, ev(jnp.asarray(x0)), eps)[0])

    def objective(x, grad):
        r = ev(jnp.asarray(x))
        if grad.size:
            grad[:] = r["dobj"]
        return r["obj"]

    def constraints(result, x, grad):
        rows, jac = _rows(ev, ev(jnp.asarray(x)), eps)
        result[:] = rows
        if grad.size:
            grad[:] = jac

    opt = nlopt.opt(nlopt.LD_MMA, n)
    opt.set_min_objective(objective)
    opt.add_inequality_mconstraint(constraints, [1e-6] * m)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(int(opt_cfg["max_iter"]))
    opt.set_ftol_rel(float(opt_cfg["ftol"]))
    x = opt.optimize(np.asarray(x0, dtype=np.float64))
    return x, int(opt.last_optimize_result())


def solve_subproblem(ev, config, sweep, eps, x0):
    """One epsilon level, warm-started from x0.

    A design the RANS solver will not converge means the move limit was too
    generous, so the level restarts on a tighter one rather than feeding the
    optimizer a fabricated gradient.
    """
    opt_cfg = sweep["optimizer"]
    lb, ub = _bounds(config)
    step = float(opt_cfg.get("step_bound", 0.0))
    centre = np.asarray(x0, dtype=np.float64)

    for attempt in range(int(opt_cfg.get("retries", 2)) + 1):
        low, high = lb, ub
        if step > 0:
            low, high = np.maximum(lb, centre - step), np.minimum(ub, centre + step)
        try:
            x, status = _mma(ev, opt_cfg, eps, x0, low, high)
            return {**evaluate_point(ev, x), "status": status, "step_bound": step}
        except AeroFailure:
            step = step * 0.5 if step > 0 else 0.02
            print(f"  level eps={eps}: retry {attempt + 1} with step_bound {step:.4f}",
                  flush=True)
    return {**evaluate_point(ev, centre), "status": -99, "step_bound": step}


def evaluate_point(ev, theta):
    """Objectives and constraints at one design, no optimization."""
    r = ev(jnp.asarray(theta))
    return {
        "theta": np.asarray(theta, dtype=np.float64).tolist(),
        "Cd": r["Cd"],
        "Cl": r["Cl"],
        "sigma_agg": r["sigma_agg"],
        "g": r["g"].tolist(),
    }


def stealth_anchor(ev, config, sweep, x0):
    """Minimize the echo width subject to geometry only, for the low end of the sweep."""
    lb, ub = _bounds(config)

    def objective(x, grad):
        val, g = ev.sigma_and_grad(jnp.asarray(x))
        if grad.size:
            grad[:] = np.asarray(g)
        return float(val)

    def constraints(result, x, grad):
        t = jnp.asarray(x)
        result[:] = np.asarray(ev.geom_constraints(t)) / ev.geom_scale
        if grad.size:
            grad[:] = np.asarray(ev.geom_jac(t)) / ev.geom_scale[:, None]

    opt = nlopt.opt(nlopt.LD_MMA, len(x0))
    opt.set_min_objective(objective)
    opt.add_inequality_mconstraint(constraints, [1e-6] * len(ev.geom_scale))
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(int(sweep["optimizer"]["anchor_max_iter"]))
    opt.set_ftol_rel(float(sweep["optimizer"]["ftol"]))
    return opt.optimize(np.asarray(x0, dtype=np.float64))


def _save(points, ev, out_path):
    result = {"points": points, "n_aero_calls": ev.n_aero_calls,
              "cd_ref": ev.cd_ref, "sigma_ref": ev.sigma_ref,
              "cl_target": ev.cl_target,
              "alpha_deg": float(ev.config["aero"]["alpha_deg"])}
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def run_sweep(config, sweep, out_path=None):
    out_dir = Path(out_path).parent if out_path else None
    ev = Evaluator(config, out_dir=out_dir)
    theta0 = baseline_theta(config)

    ev.level = "aero_anchor"
    aero = solve_subproblem(ev, config, sweep, None, theta0)
    ev.level = "stealth_anchor"
    stealth = evaluate_point(ev, stealth_anchor(ev, config, sweep, theta0))

    hi = aero["sigma_agg"] / ev.sigma_ref
    lo = stealth["sigma_agg"] / ev.sigma_ref
    levels = np.linspace(hi, lo, int(sweep["epsilon"]["points"]))[1:-1]

    points = [dict(aero, eps=None, label="aero_anchor")]
    _save(points, ev, out_path)
    x = np.asarray(aero["theta"])
    for eps in levels:
        ev.level = float(eps)
        p = solve_subproblem(ev, config, sweep, float(eps), jnp.asarray(x))
        p["eps"] = float(eps)
        p["label"] = "sweep"
        points.append(p)
        _save(points, ev, out_path)
        print(f"  eps={eps:.4f}  Cd={p['Cd']:.6f}  Cl={p['Cl']:.4f}  "
              f"sigma={p['sigma_agg']:.6f}  status={p['status']}", flush=True)
        x = np.asarray(p["theta"])
    points.append(dict(stealth, eps=None, label="stealth_anchor", status=0))
    return _save(points, ev, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default="configs/baseline.yaml")
    parser.add_argument("--sweep", type=Path, default="configs/sweep.yaml")
    parser.add_argument("--out", type=Path, default="outputs/pareto.json")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = run_sweep(load_config(args.baseline), load_config(args.sweep), args.out)
    for p in result["points"]:
        print(f"{p['label']:14s} Cd={p['Cd']:.6f}  Cl={p['Cl']:.4f}  "
              f"sigma={p['sigma_agg']:.6f}  status={p['status']}")
    print(f"wrote {args.out}  ({result['n_aero_calls']} aero gradient calls)")


if __name__ == "__main__":
    main()
