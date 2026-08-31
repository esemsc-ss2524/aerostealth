# SPDX-License-Identifier: Apache-2.0
"""Epsilon-constraint sweep: min Cd s.t. sigma_agg <= eps_k, Cl >= Cl*, g_geom <= 0.

The angle of attack is fixed and lift is a genuine constraint carried by the
optimizer, so every gradient in the problem is an adjoint and nothing is
finite-differenced.
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
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

    def __init__(self, config, out_dir=None, tag="sweep"):
        self.config = config
        self.tag = tag
        self.aero = build_forward(config, aero=True)
        self.cheap = build_forward(config, aero=False)
        self.cd_ref, self.sigma_ref = references(config)
        self.cl_target = float(config["aero"]["cl_target"])
        self.geom_scale = _geom_scales(config)
        self.cache = {}
        self.n_aero_calls = 0
        self.trace = (Path(out_dir) / f"trajectory_{tag}.jsonl") if out_dir else None

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

    def __call__(self, theta, grad=True):
        """grad False skips the adjoints: a point being reported rather than
        stepped from needs the primal only, and some valid shapes differentiate
        badly enough to fail."""
        key = np.asarray(theta, dtype=np.float64).tobytes()
        entry = self.cache.get(key)
        if entry is not None and (entry["dobj"] is not None or not grad):
            return entry

        sigma, dsigma = self.sigma_and_grad(theta)
        g_geom = np.asarray(self.geom_constraints(theta)) / self.geom_scale

        self.n_aero_calls += 1
        try:
            if grad:
                cd, dcd = self.cd_and_grad(theta)
                cl, dcl = self.cl_and_grad(theta)
            else:
                out = self.aero(theta)
                cd, cl, dcd, dcl = out["Cd"], out["Cl"], None, None
        except Exception as exc:
            print(f"  [{self.tag}] eval {self.n_aero_calls:3d}  AERO FAILED: "
                  f"{str(exc).splitlines()[-1][:140]}", flush=True)
            raise AeroFailure(str(exc)) from exc

        entry = {
            "obj": float(cd) / self.cd_ref,
            "dobj": None if dcd is None else np.asarray(dcd) / self.cd_ref,
            "Cd": float(cd),
            "Cl": float(cl),
            "sigma_agg": float(sigma),
            "g": np.concatenate([
                [float(sigma) / self.sigma_ref],
                [1.0 - float(cl) / self.cl_target],
                g_geom,
            ]),
            "dg": None if dcl is None else np.vstack([
                np.asarray(dsigma) / self.sigma_ref,
                -np.asarray(dcl) / self.cl_target,
                np.asarray(self.geom_jac(theta)) / self.geom_scale[:, None],
            ]),
        }
        self.cache[key] = entry
        print(f"  [{self.tag}] eval {self.n_aero_calls:3d}  Cd={entry['Cd']:.6f}  "
              f"Cl={entry['Cl']:.4f}  sigma={entry['sigma_agg']:.6f}  "
              f"gmax={entry['g'][1:].max():+.4f}", flush=True)
        self._record(theta, entry)
        return entry

    def n_constraints(self, eps):
        return len(self.geom_scale) + (2 if eps is not None else 1)

    def _record(self, theta, entry):
        if self.trace is None:
            return
        x_surf, polar = self._polar(theta)
        row = {
            "eval": self.n_aero_calls,
            "level": self.tag,
            "theta": np.asarray(theta, dtype=np.float64).tolist(),
            "Cd": entry["Cd"], "Cl": entry["Cl"], "sigma_agg": entry["sigma_agg"],
            "sigma_by_angle": polar.tolist(),
            "g": entry["g"].tolist(),
            "dCd_dtheta": None if entry["dobj"] is None else entry["dobj"].tolist(),
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
    m = ev.n_constraints(eps)

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


def solve_subproblem(ev, config, sweep, eps, x0, retreat_to=None):
    """One epsilon level, warm-started from x0.

    A design the aero leg will not solve or differentiate means the move limit
    was too generous, so the level restarts on a tighter one rather than feeding
    the optimizer a fabricated gradient. If the start itself will not
    differentiate, it is also pulled toward retreat_to, a design known to solve.
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
            x, status = _mma(ev, opt_cfg, eps, centre, low, high)
            return {**evaluate_point(ev, x), "status": status, "step_bound": step}
        except AeroFailure:
            step = step * 0.5 if step > 0 else 0.02
            if retreat_to is not None:
                centre = 0.5 * (centre + np.asarray(retreat_to, dtype=np.float64))
            print(f"  [{ev.tag}] retry {attempt + 1}: step_bound {step:.4f}"
                  f"{' , start pulled toward a solvable design' if retreat_to is not None else ''}",
                  flush=True)
    return {**evaluate_point(ev, centre), "status": -99, "step_bound": step}


def evaluate_point(ev, theta):
    """Objectives and constraints at one design, no optimization and no adjoint."""
    r = ev(jnp.asarray(theta), grad=False)
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


def _save(points, ev, n_calls, out_path):
    result = {"points": points, "n_aero_calls": n_calls,
              "cd_ref": ev.cd_ref, "sigma_ref": ev.sigma_ref,
              "cl_target": ev.cl_target,
              "alpha_deg": float(ev.config["aero"]["alpha_deg"])}
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def _anchor_blend(eps, hi, lo, theta_aero, theta_stealth, lb, ub):
    """Warm start each level from where its target sits between the anchors.

    Marching level to level leaves the low-echo-width end starting far from its
    own solution, and those levels are the ones the stealth anchor already
    dominates.
    """
    t = 0.0 if hi <= lo else float(np.clip((hi - eps) / (hi - lo), 0.0, 1.0))
    return np.clip((1.0 - t) * theta_aero + t * theta_stealth, lb, ub)


def _level_worker(item):
    """One epsilon level in its own process. Module level so it pickles."""
    i, eps, x0, config, sweep, out_dir, theta_aero = item
    ev = Evaluator(config, out_dir=out_dir, tag=f"eps{i}")
    p = solve_subproblem(ev, config, sweep, eps, x0, retreat_to=theta_aero)
    return i, {**p, "eps": eps, "label": "sweep"}, ev.n_aero_calls


def run_sweep(config, sweep, out_path=None):
    out_dir = Path(out_path).parent if out_path else None
    anchor_ev = Evaluator(config, out_dir=out_dir, tag="anchor")
    theta0 = baseline_theta(config)

    aero = solve_subproblem(anchor_ev, config, sweep, None, theta0)
    stealth = evaluate_point(anchor_ev, stealth_anchor(anchor_ev, config, sweep, theta0))
    print(f"  anchors: aero Cd={aero['Cd']:.6f} sigma={aero['sigma_agg']:.6f} | "
          f"stealth Cd={stealth['Cd']:.6f} sigma={stealth['sigma_agg']:.6f}", flush=True)

    hi = aero["sigma_agg"] / anchor_ev.sigma_ref
    lo = stealth["sigma_agg"] / anchor_ev.sigma_ref
    levels = np.linspace(hi, lo, int(sweep["epsilon"]["points"]))[1:-1]
    lb, ub = _bounds(config)
    theta_aero, theta_stealth = np.asarray(aero["theta"]), np.asarray(stealth["theta"])

    jobs = int(sweep["optimizer"].get("jobs", len(levels)))
    work = [
        (i, float(eps), _anchor_blend(eps, hi, lo, theta_aero, theta_stealth, lb, ub),
         config, sweep, out_dir, theta_aero)
        for i, eps in enumerate(levels)
    ]
    done, n_calls = {}, anchor_ev.n_aero_calls
    # Separate processes, not threads: concurrent apply_tesseract calls from
    # several threads of one interpreter abort the process without a traceback.
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(_level_worker, item) for item in work]
        for future in as_completed(futures):
            i, p, calls = future.result()
            done[i], n_calls = p, n_calls + calls
            print(f"  level {i} done  eps={p['eps']:.4f}  Cd={p['Cd']:.6f}  "
                  f"Cl={p['Cl']:.4f}  sigma={p['sigma_agg']:.6f}  status={p['status']}",
                  flush=True)
            points = ([dict(aero, eps=None, label="aero_anchor")]
                      + [done[k] for k in sorted(done)]
                      + [dict(stealth, eps=None, label="stealth_anchor", status=0)])
            _save(points, anchor_ev, n_calls, out_path)

    points = ([dict(aero, eps=None, label="aero_anchor")]
              + [done[k] for k in sorted(done)]
              + [dict(stealth, eps=None, label="stealth_anchor", status=0)])
    return _save(points, anchor_ev, n_calls, out_path)


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
