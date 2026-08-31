# SPDX-License-Identifier: Apache-2.0
"""Epsilon-constraint sweep: min Cd s.t. sigma_agg <= eps_k, Cl = Cl*, g_geom <= 0.

`Cl = Cl*` is not a constraint here: the cfd tesseract trims to it internally and
reports the gradient at fixed lift, so the outer problem only carries the echo
width and the geometric constraints.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import nlopt
import numpy as np

from driver.forward import baseline_theta, build_forward
from driver.objectives import references
from driver.tesseracts import load_config

jax.config.update("jax_enable_x64", True)

# Normalized objective returned when the aero leg will not solve. Large enough
# that any real design beats it, finite so SLSQP keeps its line search.
FAILED_OBJECTIVE = 100.0


class Evaluator:
    """Values and gradients for one design vector, memoized across nlopt callbacks.

    nlopt asks for the objective and the constraints separately at the same point.
    Only Cd needs OpenFOAM, so the aero leg is differentiated on its own and the
    cheap leg supplies the constraint jacobian; nothing runs a wasted adjoint.
    """

    def __init__(self, config):
        self.aero = build_forward(config, aero=True)
        self.cheap = build_forward(config, aero=False)
        self.cd_ref, self.sigma_ref = references(config)
        self.cache = {}
        self.n_aero_calls = 0
        self.n_failures = 0
        self.last_good = None
        # Built once: a transform defined inside an nlopt callback retraces on
        # every call, which dominates the cheap leg.
        self.cd_and_grad = jax.value_and_grad(lambda t: self.aero(t)["Cd"])
        self.constraints = self._constraint_vec
        self.constraints_jac = jax.jacrev(self._constraint_vec)
        self.sigma_and_grad = jax.value_and_grad(lambda t: self._constraint_vec(t)[0])
        self.geom_constraints = lambda t: self.cheap(t)["g_geom"]
        self.geom_jac = jax.jacrev(self.geom_constraints)

    def _constraint_vec(self, theta):
        out = self.cheap(theta)
        return jnp.concatenate([
            jnp.atleast_1d(out["sigma_agg"] / self.sigma_ref),
            out["g_geom"],
        ])

    def __call__(self, theta):
        key = np.asarray(theta, dtype=np.float64).tobytes()
        if key in self.cache:
            return self.cache[key]

        # The cheap leg is defined for any shape, so constraints are always real
        # even when the aero leg fails.
        g = np.asarray(self.constraints(theta))
        dg = np.asarray(self.constraints_jac(theta))
        self.n_aero_calls += 1
        try:
            cd, dcd = self.cd_and_grad(theta)
            entry = {"Cd": float(cd), "obj": float(cd) / self.cd_ref,
                     "dobj": np.asarray(dcd) / self.cd_ref, "failed": False}
            self.last_good = np.asarray(theta, dtype=np.float64)
        except Exception as exc:
            self.n_failures += 1
            entry = {"Cd": float("nan"), "obj": FAILED_OBJECTIVE,
                     "dobj": self._retreat(theta), "failed": True}
            print(f"    eval {self.n_aero_calls:3d}  AERO FAILED: "
                  f"{str(exc).splitlines()[0][:140]}", flush=True)

        entry.update(g=g, dg=dg)
        self.cache[key] = entry
        if not entry["failed"]:
            print(f"    eval {self.n_aero_calls:3d}  Cd={entry['Cd']:.6f}  "
                  f"sigma={g[0] * self.sigma_ref:.6f}  "
                  f"area_slack={-g[4]:+.5f}", flush=True)
        return entry

    def _retreat(self, theta):
        """Gradient for a failed design: uphill away from the last good one, so
        the line search walks back instead of trusting a diverged solve."""
        ref = self.last_good if self.last_good is not None else np.asarray(theta)
        direction = np.asarray(theta, dtype=np.float64) - ref
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return np.zeros_like(direction)
        return FAILED_OBJECTIVE * direction / norm


def _bounds(config):
    lo, hi = config["geometry"]["cst_weight_bounds"]
    n = int(config["geometry"]["cst_coeffs"]) // 2
    return (np.concatenate([np.full(n, lo), np.full(n, -hi)]),
            np.concatenate([np.full(n, hi), np.full(n, -lo)]))


def solve_subproblem(ev, config, sweep, eps, x0):
    """One epsilon level, warm-started from x0. eps None drops the echo-width row."""
    opt_cfg = sweep["optimizer"]
    lb, ub = _bounds(config)
    n = len(x0)
    history = []

    # Trust region. Without it SLSQP can cross the whole design box in one step
    # and land on a section the RANS solver will not converge.
    step = float(opt_cfg.get("step_bound", 0.0))
    if step > 0:
        centre = np.asarray(x0, dtype=np.float64)
        lb, ub = np.maximum(lb, centre - step), np.minimum(ub, centre + step)

    def objective(x, grad):
        r = ev(jnp.asarray(x))
        if grad.size:
            grad[:] = r["dobj"]
        history.append({"Cd": r["Cd"], "sigma": float(r["g"][0]) * ev.sigma_ref})
        return r["obj"]

    def constraints(result, x, grad):
        r = ev(jnp.asarray(x))
        rows, jac = r["g"].copy(), r["dg"]
        if eps is None:
            rows, jac = rows[1:], jac[1:]
        else:
            rows[0] -= eps
        result[:] = rows
        if grad.size:
            grad[:] = jac

    m = len(ev(jnp.asarray(x0))["g"])
    m_active = m if eps is not None else m - 1
    opt = nlopt.opt(nlopt.LD_SLSQP, n)
    opt.set_min_objective(objective)
    opt.add_inequality_mconstraint(constraints, [1e-6] * m_active)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(int(opt_cfg["max_iter"]))
    opt.set_ftol_rel(float(opt_cfg["ftol"]))
    try:
        x = opt.optimize(np.asarray(x0, dtype=np.float64))
        status = int(opt.last_optimize_result())
    except Exception as exc:
        # A level that stalls should not cost the rest of the sweep; keep the
        # best point nlopt reached and record why it stopped.
        x, status = np.asarray(x0, dtype=np.float64), -99
        print(f"  level eps={eps} stopped early: {exc}")
    return {**evaluate_point(ev, x), "status": status, "n_eval": len(history)}


def evaluate_point(ev, theta):
    """Objectives and constraints at one design, no optimization."""
    r = ev(jnp.asarray(theta))
    return {
        "theta": np.asarray(theta, dtype=np.float64).tolist(),
        "Cd": r["Cd"],
        "sigma_agg": float(r["g"][0]) * ev.sigma_ref,
        "g_geom": r["g"][1:].tolist(),
    }


def stealth_anchor(ev, config, sweep, x0):
    """Minimize the echo width subject to geometry only, for the low end of the sweep."""
    lb, ub = _bounds(config)
    n = len(x0)

    def objective(x, grad):
        val, g = ev.sigma_and_grad(jnp.asarray(x))
        if grad.size:
            grad[:] = np.asarray(g)
        return float(val)

    def constraints(result, x, grad):
        th = jnp.asarray(x)
        result[:] = np.asarray(ev.geom_constraints(th))
        if grad.size:
            grad[:] = np.asarray(ev.geom_jac(th))

    opt = nlopt.opt(nlopt.LD_SLSQP, n)
    opt.set_min_objective(objective)
    opt.add_inequality_mconstraint(constraints, [1e-6] * 4)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(int(sweep["optimizer"]["max_iter"]))
    opt.set_ftol_rel(float(sweep["optimizer"]["ftol"]))
    return opt.optimize(np.asarray(x0, dtype=np.float64))


def _save(points, ev, out_path):
    result = {"points": points, "n_aero_calls": ev.n_aero_calls,
              "cd_ref": ev.cd_ref, "sigma_ref": ev.sigma_ref}
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def run_sweep(config, sweep, out_path=None):
    ev = Evaluator(config)
    theta0 = baseline_theta(config)

    aero = solve_subproblem(ev, config, sweep, None, theta0)
    # The low end is the shape that minimizes echo width; its drag is whatever
    # it is. Re-optimizing Cd from there with no echo-width constraint would
    # just walk back to the aero anchor.
    stealth = evaluate_point(ev, stealth_anchor(ev, config, sweep, theta0))

    hi = aero["sigma_agg"] / ev.sigma_ref
    lo = stealth["sigma_agg"] / ev.sigma_ref
    levels = np.linspace(hi, lo, int(sweep["epsilon"]["points"]))[1:-1]

    points = [dict(aero, eps=None, label="aero_anchor")]
    _save(points, ev, out_path)
    x = np.asarray(aero["theta"])
    for eps in levels:
        p = solve_subproblem(ev, config, sweep, float(eps), jnp.asarray(x))
        p["eps"] = float(eps)
        p["label"] = "sweep"
        points.append(p)
        # Written after every level: a sweep that dies at hour two still leaves
        # the points it earned.
        _save(points, ev, out_path)
        print(f"  eps={eps:.4f}  Cd={p['Cd']:.6f}  sigma={p['sigma_agg']:.6f}  "
              f"status={p['status']}  evals={p['n_eval']}", flush=True)
        x = np.asarray(p["theta"])
    points.append(dict(stealth, eps=None, label="stealth_anchor", status=0, n_eval=0))
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
        print(f"{p['label']:14s} Cd={p['Cd']:.6f}  sigma={p['sigma_agg']:.6f}  "
              f"status={p['status']}  evals={p['n_eval']}")
    print(f"wrote {args.out}  ({result['n_aero_calls']} aero gradient calls)")


if __name__ == "__main__":
    main()
