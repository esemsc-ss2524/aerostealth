# SPDX-License-Identifier: Apache-2.0
"""Adjoint versus central finite difference for dCd/dtheta and dCl/dtheta.

One primal and two adjoints give the whole gradient; the finite difference costs
two converged primals per component, which is the cost argument the writeup
makes. Every primal is cached by geometry hash, so a rerun is nearly free.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tesseracts/cfd"))
sys.path.insert(0, str(ROOT / "tesseracts/cfd/mesh"))

import sensitivity  # noqa: E402

from analysis import plots  # noqa: E402
from driver.forward import baseline_theta, geom_inputs  # noqa: E402
from driver.tesseracts import load_config, local_tesseract  # noqa: E402

COEFFS = ("Cd", "Cl")


def _labels(n):
    half = n // 2
    return [f"upper {i}" for i in range(half)] + [f"lower {i}" for i in range(half)]


def adjoint_gradients(geom, geom_in, theta, alpha_deg, reynolds, workdir):
    sens = sensitivity.shape_sensitivity(
        _x_surf(geom, geom_in, theta), alpha_deg, reynolds, workdir / "base"
    )
    grads = {
        c: _to_theta(geom, geom_in, theta, sens[f"d{c}_dx_surf"]) for c in COEFFS
    }
    return grads, sens


def _x_surf(geom, geom_in, theta):
    out = geom.apply({"theta": np.asarray(theta), **geom_in})
    return np.asarray(out["x_surf"], dtype=np.float64)


def _to_theta(geom, geom_in, theta, cotangent):
    out = geom.vector_jacobian_product(
        {"theta": np.asarray(theta), **geom_in}, ["theta"], ["x_surf"],
        {"x_surf": np.asarray(cotangent)},
    )
    return np.asarray(out["theta"], dtype=np.float64)


def finite_differences(geom, geom_in, theta, alpha_deg, reynolds, workdir, step,
                       components, jobs=1):
    """Each perturbed primal is an independent OpenFOAM case, so they run
    concurrently. The geometry is evaluated up front to keep JAX on one thread."""
    cases = []
    for k in components:
        for tag, sign in (("p", 1.0), ("m", -1.0)):
            shifted = theta.copy()
            shifted[k] += sign * step
            cases.append((k, tag, _x_surf(geom, geom_in, shifted)))

    sensitivity.runner._reference_mesh(workdir)
    done = {}

    def solve(case):
        k, tag, x_surf = case
        r = sensitivity.runner.run_primal(x_surf, alpha_deg, reynolds, workdir / f"{tag}{k}")
        print(f"  {tag}{k:<3d} Cd={r['Cd']:.7f}  Cl={r['Cl']:.6f}", flush=True)
        return (k, tag), r

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for key, result in pool.map(solve, cases):
            done[key] = result

    fd = {c: np.full(theta.size, np.nan) for c in COEFFS}
    for k in components:
        for c in COEFFS:
            fd[c][k] = (done[(k, "p")][c] - done[(k, "m")][c]) / (2.0 * step)
    return fd


def report(adj, fd, components, labels):
    for c in COEFFS:
        a, f = adj[c][components], fd[c][components]
        cos = float(a @ f / (np.linalg.norm(a) * np.linalg.norm(f)))
        print(f"\n{c}:  |adjoint|={np.linalg.norm(a):.4e}  |fd|={np.linalg.norm(f):.4e}  "
              f"cos={cos:+.4f}  angle={np.degrees(np.arccos(np.clip(cos, -1, 1))):.1f} deg")
        print("  component      adjoint            fd            ratio")
        for k in components:
            print(f"  {labels[k]:12s} {adj[c][k]:+.6e}  {fd[c][k]:+.6e}  "
                  f"{adj[c][k] / fd[c][k]:8.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--step", type=float, default=0.004)
    parser.add_argument("--workdir", type=Path, default=ROOT / "tesseracts/cfd/_check")
    parser.add_argument("--figures", type=Path, default=ROOT / "analysis/figures")
    parser.add_argument("--data", type=Path, default=ROOT / "analysis/figures/gradient_check.npz")
    parser.add_argument("--jobs", type=int, default=6, help="concurrent OpenFOAM cases")
    args = parser.parse_args()

    config = load_config(args.baseline)
    alpha = float(config["aero"]["alpha_deg"])
    reynolds = float(config["aero"]["reynolds"])
    geom, geom_in = local_tesseract("geom"), geom_inputs(config)
    theta = np.asarray(baseline_theta(config), dtype=np.float64)
    components = np.arange(theta.size)
    labels = _labels(theta.size)

    adj, sens = adjoint_gradients(geom, geom_in, theta, alpha, reynolds, args.workdir)
    print(f"base Cd={sens['Cd']:.7f}  Cl={sens['Cl']:.5f}  alpha={alpha}", flush=True)
    fd = finite_differences(geom, geom_in, theta, alpha, reynolds,
                            args.workdir, args.step, components, jobs=args.jobs)
    report(adj, fd, components, labels)

    args.figures.mkdir(parents=True, exist_ok=True)
    names = {"Cd": "cfd_adjoint_vs_fd.png", "Cl": "cfd_lift_adjoint_vs_fd.png"}
    for c in COEFFS:
        out = plots.gradient_agreement(
            [(labels[k], adj[c][k], fd[c][k]) for k in components], args.figures / names[c])
        print(f"wrote {out}")
    np.savez(args.data, theta=theta, step=args.step, alpha_deg=alpha,
             **{f"adjoint_{c}": adj[c] for c in COEFFS},
             **{f"fd_{c}": fd[c] for c in COEFFS})
    print(f"wrote {args.data}")


if __name__ == "__main__":
    main()
