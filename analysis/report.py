# SPDX-License-Identifier: Apache-2.0
"""Figures and the non-domination table for a finished sweep.

Shapes and RCS polars are recomputed from each point's design vector through the
geom and em tesseracts, so this needs no state beyond the sweep result file.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import numpy as np

# In single precision the recomputed x_surf drifts by 7e-8
jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import plots, surface, timings  # noqa: E402
from analysis.pareto import nondominated_mask  # noqa: E402
from driver.forward import em_inputs, geom_inputs  # noqa: E402
from driver.tesseracts import load_config, local_tesseract  # noqa: E402


def _case_dir(config, x_surf):
    """The run directory tesseract_api hashed this design into."""
    key = hashlib.sha1(
        np.ascontiguousarray(x_surf, dtype=np.float64).tobytes()
        + np.float64([config["aero"]["alpha_deg"], config["aero"]["reynolds"]]).tobytes()
    ).hexdigest()[:16]
    return ROOT / "tesseracts/cfd/_run" / key / "primal"


def _designs(config, points):
    geom, em = local_tesseract("geom"), local_tesseract("em")
    gi, ei = geom_inputs(config), em_inputs(config)
    for p in points:
        x_surf = np.asarray(geom.apply({"theta": np.asarray(p["theta"]), **gi})["x_surf"])
        polar = np.asarray(em.apply({"x_surf": x_surf, **ei})["sigma_by_angle"])
        yield p, x_surf, polar, _case_dir(config, x_surf)


def _table(points, keep):
    print(f"{'label':15s} {'Cd':>10s} {'Cl':>8s} {'sigma':>10s} {'eps*ref':>10s} "
          f"{'status':>6s}  front")
    for p, k in zip(points, keep, strict=True):
        eps = "-" if p.get("eps") is None else f"{p['eps'] * SIGMA_REF:.6f}"
        print(f"{p['label']:15s} {p['Cd']:10.6f} {p['Cl']:8.4f} {p['sigma_agg']:10.6f} "
              f"{eps:>10s} {p['status']:6d}  {'yes' if k else 'dominated'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--pareto", type=Path, nargs="+",
                        default=[ROOT / "outputs/pareto.json"])
    parser.add_argument("--figures", type=Path, default=ROOT / "analysis/figures")
    parser.add_argument("--overlay", type=int, default=3, help="designs in the shape plots")
    parser.add_argument("--run-root", type=Path, default=ROOT / "tesseracts/cfd/_run",
                        help="OpenFOAM run directories to scrape solver times from")
    args = parser.parse_args()

    config = load_config(args.baseline)
    results = [json.loads(p.read_text()) for p in args.pareto]

    points, seen = [], set()
    for r in results:
        for p in r["points"]:
            key = round(p["sigma_agg"], 9)
            if key not in seen:
                seen.add(key)
                points.append(p)
    points.sort(key=lambda p: -p["sigma_agg"])
    result = {**results[0], "points": points,
              "n_aero_calls": sum(r["n_aero_calls"] for r in results)}

    global SIGMA_REF
    SIGMA_REF = result["sigma_ref"]

    cd = np.array([p["Cd"] for p in points])
    sigma = np.array([p["sigma_agg"] for p in points])
    keep = nondominated_mask(np.column_stack([cd, sigma]))
    _table(points, keep)
    print(f"\nnon-dominated: {int(keep.sum())} of {len(points)}")
    print(f"Cd    {cd.min():.6f} -> {cd.max():.6f}  ({100 * (cd.max() / cd.min() - 1):+.1f}%)")
    print(f"sigma {sigma.max():.6f} -> {sigma.min():.6f}  "
          f"({100 * (sigma.min() / sigma.max() - 1):+.1f}%)")
    print(f"aero gradient calls: {result['n_aero_calls']}")

    args.figures.mkdir(parents=True, exist_ok=True)
    plots.pareto_front(cd, sigma, [p["label"] for p in points],
                       args.figures / "pareto_front.png")

    front = [i for i in np.argsort(sigma)[::-1] if keep[i]]
    picks = [front[i] for i in np.linspace(0, len(front) - 1, args.overlay).astype(int)]
    chosen = [points[i] for i in picks]
    e = config["em"]
    angles = np.linspace(e["incidence_deg"][0], e["incidence_deg"][1], e["incidence_count"])

    shapes, polars, cps = [], [], []
    for p, x_surf, polar, case in _designs(config, chosen):
        name = f"Cd {p['Cd']:.5f}, sigma {p['sigma_agg']:.5f}"
        shapes.append((name, x_surf))
        polars.append((name, polar))
        if case.exists():
            cps.append((name, surface.ordered_cp(case)))
    plots.shape_overlay(shapes, args.figures / "shapes.png")
    plots.rcs_polar(angles, polars, args.figures / "rcs_polar.png")
    if cps:
        plots.cp_distribution(cps, args.figures / "cp.png")
    else:
        print("no OpenFOAM cases found for the selected designs, skipping Cp")

    solve_times = timings.solver_times(args.run_root)
    grad_times = timings.gradient_times(args.run_root)
    if solve_times and grad_times.size:
        n = len(points[0]["theta"])
        for name, v in solve_times.items():
            print(f"  {name:14s} n={len(v):4d}  median {np.median(v):6.1f} s")
        print(f"  {'adjoint gradient':14s} n={len(grad_times):4d}  "
              f"median {np.median(grad_times):6.1f} s")
        plots.solver_cost(solve_times, grad_times, solve_times["primal"], n,
                          out_path=args.figures / "cost.png")
    else:
        print("no OpenFOAM logs under the run root, skipping the cost figure")

    series = []
    dirs = list(dict.fromkeys(q.parent for q in args.pareto))
    for d in dirs:
        for path in sorted(d.glob("trajectory_eps*.jsonl")):
            rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
            if not rows:
                continue
            name = path.stem.replace("trajectory_", "")
            if len(dirs) > 1:
                name = f"{d.name}/{name}"
            series.append((name, [r["Cd"] for r in rows]))
    if series:
        plots.convergence(series, args.figures / "convergence.png")
    print(f"wrote figures to {args.figures}")


if __name__ == "__main__":
    main()
