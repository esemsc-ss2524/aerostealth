# SPDX-License-Identifier: Apache-2.0
"""Figures and the non-domination table for a finished sweep.

Shapes and RCS polars are recomputed from each point's design vector through the
geom and em tesseracts, so this needs no state beyond the sweep result file.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import plots  # noqa: E402
from analysis.pareto import nondominated_mask  # noqa: E402
from driver.forward import em_inputs, geom_inputs  # noqa: E402
from driver.tesseracts import load_config, local_tesseract  # noqa: E402


def _designs(config, points):
    geom, em = local_tesseract("geom"), local_tesseract("em")
    gi, ei = geom_inputs(config), em_inputs(config)
    for p in points:
        x_surf = np.asarray(geom.apply({"theta": np.asarray(p["theta"]), **gi})["x_surf"])
        polar = np.asarray(em.apply({"x_surf": x_surf, **ei})["sigma_by_angle"])
        yield p, x_surf, polar


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
    parser.add_argument("--pareto", type=Path, default=ROOT / "outputs/pareto.json")
    parser.add_argument("--figures", type=Path, default=ROOT / "analysis/figures")
    parser.add_argument("--overlay", type=int, default=3, help="designs in the shape plots")
    args = parser.parse_args()

    config = load_config(args.baseline)
    result = json.loads(args.pareto.read_text())
    points = result["points"]

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

    shapes, polars = [], []
    for p, x_surf, polar in _designs(config, chosen):
        name = f"Cd {p['Cd']:.5f}, sigma {p['sigma_agg']:.5f}"
        shapes.append((name, x_surf))
        polars.append((name, polar))
    plots.shape_overlay(shapes, args.figures / "shapes.png")
    plots.rcs_polar(angles, polars, args.figures / "rcs_polar.png")
    print(f"wrote figures to {args.figures}")


if __name__ == "__main__":
    main()
