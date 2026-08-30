# SPDX-License-Identifier: Apache-2.0
"""Plot the z-plane of a 2D OpenFOAM polyMesh, optionally highlighting a cellSet."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection


def _foam_list(text):
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start:i]
    raise ValueError("no balanced list")


def read_points(path):
    body = _foam_list(path.read_text())
    return np.array(
        [[float(v) for v in ln.strip()[1:-1].split()]
         for ln in body.splitlines() if ln.strip().startswith("(")]
    )


def read_faces(path):
    body = _foam_list(path.read_text())
    faces = []
    for tok in re.findall(r"\d+\(([\d\s]+)\)", body):
        faces.append([int(v) for v in tok.split()])
    return faces


def read_boundary(path):
    text = path.read_text()
    out = {}
    for name, block in re.findall(r"(\w+)\s*\{([^}]*)\}", text):
        n = re.search(r"nFaces\s+(\d+)", block)
        s = re.search(r"startFace\s+(\d+)", block)
        if n and s:
            out[name] = (int(s.group(1)), int(n.group(1)))
    return out


def read_cellset(path):
    body = _foam_list(path.read_text())
    return {int(v) for v in re.findall(r"^\s*(\d+)\s*$", body, re.M)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, default=Path("mesh.png"))
    parser.add_argument("--set", dest="cellset", default=None, help="name of a cellSet to shade")
    parser.add_argument("--xlim", type=float, nargs=2, default=None)
    parser.add_argument("--ylim", type=float, nargs=2, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    pm = args.case / "constant/polyMesh"
    points = read_points(pm / "points")
    faces = read_faces(pm / "faces")
    boundary = read_boundary(pm / "boundary")

    back_name = "back" if "back" in boundary else next(iter(boundary))
    start, n = boundary[back_name]
    cell_faces = faces[start : start + n]

    polys = [points[f][:, :2] for f in cell_faces]
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.add_collection(
        PolyCollection(polys, facecolors="none", edgecolors="#3a6ea5", linewidths=0.15)
    )

    if args.cellset:
        ids = read_cellset(pm / "sets" / args.cellset)
        hits = [polys[i] for i in sorted(ids) if i < len(polys)]
        ax.add_collection(PolyCollection(hits, facecolors="#e2483d", edgecolors="none", alpha=0.6))

    for name in ("airfoil", "wall"):
        if name in boundary:
            s, m = boundary[name]
            segs = []
            for f in faces[s : s + m]:
                p = points[f][:, :2]
                zmin = points[f][:, 2].argmin()
                segs.append(p[[zmin, (zmin + 1) % len(f)]])
            ax.add_collection(LineCollection(segs, colors="k", linewidths=0.8))

    allp = np.vstack(polys)
    ax.set_xlim(args.xlim or (allp[:, 0].min(), allp[:, 0].max()))
    ax.set_ylim(args.ylim or (allp[:, 1].min(), allp[:, 1].max()))
    ax.set_aspect("equal")
    ax.set_title(f"{args.case.name}  ({n} cells)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
