# SPDX-License-Identifier: Apache-2.0
"""RBF morphing of the reference C-grid onto a new airfoil boundary curve.

The surface displacement is a compact-support Wendland RBF interpolation from the
reference airfoil nodes; it decays to zero before the far field, so the volume
mesh moves without remeshing. The map is linear in the target curve for fixed
centers and support radius, which is what the drag adjoint projects back through.
"""

import re
from pathlib import Path

import numpy as np


def _foam_list(text):
    depth, start = 0, None
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
    body = _foam_list(Path(path).read_text())
    return np.array(
        [[float(v) for v in ln.strip()[1:-1].split()]
         for ln in body.splitlines() if ln.strip().startswith("(")]
    )


def write_points(path, points):
    path = Path(path)
    head = path.read_text().split("(", 1)[0]
    lines = [f"({p[0]:.10g} {p[1]:.10g} {p[2]:.10g})" for p in points]
    path.write_text(f"{head}(\n" + "\n".join(lines) + "\n)\n")


def _boundary(path):
    text = Path(path).read_text()
    out = {}
    for name, block in re.findall(r"(\w+)\s*\{([^}]*)\}", text):
        n = re.search(r"nFaces\s+(\d+)", block)
        s = re.search(r"startFace\s+(\d+)", block)
        if n and s:
            out[name] = (int(s.group(1)), int(n.group(1)))
    return out


def _faces(path):
    body = _foam_list(Path(path).read_text())
    return [[int(v) for v in tok.split()] for tok in re.findall(r"\d+\(([\d\s]+)\)", body)]


def patch_point_ids(case, patch="airfoil"):
    pm = Path(case) / "constant/polyMesh"
    start, n = _boundary(pm / "boundary")[patch]
    faces = _faces(pm / "faces")
    ids = {vid for f in faces[start : start + n] for vid in f}
    return np.array(sorted(ids))


def _wendland(r):
    r = np.clip(r, 0.0, 1.0)
    return (1.0 - r) ** 4 * (4.0 * r + 1.0)


def surface_displacement(ref_xy, target_curve):
    """Displacement of each reference surface node to its closest point on the
    target boundary polyline. No ordering assumption; linear in target_curve for
    a fixed nearest-segment assignment."""
    a = np.asarray(target_curve)[:, :2]
    b = np.roll(a, -1, axis=0)
    ab = b - a
    ab2 = np.einsum("sd,sd->s", ab, ab) + 1e-30
    ap = ref_xy[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nsd,sd->ns", ap, ab) / ab2, 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d2 = np.einsum("nsd,nsd->ns", ref_xy[:, None, :] - proj, ref_xy[:, None, :] - proj)
    nearest = np.argmin(d2, axis=1)
    return proj[np.arange(len(ref_xy)), nearest] - ref_xy


def build_operator(mesh_xy, centers_xy, radius):
    """Rows of the linear map from surface-node displacement to mesh-node
    displacement: mesh_disp = (evaluate @ solve) @ surface_disp."""
    d = np.linalg.norm(centers_xy[:, None, :] - centers_xy[None, :, :], axis=-1)
    a = _wendland(d / radius) + 1e-9 * np.eye(len(centers_xy))
    g = np.linalg.norm(mesh_xy[:, None, :] - centers_xy[None, :, :], axis=-1)
    evaluate = _wendland(g / radius)
    return evaluate @ np.linalg.inv(a)


def morph_case(case, target_curve, radius=0.6, patch="airfoil"):
    pm = Path(case) / "constant/polyMesh"
    points = read_points(pm / "points")
    ids = patch_point_ids(case, patch)
    centers_ids = ids[points[ids, 2] < points[ids, 2].mean()]
    centers_xy = points[centers_ids, :2]

    disp = surface_displacement(centers_xy, np.asarray(target_curve)[:, :2])
    operator = build_operator(points[:, :2], centers_xy, radius)
    points[:, :2] += operator @ disp
    write_points(pm / "points", points)
    return points
