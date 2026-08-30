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


def projection_weights(ref_xy, target_curve):
    """Sparse weights W (n_ref, n_target) for the nearest-point-on-polyline map:
    proj_i = W[i] @ target_curve, linearized at the current nearest segment."""
    a = np.asarray(target_curve)[:, :2]
    b = np.roll(a, -1, axis=0)
    ab = b - a
    ab2 = np.einsum("sd,sd->s", ab, ab) + 1e-30
    ap = ref_xy[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nsd,sd->ns", ap, ab) / ab2, 0.0, 1.0)
    proj = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d2 = np.einsum("nsd,nsd->ns", ref_xy[:, None, :] - proj, ref_xy[:, None, :] - proj)
    nearest = np.argmin(d2, axis=1)
    rows = np.arange(len(ref_xy))
    tn = t[rows, nearest]
    w = np.zeros((len(ref_xy), len(a)))
    w[rows, nearest] = 1.0 - tn
    w[rows, (nearest + 1) % len(a)] += tn
    return w


def surface_displacement(ref_xy, target_curve):
    """Displacement of each reference surface node to its closest point on the
    target boundary polyline. Linear in target_curve for a fixed assignment."""
    return projection_weights(ref_xy, target_curve) @ np.asarray(target_curve)[:, :2] - ref_xy


def build_operator(mesh_xy, centers_xy, radius):
    """Rows of the linear map from surface-node displacement to mesh-node
    displacement: mesh_disp = (evaluate @ solve) @ surface_disp."""
    d = np.linalg.norm(centers_xy[:, None, :] - centers_xy[None, :, :], axis=-1)
    a = _wendland(d / radius) + 1e-9 * np.eye(len(centers_xy))
    g = np.linalg.norm(mesh_xy[:, None, :] - centers_xy[None, :, :], axis=-1)
    evaluate = _wendland(g / radius)
    return evaluate @ np.linalg.inv(a)


def _centers(case, points, patch):
    ids = patch_point_ids(case, patch)
    return ids[points[ids, 2] < points[ids, 2].mean()]


def morph_case(case, target_curve, radius=0.6, patch="airfoil"):
    pm = Path(case) / "constant/polyMesh"
    points = read_points(pm / "points")
    centers_ids = _centers(case, points, patch)
    centers_xy = points[centers_ids, :2]

    disp = surface_displacement(centers_xy, np.asarray(target_curve)[:, :2])
    operator = build_operator(points[:, :2], centers_xy, radius)
    points[:, :2] += operator @ disp
    write_points(pm / "points", points)
    return points


def patch_point_normals(case, patch="airfoil"):
    """Unit outward xy normal at each patch point, indexed by mesh point id
    (nPoints, 2), zero away from the patch. Outward means out of the fluid."""
    pm = Path(case) / "constant/polyMesh"
    points = read_points(pm / "points")
    start, n = _boundary(pm / "boundary")[patch]
    faces = _faces(pm / "faces")

    normals = np.zeros((len(points), 2))
    centroid = points[_centers(case, points, patch), :2].mean(axis=0)
    for f in faces[start : start + n]:
        p = points[f]
        fn = np.cross(p[1] - p[0], p[2] - p[0])
        nrm = np.array([fn[0], fn[1]])
        mid = p[:, :2].mean(axis=0)
        if np.dot(nrm, mid - centroid) < 0:
            nrm = -nrm
        for vid in f:
            normals[vid] += nrm
    mag = np.linalg.norm(normals, axis=1)
    nz = mag > 1e-14
    normals[nz] /= mag[nz, None]
    return normals


def morph_vjp(case, target_curve, cotangent_xy, radius=0.6, patch="airfoil"):
    """Pull dJ/d(mesh point xy) back to dJ/d(target_curve) through the morph.
    cotangent_xy: (nPoints, 2). Returns (n_target, 2)."""
    pm = Path(case) / "constant/polyMesh"
    points = read_points(pm / "points")
    centers_xy = points[_centers(case, points, patch), :2]

    operator = build_operator(points[:, :2], centers_xy, radius)
    w = projection_weights(centers_xy, np.asarray(target_curve)[:, :2])
    return w.T @ (operator.T @ np.asarray(cotangent_xy))
