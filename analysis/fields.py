# SPDX-License-Identifier: Apache-2.0
"""Cell-centred fields from a finished OpenFOAM case, sampled onto a plot grid.

The mesh is one cell thick, so cell centres collapse to a plane and the volume
fields are already two-dimensional.
"""

import re
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tesseracts/cfd/mesh"))

import morph  # noqa: E402


def _latest_time(case):
    times = [d for d in Path(case).iterdir() if d.is_dir() and _is_time(d.name)]
    return max(times, key=lambda d: float(d.name))


def _is_time(name):
    try:
        float(name)
    except ValueError:
        return False
    return True


def _internal(path, width):
    """Parse an internalField into (n,) or (n, width)."""
    body = Path(path).read_text().split("internalField", 1)[1]
    if body.lstrip().startswith("uniform"):
        return None
    text = morph._foam_list(body)
    if width == 1:
        return np.array([float(v) for v in text.split()])
    rows = [ln.strip()[1:-1].split() for ln in text.splitlines() if ln.strip().startswith("(")]
    return np.array([[float(v) for v in r[:width]] for r in rows])


def _labels(path):
    return np.array([int(v) for v in morph._foam_list(Path(path).read_text()).split()])


def cell_centres(case):
    """Approximate cell centres as the mean of their face centres."""
    pm = Path(case) / "constant/polyMesh"
    points = morph.read_points(pm / "points")
    faces = morph._faces(pm / "faces")
    owner, neighbour = _labels(pm / "owner"), _labels(pm / "neighbour")

    fc = np.array([points[f].mean(axis=0) for f in faces])
    n_cells = int(max(owner.max(), neighbour.max())) + 1
    total = np.zeros((n_cells, 3))
    count = np.zeros(n_cells)
    np.add.at(total, owner, fc)
    np.add.at(count, owner, 1.0)
    np.add.at(total, neighbour, fc[: len(neighbour)])
    np.add.at(count, neighbour, 1.0)
    return total / count[:, None]


def read_case(case):
    """Cell centres and the latest-time p and U for one case."""
    case = Path(case)
    latest = _latest_time(case)
    return {
        "xy": cell_centres(case)[:, :2],
        "p": _internal(latest / "p", 1),
        "U": _internal(latest / "U", 3)[:, :2],
    }


def on_grid(field, bbox=(-0.4, 1.7, -0.6, 0.6), nx=360, ny=260, loop=None):
    """Interpolate a cell field onto a regular grid, masking the interior."""
    xmin, xmax, ymin, ymax = bbox
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    out = {"x": gx, "y": gy}
    for name in ("p", "U"):
        v = field[name]
        if v.ndim == 1:
            out[name] = griddata(field["xy"], v, (gx, gy), method="linear")
        else:
            out[name] = np.stack(
                [griddata(field["xy"], v[:, k], (gx, gy), method="linear")
                 for k in range(v.shape[1])], axis=-1)
    if loop is not None:
        inside = _inside(loop, gx, gy)
        out["p"][inside] = np.nan
        out["U"][inside] = np.nan
        out["inside"] = inside
    return out


def _inside(loop, gx, gy):
    """Winding-number test against the closed boundary polygon."""
    a = np.asarray(loop)[:, :2]
    b = np.roll(a, -1, axis=0)
    angle = np.zeros(gx.shape)
    for p0, p1 in zip(a, b, strict=True):
        v0x, v0y = p0[0] - gx, p0[1] - gy
        v1x, v1y = p1[0] - gx, p1[1] - gy
        angle += np.arctan2(v0x * v1y - v0y * v1x, v0x * v1x + v0y * v1y)
    return np.abs(angle) > np.pi


def _rotate(vec, alpha_deg):
    a = np.radians(alpha_deg)
    c, s = np.cos(a), np.sin(a)
    return np.stack([vec[..., 0] * c + vec[..., 1] * s,
                     -vec[..., 0] * s + vec[..., 1] * c], axis=-1)


def body_frame(grid, alpha_deg):
    """Rotate the velocity into the body frame so the section sits level."""
    out = dict(grid)
    out["U"] = _rotate(grid["U"], alpha_deg)
    return out


def cp_field(grid, uinf):
    return grid["p"] / (0.5 * uinf**2)


CASE_RE = re.compile(r"[0-9a-f]{16}")
