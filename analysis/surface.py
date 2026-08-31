# SPDX-License-Identifier: Apache-2.0
"""Wall pressure coefficient from a finished OpenFOAM case.

The airfoil patch is zeroGradient, so it stores no values of its own: the wall
pressure is the adjacent cell's, reached through the face owner list.
"""

import sys
from pathlib import Path

import numpy as np

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


def read_internal_scalar(path):
    body = Path(path).read_text().split("internalField", 1)[1]
    if body.lstrip().startswith("uniform"):
        return None
    return np.array([float(v) for v in morph._foam_list(body).split()])


def read_owner(path):
    return np.array([int(v) for v in morph._foam_list(Path(path).read_text()).split()])


def wall_cp(case, uinf=60.0, patch="airfoil"):
    """Return (x, y, cp) at the wall face centres.

    The mesh is one cell thick, so each wall face spans the whole span and the
    patch is already a single ring around the section.
    """
    case = Path(case)
    pm = case / "constant/polyMesh"
    p = read_internal_scalar(_latest_time(case) / "p")
    if p is None:
        raise ValueError(f"pressure is uniform in {case}")

    start, n = morph._boundary(pm / "boundary")[patch]
    owner = read_owner(pm / "owner")
    faces = morph._faces(pm / "faces")
    points = morph.read_points(pm / "points")

    centres = np.array([points[f].mean(axis=0) for f in faces[start:start + n]])
    cp = p[owner[start:start + n]] / (0.5 * uinf**2)
    return centres[:, 0], centres[:, 1], cp


def ordered_cp(case, uinf=60.0, patch="airfoil"):
    """Upper and lower surface Cp, each sorted by chordwise station."""
    x, y, cp = wall_cp(case, uinf, patch)
    out = []
    for side in (y >= 0, y < 0):
        order = np.argsort(x[side])
        out.append((x[side][order], cp[side][order]))
    return out
