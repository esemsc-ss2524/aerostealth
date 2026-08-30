# SPDX-License-Identifier: Apache-2.0
"""RBF morph: surface nodes reach the target curve, far nodes stay put, map is linear."""

import sys
from pathlib import Path

import numpy as np

MESH_DIR = Path(__file__).resolve().parents[1] / "tesseracts/cfd/mesh"
sys.path.insert(0, str(MESH_DIR))

import morph  # noqa: E402


def _ellipse(n, a=0.5, b=0.06, cx=0.5):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + a * np.cos(t), b * np.sin(t)])


def test_surface_displacement_hits_target():
    ref = _ellipse(120)
    target = _ellipse(200, b=0.09)
    disp = morph.surface_displacement(ref, target)
    residual = morph.surface_displacement(ref + disp, target)
    assert np.linalg.norm(residual, axis=1).max() < 1e-9


def test_operator_is_local_and_linear():
    centers = _ellipse(80)
    near = _ellipse(80, a=0.55, b=0.2)
    far = np.column_stack([20 * np.cos(np.linspace(0, 6, 50)), 20 * np.sin(np.linspace(0, 6, 50))])
    mesh_xy = np.vstack([centers, near, far])

    op = morph.build_operator(mesh_xy, centers, radius=0.6)
    d1 = morph.surface_displacement(centers, _ellipse(200, b=0.09))
    d2 = morph.surface_displacement(centers, _ellipse(200, b=0.03))

    assert np.abs(op[-50:] @ d1).max() < 1e-12
    assert np.allclose(op @ (d1 + d2), op @ d1 + op @ d2, atol=1e-12)
    assert np.linalg.norm((op @ d1)[:80] - d1, axis=1).max() < 1e-6
