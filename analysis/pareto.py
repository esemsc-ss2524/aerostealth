# SPDX-License-Identifier: Apache-2.0
"""Pareto-front bookkeeping for the (Cd, sigma_agg) objective pair."""

import numpy as np


def nondominated_mask(points):
    """Boolean mask of non-dominated rows, minimizing every column."""
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        others = np.delete(pts, i, axis=0)
        dominated = np.any(
            np.all(others <= pts[i], axis=1) & np.any(others < pts[i], axis=1)
        )
        mask[i] = not dominated
    return mask


def sorted_front(points):
    pts = np.asarray(points, dtype=float)
    front = pts[nondominated_mask(pts)]
    return front[np.argsort(front[:, 0])]
