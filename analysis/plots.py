# SPDX-License-Identifier: Apache-2.0
"""Figures for the writeup: Pareto front, shape overlays, RCS polars.

Every function takes plain arrays and an output path so they can be driven from
a sweep result file or from a notebook without importing the driver.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.pareto import nondominated_mask  # noqa: E402

AERO_COLOR = "#e2483d"
STEALTH_COLOR = "#3a6ea5"


def _finish(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def pareto_front(cd, sigma, labels=None, out_path="pareto.png"):
    """Drag against echo width, with the dominated points shown but greyed."""
    cd, sigma = np.asarray(cd, float), np.asarray(sigma, float)
    keep = nondominated_mask(np.column_stack([cd, sigma]))
    order = np.argsort(cd[keep])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cd[~keep], sigma[~keep], "o", color="0.7", ms=6, label="dominated")
    ax.plot(cd[keep][order], sigma[keep][order], "o-", color=STEALTH_COLOR, ms=7,
            label="Pareto front")
    if labels is not None:
        for x, y, name in zip(cd, sigma, labels, strict=True):
            if name in ("aero_anchor", "stealth_anchor"):
                ax.annotate(name.replace("_", " "), (x, y),
                            textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Cd at trimmed Cl")
    ax.set_ylabel("KS-aggregated echo width (m)")
    ax.set_title("stealth versus drag Pareto front")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def shape_overlay(designs, out_path="shapes.png"):
    """designs: sequence of (label, x_surf) with x_surf a closed (N, 2) loop."""
    fig, ax = plt.subplots(figsize=(9, 3))
    for label, loop in designs:
        loop = np.asarray(loop, float)
        ax.plot(loop[:, 0], loop[:, 1], lw=1.4, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("x / c")
    ax.set_title("section shapes along the front")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def rcs_polar(angles_deg, sigma_by_design, out_path="rcs.png"):
    """sigma_by_design: sequence of (label, sigma over the incidence sector)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, sigma in sigma_by_design:
        ax.semilogy(np.asarray(angles_deg, float), np.asarray(sigma, float),
                    lw=1.4, marker="o", ms=4, label=label)
    ax.set_xlabel("incidence angle (deg)")
    ax.set_ylabel("echo width (m)")
    ax.set_title("monostatic echo width over the frontal sector")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    return _finish(fig, out_path)


def gradient_agreement(rows, out_path="gradients.png"):
    """rows: sequence of (label, adjoint, finite_difference) scalars."""
    labels = [r[0] for r in rows]
    ad = np.array([r[1] for r in rows], float)
    fd = np.array([r[2] for r in rows], float)
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.19, fd, 0.36, color=AERO_COLOR, label="central finite difference")
    ax.bar(x + 0.19, ad, 0.36, color=STEALTH_COLOR, label="adjoint")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("dJ / dtheta")
    ax.set_title("gradient agreement")
    ax.legend(fontsize=9)
    return _finish(fig, out_path)
