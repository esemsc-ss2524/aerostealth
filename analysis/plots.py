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

# One width for every figure, so that all of them render at the same text size
# when displayed at a common width. Only the height varies.
FIG_W = 7.5
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


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

    fig, ax = plt.subplots(figsize=(FIG_W, 5.2))
    ax.plot(cd[~keep], sigma[~keep], "o", color="0.7", ms=6, label="dominated")
    ax.plot(cd[keep][order], sigma[keep][order], "o-", color=STEALTH_COLOR, ms=7,
            label="Pareto front")
    if labels is not None:
        for x, y, name in zip(cd, sigma, labels, strict=True):
            if name in ("aero_anchor", "stealth_anchor"):
                ax.annotate(name.replace("_", " "), (x, y),
                            textcoords="offset points", xytext=(8, 6))
    ax.set_xlabel("Cd at fixed alpha, Cl >= Cl*")
    ax.set_ylabel("KS-aggregated echo width (m)")
    ax.set_title("stealth versus drag Pareto front")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def shape_overlay(designs, out_path="shapes.png"):
    """designs: sequence of (label, x_surf) with x_surf a closed (N, 2) loop."""
    fig, ax = plt.subplots(figsize=(FIG_W, 2.8))
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
    fig, ax = plt.subplots(figsize=(FIG_W, 5.2))
    for label, sigma in sigma_by_design:
        ax.semilogy(np.asarray(angles_deg, float), np.asarray(sigma, float),
                    lw=1.4, marker="o", ms=4, label=label)
    ax.set_xlabel("incidence angle (deg)")
    ax.set_ylabel("echo width (m)")
    ax.set_title("monostatic echo width over the frontal sector")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    return _finish(fig, out_path)


def cp_distribution(series, out_path="cp.png"):
    """series: sequence of (label, [(x_upper, cp_upper), (x_lower, cp_lower)])."""
    fig, ax = plt.subplots(figsize=(FIG_W, 5.2))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(series)))
    for (label, sides), color in zip(series, colors, strict=True):
        for i, (x, cp) in enumerate(sides):
            ax.plot(x, cp, lw=1.3, color=color, label=label if i == 0 else None)
    ax.invert_yaxis()
    ax.set_xlabel("x / c")
    ax.set_ylabel("Cp")
    ax.set_title("wall pressure coefficient")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def solver_cost(solve_times, adjoint_gradient, primal_times, n_params,
                out_path="cost.png"):
    """Measured solver seconds: the three solve types, then the two ways to get
    a gradient at n_params variables.

    Every box is measured. The finite-difference row is the measured primal
    distribution scaled by the 2n solves a central difference needs, which is
    arithmetic rather than an extrapolation of anything untested.
    """
    rows = list(solve_times.items())
    rows.append(("adjoint gradient\n(1 primal + 2 adjoints)", adjoint_gradient))
    rows.append((f"central FD gradient\n({2 * n_params} primals, n={n_params})",
                 2 * n_params * np.asarray(primal_times, float)))
    labels = [f"{name}\nn={len(v)}" if "\n" not in name else name for name, v in rows]

    fig, ax = plt.subplots(figsize=(FIG_W, 4.6))
    colors = [STEALTH_COLOR] * len(solve_times) + [STEALTH_COLOR, AERO_COLOR]
    bp = ax.boxplot([v for _, v in rows], vert=False, widths=0.6, patch_artist=True,
                    labels=labels, showfliers=False)
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for median in bp["medians"]:
        median.set_color("k")

    ratio = np.median(rows[-1][1]) / np.median(adjoint_gradient)
    ax.set_xscale("log")
    ax.set_xlabel("solver seconds (log scale)")
    ax.set_title(f"measured gradient cost, {ratio:.1f}x at n={n_params} "
                 f"and linear in n thereafter")
    ax.grid(alpha=0.3, axis="x", which="both")
    return _finish(fig, out_path)


def convergence(series, out_path="convergence.png"):
    """series: sequence of (label, Cd per evaluation) for one optimizer run each."""
    fig, ax = plt.subplots(figsize=(FIG_W, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(series)))
    for (label, cd), color in zip(series, colors, strict=True):
        ax.plot(np.arange(1, len(cd) + 1), np.asarray(cd, float), "o-", ms=3.5,
                lw=1.3, color=color, label=label)
    ax.set_xlabel("MMA evaluation")
    ax.set_ylabel("Cd")
    ax.set_title("drag against evaluation, one line per epsilon level")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    return _finish(fig, out_path)


def gradient_agreement(rows, out_path="gradients.png"):
    """rows: sequence of (label, adjoint, finite_difference) scalars."""
    labels = [r[0] for r in rows]
    ad = np.array([r[1] for r in rows], float)
    fd = np.array([r[2] for r in rows], float)
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(FIG_W, 4.2))
    ax.bar(x - 0.19, fd, 0.36, color=AERO_COLOR, label="central finite difference")
    ax.bar(x + 0.19, ad, 0.36, color=STEALTH_COLOR, label="adjoint")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("dJ / dtheta")
    ax.set_title("gradient agreement")
    ax.legend(fontsize=9)
    return _finish(fig, out_path)
