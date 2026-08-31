# SPDX-License-Identifier: Apache-2.0
"""Animations of the Pareto front for the README and for talks.

`tour` walks the front, showing the flow and the radar signature of the design
under the cursor. `mechanism` morphs between the two ends of the front to show
why the trade-off exists: the leading edge sharpens, which removes the specular
return and costs suction peak.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tesseracts/geom"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import cst  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from analysis import em_field, fields  # noqa: E402
from analysis.pareto import nondominated_mask  # noqa: E402
from driver.forward import geom_inputs  # noqa: E402
from driver.tesseracts import load_config, local_tesseract  # noqa: E402

C_LIGHT = 299792458.0
INK = "#1b1b1b"
AERO = "#e2483d"
STEALTH = "#3a6ea5"


def _case_dir(config, x_surf):
    key = hashlib.sha1(
        np.ascontiguousarray(x_surf, dtype=np.float64).tobytes()
        + np.float64([config["aero"]["alpha_deg"], config["aero"]["reynolds"]]).tobytes()
    ).hexdigest()[:16]
    return ROOT / "tesseracts/cfd/_run" / key / "primal"


def load_front(pareto_paths):
    """Every evaluated design, and the non-dominated subset ordered by echo width."""
    points, seen = [], set()
    for path in pareto_paths:
        for p in json.loads(Path(path).read_text())["points"]:
            key = round(p["sigma_agg"], 9)
            if key not in seen:
                seen.add(key)
                points.append(p)
    cd = np.array([p["Cd"] for p in points])
    sigma = np.array([p["sigma_agg"] for p in points])
    keep = nondominated_mask(np.column_stack([cd, sigma]))
    front = [points[i] for i in np.argsort(sigma)[::-1] if keep[i]]

    cloud = []
    for path in pareto_paths:
        for tr in sorted(Path(path).parent.glob("trajectory_*.jsonl")):
            for line in tr.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    cloud.append((row["Cd"], row["sigma_agg"]))
    return points, front, np.array(cloud)


def _geom(config):
    return local_tesseract("geom"), geom_inputs(config)


def surface(geom, gi, theta):
    return np.asarray(geom.apply({"theta": np.asarray(theta), **gi})["x_surf"])


def le_radius(theta, n=401):
    psi, yu, _ = cst.evaluate(np.asarray(theta), n, 0.0)
    psi, yu = np.asarray(psi), np.asarray(yu)
    m = (psi > 1e-4) & (psi < 0.01)
    return float(np.median(yu[m] ** 2 / (2 * psi[m])))


def k_chord(config):
    return 2.0 * np.pi * float(config["em"]["frequency_hz"]) / C_LIGHT * float(
        config["em"]["chord_m"])


def fit_y(fig, ax, xlim, box, y_centre=0.0):
    """Set ylim so that equal-aspect data exactly fills the axes box.

    `box` is the position captured before any aspect adjustment: matplotlib
    shrinks the axes to satisfy `set_aspect`, so reading it back each frame
    would compound.
    """
    ax.set_position(box)
    ax.set_xlim(*xlim)
    w = box.width * fig.get_figwidth()
    h = box.height * fig.get_figheight()
    half = 0.5 * (xlim[1] - xlim[0]) * h / w
    ax.set_ylim(y_centre - half, y_centre + half)
    ax.set_aspect("equal")


def _style(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.75")
    if title:
        ax.set_title(title, fontsize=11, color=INK, pad=5)


def load_walks(pareto_paths, featured="trajectory_eps5.jsonl"):
    """Every epsilon-level MMA run as an ordered walk, and the one that reaches
    the knee.

    The anchor run is left out. It solves a different problem, with no echo
    width constraint, and its excursion to $C_d$ 0.0113 stretches both axes far
    enough to compress everything else.
    """
    walks, feat = [], None
    for path in pareto_paths:
        for tr in sorted(Path(path).parent.glob("trajectory_eps*.jsonl")):
            rows = [json.loads(ln) for ln in tr.read_text().splitlines() if ln.strip()]
            if not rows:
                continue
            is_feat = tr.name == featured and tr.parent == Path(pareto_paths[0]).parent
            walks.append({"rows": rows, "featured": is_feat})
            if is_feat:
                feat = rows
    return walks, feat


def _front_distance(pts, front_xy, xr, yr):
    """Normalized distance from each point to the front polyline."""
    p = np.asarray(pts, dtype=float) / np.array([xr, yr])
    f = np.asarray(front_xy, dtype=float) / np.array([xr, yr])
    a, b = f[:-1], f[1:]
    ab = b - a
    t = np.clip(((p[:, None, :] - a) * ab).sum(-1) / (ab**2).sum(-1), 0.0, 1.0)
    proj = a + t[..., None] * ab
    return np.linalg.norm(p[:, None, :] - proj, axis=-1).min(axis=1)


def _blend(c0, c1, t):
    c0, c1 = np.array(matplotlib.colors.to_rgb(c0)), np.array(matplotlib.colors.to_rgb(c1))
    return [tuple(c0 + (c1 - c0) * float(x)) for x in np.clip(t, 0.0, 1.0)]


def tour(config, front, walks, featured, out, fps=20, hold=4, dwell=9, ease=5):
    """Optimizer walks converging on the front, then a tour along it."""
    geom, gi = _geom(config)
    k = k_chord(config)
    angles = np.linspace(float(config["em"]["incidence_deg"][0]),
                         float(config["em"]["incidence_deg"][1]),
                         int(config["em"]["incidence_count"]))
    dirs = np.stack([np.cos(np.radians(angles)), np.sin(np.radians(angles))], axis=-1)
    chord_m = float(config["em"]["chord_m"])
    uinf = 60.0

    def panel_data(loop, case_dir):
        case = Path(case_dir)
        grid = fields.on_grid(fields.read_case(case), loop=loop) if case.exists() else None
        return {"loop": loop, "grid": grid,
                "polar": em_field.echo_width(loop[:-1], k, dirs) * chord_m}

    print("precomputing front stations", flush=True)
    stations = []
    for i, pt in enumerate(front):
        loop = surface(geom, gi, pt["theta"])
        stations.append({"p": pt, **panel_data(loop, _case_dir(config, loop))})
        print(f"  station {i + 1}/{len(front)}", flush=True)

    print("precomputing the featured run", flush=True)
    feat = []
    for i, row in enumerate(featured):
        loop = np.asarray(row["x_surf"])
        feat.append({"row": row, **panel_data(loop, ROOT / row["case"] / "primal")})
        print(f"  iteration {i + 1}/{len(featured)}", flush=True)

    cd = np.array([s["p"]["Cd"] for s in stations])
    sg = np.array([s["p"]["sigma_agg"] for s in stations])
    front_xy = np.column_stack([cd, sg])

    allpts = np.array([[r["Cd"], r["sigma_agg"]] for w in walks for r in w["rows"]])
    xlo, xhi = allpts[:, 0].min(), max(allpts[:, 0].max(), cd.max())
    ylo = min(allpts[:, 1].min(), sg.min())
    yhi = max(allpts[:, 1].max(), sg.max())
    xr, yr = xhi - xlo, yhi - ylo
    dref = np.percentile(_front_distance(allpts, front_xy, xr, yr), 75)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1, 1],
                          left=0.07, right=0.97, top=0.90, bottom=0.09,
                          wspace=0.18, hspace=0.28)
    ax_p = fig.add_subplot(gs[:, 0])
    ax_f = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1], projection="polar")
    flow_box = ax_f.get_position().frozen()

    ax_p.set_xlabel("drag coefficient $C_d$", fontsize=11)
    ax_p.set_ylabel("radar echo width (m)", fontsize=11)
    ax_p.tick_params(labelsize=9)
    ax_p.grid(alpha=0.25)
    ax_p.set_xlim(xlo - 0.03 * xr, xhi + 0.05 * xr)
    ax_p.set_ylim(ylo - 0.05 * yr, yhi + 0.05 * yr)

    scat = ax_p.scatter([], [], s=22, zorder=3)
    fscat = ax_p.scatter([], [], s=46, zorder=5, edgecolors="white", linewidths=0.6)
    line, = ax_p.plot([], [], "-", color=STEALTH, lw=2.4, zorder=4)
    dots, = ax_p.plot([], [], "o", color=STEALTH, ms=6, zorder=5)
    cursor, = ax_p.plot([], [], "o", color=AERO, ms=14, zorder=7)

    lo = min(min(s["polar"].min() for s in stations), min(f["polar"].min() for f in feat))
    hi = max(max(s["polar"].max() for s in stations), max(f["polar"].max() for f in feat))
    rad = np.radians(angles)

    def draw_flow(d, title):
        ax_f.clear()
        _style(ax_f, title)
        if d["grid"] is None:
            return
        g = d["grid"]
        ax_f.contourf(g["x"], g["y"], fields.cp_field(g, uinf),
                      levels=np.linspace(-1.4, 1.4, 41), cmap="RdBu_r", extend="both")
        u = g["U"]
        ax_f.streamplot(g["x"][0], g["y"][:, 0], u[..., 0], u[..., 1],
                        color="0.25", linewidth=0.5, density=1.1, arrowsize=0.6)
        ax_f.fill(d["loop"][:, 0], d["loop"][:, 1], color=INK, zorder=5)
        fit_y(fig, ax_f, (-0.3, 1.45), flow_box)

    def draw_em(d, colour, subtitle):
        ax_e.clear()
        ax_e.set_theta_zero_location("E")
        ax_e.set_thetamin(angles[0])
        ax_e.set_thetamax(angles[-1])
        ax_e.set_ylim(lo - 0.12 * (hi - lo), hi + 0.05 * (hi - lo))
        for s in stations:
            ax_e.plot(rad, s["polar"], color="0.85", lw=1.0, zorder=1)
        ax_e.fill_between(rad, lo - 0.12 * (hi - lo), d["polar"],
                          color=colour, alpha=0.25, zorder=2)
        ax_e.plot(rad, d["polar"], color=colour, lw=2.6, zorder=3)
        ax_e.tick_params(labelsize=9)
        ax_e.set_rlabel_position(-34)
        ax_e.set_yticks(np.linspace(lo, hi, 3))
        ax_e.set_yticklabels([f"{v:.4f}" for v in np.linspace(lo, hi, 3)], fontsize=8)
        ax_e.set_title(f"radar signature over the frontal sector\n{subtitle}",
                       fontsize=11, pad=14)

    n_iter = max(len(w["rows"]) for w in walks)
    walk = []
    for i in range(len(stations)):
        walk += [(i, 0.0)] * dwell
        if i < len(stations) - 1:
            walk += [(i, (j + 1) / (ease + 1)) for j in range(ease)]
    n_a = n_iter * hold
    n_b = 12
    total = n_a + n_b + len(walk)
    arrows = []

    def clear_arrows():
        while arrows:
            arrows.pop().remove()

    def update(f):
        clear_arrows()
        if f < n_a + n_b:
            step = min(f // hold, n_iter - 1) if f < n_a else n_iter - 1
            pts, cols, fpts, fcols = [], [], [], []
            for w in walks:
                rows = w["rows"][: step + 1]
                xy = np.array([[r["Cd"], r["sigma_agg"]] for r in rows])
                d = _front_distance(xy, front_xy, xr, yr)
                if w["featured"] and f < n_a:
                    fpts.append(xy)
                    fcols.append(_blend("#f6b8b2", AERO, np.linspace(0.25, 1.0, len(xy))))
                else:
                    pts.append(xy)
                    cols.append(_blend("0.80", STEALTH, 1.0 - d / dref))
                tail = xy[-3:] if f < n_a else xy[:0]
                for q in range(len(tail) - 1):
                    arrows.append(ax_p.annotate(
                        "", xy=tail[q + 1], xytext=tail[q],
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, alpha=0.55,
                                        color=AERO if w["featured"] else "0.55"),
                        zorder=2))
            scat.set_offsets(np.vstack(pts))
            scat.set_facecolor([c for cc in cols for c in cc])
            if fpts:
                fscat.set_offsets(np.vstack(fpts))
                fscat.set_facecolor([c for cc in fcols for c in cc])
            else:
                fscat.set_offsets(np.empty((0, 2)))

            d = feat[min(step, len(feat) - 1)]
            draw_flow(d, "flow: pressure coefficient and streamlines")
            draw_em(d, AERO, "one optimizer run, in red on the left")
            if f < n_a:
                fig.suptitle(f"{len(walks)} optimizer runs, iteration {step + 1}"
                             f" of {n_iter}", fontsize=15, color=INK)
            else:
                line.set_data(cd, sg)
                dots.set_data(cd, sg)
                fig.suptitle("the non-dominated front", fontsize=15, color=INK)
            return ()

        line.set_data(cd, sg)
        dots.set_data(cd, sg)
        i, frac = walk[f - n_a - n_b]
        j = min(i + 1, len(stations) - 1)
        x = cd[i] + frac * (cd[j] - cd[i])
        y = sg[i] + frac * (sg[j] - sg[i])
        cursor.set_data([x], [y])

        st = stations[i if frac < 0.5 else j]
        draw_flow(st, "flow: pressure coefficient and streamlines")
        draw_em(st, STEALTH, "grey: every design on the front")
        fig.suptitle(f"$C_d$ = {st['p']['Cd']:.5f}      "
                     f"echo width = {st['p']['sigma_agg']:.5f} m      "
                     f"$C_l$ = {st['p']['Cl']:.4f}", fontsize=15, color=INK)
        return ()

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / fps)
    _write(anim, out, fps)
    plt.close(fig)



def mechanism(config, front, out, fps=20, steps=6):
    """Morph along the front: leading edge, scattered field, and the two costs."""
    geom, gi = _geom(config)
    k = k_chord(config)
    thetas = [np.asarray(p["theta"]) for p in front]
    cd = np.array([p["Cd"] for p in front])
    sg = np.array([p["sigma_agg"] for p in front])

    path = []
    for i in range(len(thetas) - 1):
        for s in range(steps):
            t = s / steps
            path.append((i, t))
    path.append((len(thetas) - 2, 1.0))
    path = path + path[::-1]

    gx, gy = np.meshgrid(np.linspace(-0.7, 1.9, 230), np.linspace(-0.9, 0.9, 160))

    print("precomputing morph frames", flush=True)
    frames = []
    for n, (i, t) in enumerate(path[: len(path) // 2]):
        th = (1 - t) * thetas[i] + t * thetas[i + 1]
        loop = surface(geom, gi, th)
        frames.append({
            "loop": loop,
            "R": le_radius(th),
            "cd": (1 - t) * cd[i] + t * cd[i + 1],
            "sg": (1 - t) * sg[i] + t * sg[i + 1],
            "E": em_field.near_field(loop[:-1], k, [1.0, 0.0], gx, gy),
        })
        if n % 10 == 0:
            print(f"  {n}/{len(path) // 2}", flush=True)
    frames = frames + frames[::-1]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.62],
                          left=0.05, right=0.97, top=0.90, bottom=0.09,
                          wspace=0.14, hspace=0.30)
    ax_e = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[0, 1])
    ax_s = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[1, 1])
    boxes = {name: ax.get_position().frozen()
             for name, ax in (("e", ax_e), ("l", ax_l), ("s", ax_s))}

    lim = max(abs(f["E"].real).max() for f in frames[: len(frames) // 2])
    ghost, ghost_r = frames[0]["loop"], frames[0]["R"]

    def update(f):
        d = frames[f]
        ax_s.clear()
        ax_s.plot(ghost[:, 0], ghost[:, 1], color=AERO, lw=1.4, alpha=0.75, zorder=2)
        ax_s.fill(d["loop"][:, 0], d["loop"][:, 1], color=INK, zorder=3)
        _style(ax_s, "section, against the low-drag design in red")
        fit_y(fig, ax_s, (-0.04, 1.04), boxes["s"])

        ax_l.clear()
        ax_l.plot(ghost[:, 0], ghost[:, 1], color=AERO, lw=1.4, alpha=0.75, zorder=2)
        ax_l.add_patch(Circle((ghost_r, 0.0), ghost_r, fill=False, color=AERO,
                              lw=1.4, alpha=0.6, ls="--", zorder=2))
        ax_l.fill(d["loop"][:, 0], d["loop"][:, 1], color=INK, zorder=3)
        ax_l.add_patch(Circle((d["R"], 0.0), d["R"], fill=False, color="white",
                              lw=2.6, zorder=4))
        _style(ax_l, f"leading edge radius {d['R']:.4f} c, "
                     f"was {ghost_r:.4f} c")
        fit_y(fig, ax_l, (-0.010, 0.062), boxes["l"])

        ax_e.clear()
        ax_e.pcolormesh(gx, gy, d["E"].real, cmap="RdBu_r", vmin=-lim, vmax=lim,
                        shading="auto", rasterized=True)
        ax_e.fill(d["loop"][:, 0], d["loop"][:, 1], color=INK, zorder=3)
        _style(ax_e, "total field, plane wave incident from the left")
        fit_y(fig, ax_e, (-0.6, 1.8), boxes["e"])

        ax_t.clear()
        ax_t.plot(sg, cd, "-", color="0.8", lw=2)
        ax_t.plot([d["sg"]], [d["cd"]], "o", color=AERO, ms=12)
        ax_t.invert_xaxis()
        ax_t.set_xlabel("radar echo width (m)", fontsize=10)
        ax_t.set_ylabel("drag coefficient $C_d$", fontsize=10)
        ax_t.grid(alpha=0.25)
        ax_t.tick_params(labelsize=9)
        ax_t.set_title("less radar return costs drag", fontsize=11, pad=5)

        fig.suptitle("a sharper nose hides from radar, and costs drag",
                     fontsize=16, color=INK)
        return ()

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    _write(anim, out, fps)
    plt.close(fig)


def _write(anim, out, fps):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out.with_suffix(".mp4"), writer="ffmpeg", fps=fps, dpi=100,
              savefig_kwargs={"facecolor": "white"})
    print(f"wrote {out.with_suffix('.mp4')}", flush=True)
    anim.save(out.with_suffix(".gif"), writer="pillow", fps=fps, dpi=60,
              savefig_kwargs={"facecolor": "white"})
    print(f"wrote {out.with_suffix('.gif')}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "configs/baseline.yaml")
    parser.add_argument("--pareto", type=Path, nargs="+",
                        default=[ROOT / "outputs/pareto.json",
                                 ROOT / "outputs/knee/pareto.json"])
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/figures")
    parser.add_argument("--which", choices=["tour", "mechanism", "both"], default="both")
    args = parser.parse_args()

    config = load_config(args.baseline)
    _, front, cloud = load_front(args.pareto)
    walks, featured = load_walks(args.pareto)
    print(f"{len(front)} front points, {len(cloud)} evaluated designs, "
          f"{len(walks)} optimizer runs")
    if args.which in ("tour", "both"):
        tour(config, front, walks, featured, args.out / "pareto_tour")
    if args.which in ("mechanism", "both"):
        mechanism(config, front, args.out / "pareto_mechanism")


if __name__ == "__main__":
    main()
