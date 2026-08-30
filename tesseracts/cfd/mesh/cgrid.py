# SPDX-License-Identifier: Apache-2.0
"""Generate a blockMeshDict for a structured C-grid around a closed airfoil section.

Four blocks wrap the section from its surface to a semicircular far field; two
wake blocks carry the grid downstream to the outlet with the wake cut kept
internal. Radial grading packs the first cell to a target wall spacing for
wall-resolved RANS.
"""

import argparse
from pathlib import Path

import numpy as np

TEMPLATE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = TEMPLATE_DIR.parent / "case_template/system/blockMeshDict"


def naca0012(n_per_side, closed_te=True):
    """Closed NACA 0012 loop, Selig order (trailing edge over the upper surface to
    the leading edge and back). Length 2 * n_per_side - 1."""
    tail = -0.1036 if closed_te else -0.1015
    beta = np.linspace(0.0, np.pi, n_per_side)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 0.6 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 + tail * x**4
    )
    upper = np.column_stack([x[::-1], yt[::-1]])
    lower = np.column_stack([x[1:], -yt[1:]])
    return np.vstack([upper, lower])


def _radial_ratio(length, first_cell, n_cells):
    """blockMesh expansion ratio (last cell / first cell) for a geometric spacing
    that starts at first_cell and sums to length over n_cells."""
    target = length / first_cell
    lo, hi = 1.0 + 1e-9, 3.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        total = (mid**n_cells - 1.0) / (mid - 1.0)
        if total < target:
            lo = mid
        else:
            hi = mid
    growth = 0.5 * (lo + hi)
    return growth ** (n_cells - 1)


def build_dict(
    n_per_side=161,
    n_wrap=48,
    n_wake=64,
    n_radial=128,
    farfield_radius=20.0,
    wake_length=25.0,
    first_cell=8.0e-6,
    wake_ratio=300.0,
    span=0.05,
):
    loop = naca0012(n_per_side)
    m = loop.shape[0] - 1
    if m % 4 != 0:
        raise ValueError(f"need a multiple of 4 angular stations, got {m}")
    q = m // 4
    te, umid, le, lmid = 0, q, 2 * q, 3 * q

    r = farfield_radius
    xout = wake_length
    hz = 0.5 * span

    pts2d = {
        "te": (1.0, 0.0),
        "le": (0.0, 0.0),
        "umid": tuple(loop[umid]),
        "lmid": tuple(loop[lmid]),
        "o_te_top": (1.0, r),
        "o_le_top": (0.0, r),
        "o_up": (-r, 0.0),
        "o_le_bot": (0.0, -r),
        "o_te_bot": (1.0, -r),
        "out_top": (xout, r),
        "out_mid": (xout, 0.0),
        "out_bot": (xout, -r),
    }
    names = list(pts2d)
    idx = {name: i for i, name in enumerate(names)}
    n = len(names)

    verts = []
    for z in (-hz, hz):
        for name in names:
            x, y = pts2d[name]
            verts.append((x, y, z))

    def v(name, top=False):
        return idx[name] + (n if top else 0)

    RR, WR = _radial_ratio(r, first_cell, n_radial), wake_ratio

    # Each block: v0..v3 of the z- face, the (x, y) cell counts along local x
    # (v0->v1) and local y (v0->v3), and the (gx, gy) expansion ratios.
    blocks = [
        ("te", "o_te_top", "o_le_top", "umid", n_radial, n_wrap, RR, 1.0),
        ("umid", "o_le_top", "o_up", "le", n_radial, n_wrap, RR, 1.0),
        ("le", "o_up", "o_le_bot", "lmid", n_radial, n_wrap, RR, 1.0),
        ("lmid", "o_le_bot", "o_te_bot", "te", n_radial, n_wrap, RR, 1.0),
        ("te", "out_mid", "out_top", "o_te_top", n_wake, n_radial, WR, RR),
        ("te", "o_te_bot", "out_bot", "out_mid", n_radial, n_wake, RR, WR),
    ]

    block_lines = []
    for a, b, c, d, nx, ny, gx, gy in blocks:
        vs = [v(a), v(b), v(c), v(d), v(a, True), v(b, True), v(c, True), v(d, True)]
        block_lines.append(
            f"    hex ({' '.join(map(str, vs))}) ({nx} {ny} 1) "
            f"simpleGrading ({gx:.6g} {gy:.6g} 1)"
        )

    def spline(v0, v1, chain, dz):
        body = " ".join(f"({p[0]:.10g} {p[1]:.10g} {dz:.10g})" for p in chain)
        return f"    spline {v0} {v1} ({body})"

    def arc(v0, v1, angle_deg, dz):
        a = np.radians(angle_deg)
        return f"    arc {v0} {v1} ({r * np.cos(a):.10g} {r * np.sin(a):.10g} {dz:.10g})"

    surf = {
        ("te", "umid"): loop[te + 1 : umid],
        ("umid", "le"): loop[umid + 1 : le],
        ("le", "lmid"): loop[le + 1 : lmid],
        ("lmid", "te"): loop[lmid + 1 : m],
    }
    edge_lines = []
    for top in (False, True):
        dz = hz if top else -hz
        for (v0name, v1name), chain in surf.items():
            edge_lines.append(spline(v(v0name, top), v(v1name, top), chain, dz))
        edge_lines.append(arc(v("o_le_top", top), v("o_up", top), 135.0, dz))
        edge_lines.append(arc(v("o_up", top), v("o_le_bot", top), 225.0, dz))

    def quad(a, b, c, d):
        return f"({a} {b} {c} {d})"

    airfoil = [
        quad(v("te"), v("umid"), v("umid", True), v("te", True)),
        quad(v("umid"), v("le"), v("le", True), v("umid", True)),
        quad(v("le"), v("lmid"), v("lmid", True), v("le", True)),
        quad(v("lmid"), v("te"), v("te", True), v("lmid", True)),
    ]
    farfield = [
        quad(v("o_te_top"), v("o_le_top"), v("o_le_top", True), v("o_te_top", True)),
        quad(v("o_le_top"), v("o_up"), v("o_up", True), v("o_le_top", True)),
        quad(v("o_up"), v("o_le_bot"), v("o_le_bot", True), v("o_up", True)),
        quad(v("o_le_bot"), v("o_te_bot"), v("o_te_bot", True), v("o_le_bot", True)),
        quad(v("o_te_top"), v("out_top"), v("out_top", True), v("o_te_top", True)),
        quad(v("out_bot"), v("o_te_bot"), v("o_te_bot", True), v("out_bot", True)),
        quad(v("out_top"), v("out_mid"), v("out_mid", True), v("out_top", True)),
        quad(v("out_mid"), v("out_bot"), v("out_bot", True), v("out_mid", True)),
    ]

    def zface(top):
        ring = [b[:4] for b in blocks]
        return [quad(*[v(name, top) for name in face]) for face in ring]

    vert_lines = [f"    ({x:.10g} {y:.10g} {z:.10g})" for x, y, z in verts]

    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| generated by tesseracts/cfd/mesh/cgrid.py                                    |
\\*-------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale 1;

vertices
(
{chr(10).join(vert_lines)}
);

blocks
(
{chr(10).join(block_lines)}
);

edges
(
{chr(10).join(edge_lines)}
);

boundary
(
    airfoil
    {{
        type wall;
        faces ({" ".join(airfoil)});
    }}
    farfield
    {{
        type patch;
        faces ({" ".join(farfield)});
    }}
    front
    {{
        type empty;
        faces ({" ".join(zface(True))});
    }}
    back
    {{
        type empty;
        faces ({" ".join(zface(False))});
    }}
);
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-per-side", type=int, default=161)
    parser.add_argument("--n-radial", type=int, default=128)
    parser.add_argument("--farfield-radius", type=float, default=20.0)
    parser.add_argument("--wake-length", type=float, default=25.0)
    parser.add_argument("--first-cell", type=float, default=8.0e-6)
    args = parser.parse_args()
    text = build_dict(
        n_per_side=args.n_per_side,
        n_radial=args.n_radial,
        farfield_radius=args.farfield_radius,
        wake_length=args.wake_length,
        first_cell=args.first_cell,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
