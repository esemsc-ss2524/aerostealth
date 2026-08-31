# SPDX-License-Identifier: Apache-2.0
"""Compose geom -> {cfd, em} into one differentiable map from the design vector.

`geom` is the shared spine: its `x_surf` feeds both physics legs, so a cotangent
on either objective flows back through the same CST jacobian. `em` is JAX all the
way down and `cfd` is an OpenFOAM adjoint behind a vjp endpoint; `tesseract-jax`
makes the difference invisible to `jax.grad`.
"""

import jax.numpy as jnp
from tesseract_jax import apply_tesseract

from driver.tesseracts import local_tesseract

# Reverse mode only. Left to itself `apply_tesseract` prefers a Tesseract's
# `jacobian` endpoint, which for `geom` materializes the whole 96x96 level-set
# jacobian that nothing downstream reads; one constraint jacobian then costs
# minutes instead of seconds.
_VJP = {"materialize_jacobian": False, "vmap_method": "sequential"}


def geom_inputs(config):
    c = config["geometry"]["constraints"]
    return {
        "n_surface": int(config["geometry"]["n_surface"]),
        "thickness_min": float(c["thickness_min"]),
        "thickness_max": float(c["thickness_max"]),
        "trailing_edge_gap_max": float(c["trailing_edge_gap_max"]),
        "enclosed_area_min": float(c["enclosed_area_min"]),
    }


def cfd_inputs(config):
    a = config["aero"]
    return {"alpha_deg": float(a["alpha_deg"]), "reynolds": float(a["reynolds"])}


def em_inputs(config):
    e = config["em"]
    lo, hi = e["incidence_deg"]
    return {
        "frequency_hz": float(e["frequency_hz"]),
        "chord_m": float(e["chord_m"]),
        "incidence_deg_min": float(lo),
        "incidence_deg_max": float(hi),
        "incidence_count": int(e["incidence_count"]),
        "ks_rho": float(e["ks_rho"]),
    }


def build_forward(config, aero=True):
    """theta -> {sigma_agg, g_geom} and, with aero, {Cd, Cl}.

    Set aero False for the all-JAX path: same geometry spine and same EM leg,
    but no OpenFOAM, so gradient checks run in seconds instead of minutes.
    """
    geom, em = local_tesseract("geom"), local_tesseract("em")
    cfd = local_tesseract("cfd") if aero else None
    gi, ei, ci = geom_inputs(config), em_inputs(config), cfd_inputs(config)

    def forward(theta):
        g = apply_tesseract(geom, {"theta": theta, **gi}, **_VJP)
        out = {
            "sigma_agg": apply_tesseract(em, {"x_surf": g["x_surf"], **ei}, **_VJP)["sigma_agg"],
            "g_geom": g["g_geom"],
        }
        if cfd is not None:
            a = apply_tesseract(cfd, {"x_surf": g["x_surf"], **ci}, **_VJP)
            out["Cd"], out["Cl"] = a["Cd"], a["Cl"]
        return out

    return forward


def baseline_theta(config):
    """Symmetric CST weights that approximate the baseline section."""
    n = int(config["geometry"]["cst_coeffs"]) // 2
    w = jnp.full((n,), float(config["geometry"]["cst_weight_init"]))
    return jnp.concatenate([w, -w])
