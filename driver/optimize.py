# SPDX-License-Identifier: Apache-2.0
"""Epsilon-constraint sweep: min Cd s.t. sigma_agg <= eps_k, Cl = Cl*, g_geom <= 0."""

import argparse
from pathlib import Path

from driver.tesseracts import load_config


def build_forward(baseline: dict):
    """Compose geom -> {cfd, em} into a single jax.value_and_grad-able map."""
    raise NotImplementedError


def solve_subproblem(forward, eps, x0, sweep: dict):
    """One epsilon level, warm-started from x0. Returns (x, Cd, sigma_agg, history)."""
    raise NotImplementedError


def run_sweep(baseline: dict, sweep: dict):
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default="configs/baseline.yaml")
    parser.add_argument("--sweep", type=Path, default="configs/sweep.yaml")
    args = parser.parse_args()
    run_sweep(load_config(args.baseline), load_config(args.sweep))


if __name__ == "__main__":
    main()
