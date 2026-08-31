# SPDX-License-Identifier: Apache-2.0
"""Measured solver wall times, scraped from the OpenFOAM logs left by a sweep.

`ExecutionTime` is cumulative, so the last one in a log is that solve's total.
"""

import re
from pathlib import Path

import numpy as np

EXEC_TIME = re.compile(r"ExecutionTime = ([0-9.]+)")

LOGS = {
    "primal": "primal/log.simpleFoam",
    "drag adjoint": "adj/drag/log.adjoint",
    "lift adjoint": "adj/lift/log.adjoint",
}


def _seconds(path):
    hits = EXEC_TIME.findall(Path(path).read_text())
    return float(hits[-1]) if hits else None


def solver_times(run_root):
    """Per-design seconds for each solve type, over every design in run_root."""
    out = {name: [] for name in LOGS}
    for design in sorted(Path(run_root).iterdir()):
        if not design.is_dir() or design.name == "reference":
            continue
        for name, rel in LOGS.items():
            log = design / rel
            if log.exists():
                seconds = _seconds(log)
                if seconds is not None:
                    out[name].append(seconds)
    return {k: np.array(v) for k, v in out.items() if v}


def gradient_times(run_root):
    """Seconds for one adjoint gradient, over designs that have all three solves."""
    totals = []
    for design in sorted(Path(run_root).iterdir()):
        if not design.is_dir() or design.name == "reference":
            continue
        parts = [_seconds(design / rel) for rel in LOGS.values()
                 if (design / rel).exists()]
        if len(parts) == len(LOGS) and all(p is not None for p in parts):
            totals.append(sum(parts))
    return np.array(totals)
