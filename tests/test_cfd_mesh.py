# SPDX-License-Identifier: Apache-2.0
"""The C-grid generator produces a blockMeshDict that blockMesh builds cleanly.

The CFD reference grid is the vendored tutorial mesh; this generator stays as the
own-grid path and is exercised on a throwaway case."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "tesseracts/cfd/mesh"
sys.path.insert(0, str(MESH_DIR))

import cgrid  # noqa: E402


def test_blockmeshdict_structure():
    text = cgrid.build_dict(n_per_side=41, n_wrap=12, n_wake=16, n_radial=24)
    assert text.count("hex (") == 6
    assert "airfoil" in text and "farfield" in text
    assert "spline" in text and "arc" in text


@pytest.mark.skipif(shutil.which("blockMesh") is None, reason="OpenFOAM not on PATH")
def test_blockmesh_builds(tmp_path):
    template = ROOT / "tesseracts/cfd/case_template"
    case = tmp_path / "case"
    shutil.copytree(template, case, ignore=shutil.ignore_patterns("polyMesh"))
    (case / "system/blockMeshDict").write_text(
        cgrid.build_dict(n_per_side=81, n_wrap=24, n_wake=32, n_radial=48)
    )

    build = subprocess.run(
        ["blockMesh", "-case", str(case)], capture_output=True, text=True
    )
    assert build.returncode == 0, build.stderr[-2000:]

    check = subprocess.run(
        ["checkMesh", "-case", str(case), "-noTopology"], capture_output=True, text=True
    )
    out = check.stdout
    assert "negative volume" not in out.lower()
    assert "incorrectly oriented" not in out.lower()
