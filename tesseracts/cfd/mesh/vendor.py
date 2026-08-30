# SPDX-License-Identifier: Apache-2.0
"""Import the OpenFOAM adjoint tutorial NACA 0012 mesh as the reference grid.

The tutorial ships the mesh that adjointOptimisationFoam's own sensitivity
verification runs on: a C-grid around a chord-1 NACA 0012 with a 5e-4 first
cell, max non-orthogonality 26 and max skewness 0.20. Its two wall patches are
merged into one `airfoil` patch and its far field renamed, so the case template
sees the same patch names the rest of the pipeline uses.
"""

import argparse
import gzip
import re
import shutil
from pathlib import Path

MESH_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = MESH_DIR.parent / "case_template/constant/polyMesh"
TUTORIAL_MESH = (
    "tutorials/incompressible/adjointOptimisationFoam/resources/meshes/naca0012/polyMesh"
)
RENAME = {"frontBack": "frontBack", "inlet": "farfield"}
MERGE = ("pressure", "suction")


def _entries(text):
    body = text[text.index("(", text.index("\n(")) :]
    return re.findall(r"(\w+)\s*\{([^}]*)\}", body)


def merge_boundary(text):
    """Rename the far field and fuse the two contiguous wall patches into one."""
    kept, walls = [], []
    for name, block in _entries(text):
        n = int(re.search(r"nFaces\s+(\d+)", block).group(1))
        s = int(re.search(r"startFace\s+(\d+)", block).group(1))
        if name in MERGE:
            walls.append((s, n))
        else:
            kept.append((RENAME.get(name, name), block.strip()))
    walls.sort()
    if len(walls) != len(MERGE) or walls[0][0] + walls[0][1] != walls[1][0]:
        raise ValueError(f"wall patches are not contiguous: {walls}")
    kept.append((
        "airfoil",
        f"type            wall;\n        inGroups        1 ( wall );\n"
        f"        nFaces          {walls[0][1] + walls[1][1]};\n"
        f"        startFace       {walls[0][0]};",
    ))
    sep = text.index("\n", text.index("// * *"))
    blocks = "\n".join(f"    {n}\n    {{\n        {b}\n    }}\n" for n, b in kept)
    return f"{text[:sep]}\n\n{len(kept)}\n(\n{blocks})\n"


def vendor(foam_root, out=DEFAULT_OUT):
    src = Path(foam_root) / TUTORIAL_MESH
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in ("points", "faces", "owner", "neighbour"):
        shutil.copy(src / f"{name}.gz", out / f"{name}.gz")
    text = gzip.decompress((src / "boundary.gz").read_bytes()).decode()
    (out / "boundary").write_text(merge_boundary(text))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foam-root", type=Path,
                        default=Path.home() / "side-projects/openfoam")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(f"wrote {vendor(args.foam_root, args.out)}")


if __name__ == "__main__":
    main()
