# cfd reference mesh

The reference grid is OpenFOAM's own adjoint-tutorial NACA 0012 mesh, vendored
into `../case_template/constant/polyMesh` by `vendor.py`. It is the grid
`adjointOptimisationFoam`'s `sensitivityMaps/naca0012/turbulent` tutorial runs
on, so the differentiated Spalart-Allmaras adjoint is known to converge on it.

`cgrid.py` writes an own-grid `blockMeshDict` for the same section. It is no
longer on the CFD path; `naca0012()` still supplies the reference loop and the
generator is kept for grid-independence work.

## Vendored mesh

C-grid, chord 1, leading edge at the origin, section in the body frame. 37800
cells: 700 around (398 on the airfoil, the rest in the wake) by 54 radial, first
cell 5e-4, far field at 15 chords, span 1 with empty end patches.

`vendor.py` renames `inlet` to `farfield` and fuses the tutorial's contiguous
`pressure` and `suction` wall patches into a single `airfoil` patch, so the case
template and the morph see the patch names the rest of the pipeline uses.

`checkMesh`: non-orthogonality max 26 (avg 8.4), skewness max 0.20, aspect ratio
max 869, no failed checks. After the RBF morph onto a CST section the numbers
move to 25.8 / 8.4 / 0.19, so the morph costs essentially nothing in quality.

## Conventions

- Angle of attack enters through the freestream direction in `0/U` and the
  `forceCoeffs` lift and drag directions, never through the mesh.
- Freestream speed 60, chord 1, so `nu = 60 / Re`; at the design `Re = 6e6` that
  is the tutorial's `nu = 1e-5` and a first-cell y+ around 30, which is what the
  `nutUSpaldingWallFunction` treatment is sized for.
- `morph.py` moves only mesh points inside the RBF support radius; the far field
  is fixed and the mesh is never rebuilt.

## Own-grid backlog

`cgrid.py`'s six-block C-grid reaches non-orthogonality 62 at the leading-edge
wrap-block corner. The primal tolerates it at `Re = 1e6`; neither the primal at
`Re = 6e6` nor the adjoint does, and non-orthogonal correctors do not help.
Making it usable needs a better leading-edge block layout (an extra block, an
O-grid nose, or elliptic smoothing).
