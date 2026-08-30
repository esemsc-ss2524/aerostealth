# cfd

`theta -> x_surf` (from `geom`) `-> Cd, Cl, Cm` at `Cl = cl_target`, via an
OpenFOAM incompressible RANS primal on a morphed reference mesh.

## Flow

1. `runner.prepare_mesh` copies the reference C-grid (built once by `blockMesh`
   from `case_template/system/blockMeshDict`) and morphs it onto `x_surf` with
   `mesh/morph.py` (compact-support RBF, far field fixed, never remeshed).
2. `runner.run_trim` runs `simpleFoam` (Spalart-Allmaras, freestream boundaries),
   reads `forceCoeffs`, and steps the angle of attack by a secant Newton
   iteration until `Cl` matches `cl_target`. Each iteration restarts from the
   previous angle's fields.
3. `tesseract_api.apply` hashes `x_surf` and `reynolds` to a run directory under
   `_run/` (gitignored) and returns `Cd, Cl, Cm`, the trimmed `alpha_deg`, and
   the trim iteration count.

Angle of attack enters through the freestream direction, not the mesh; the
section stays in the body frame.

## Environment

Needs OpenFOAM on `PATH`. `runner` shells out through
`source $AEROSTEALTH_OF_BASHRC` (default `~/side-projects/openfoam/etc/bashrc`).

## Status

The primal is wired end to end and reproduces the standalone case (morphing to
the baseline shape is a no-op to within 0.3 percent on `Cd` and `Cl`). Absolute
accuracy carries the biases noted in `mesh/README.md`; the mesh-independence and
domain study is deferred. The drag adjoint VJP (`vector_jacobian_product`) is not
yet implemented.

`tests/test_cfd_morph.py` covers the morph math. `tests/test_cfd_mesh.py` covers
the generator and `blockMesh`. `tests/test_cfd_primal.py` runs the full trimmed
primal and is opt in (`AEROSTEALTH_SLOW_TESTS=1`).
