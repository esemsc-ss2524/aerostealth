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

## Adjoint (vector_jacobian_product)

`sensitivity.trimmed_sensitivity` assembles the gradient the optimizer needs:

1. `runner.run_adjoint` runs `adjointOptimisationFoam` (`singleRun`, differentiated
   Spalart-Allmaras) twice off the trimmed primal, once for the drag objective and
   once for lift, each restarted from the converged fields. Output is the
   normal-projected point sensitivity `dJ/dn` on the airfoil.
2. `runner.alpha_derivatives` central-differences `dCd/dalpha` and `dCl/dalpha`
   with two warm-restarted primals.
3. `morph.patch_point_normals` and `morph.morph_vjp` turn each `dJ/dn` into a
   surface cotangent and pull it back through the linear RBF morph to `dJ/dx_surf`.
4. With the inner trim holding `Cl` at `cl_target`,
   `dCd/dx|_trim = dCd/dx - (dCd/dalpha / dCl/dalpha) dCl/dx`.

`vector_jacobian_product` returns `cot_Cd * dCd/dx|_trim + cot_Cl * dCl/dx`
(the moment cotangent is ignored). The driver chains this through `geom`'s VJP
to reach `dCd/dtheta`.

## Environment

Needs OpenFOAM on `PATH`. `runner` shells out through
`source $AEROSTEALTH_OF_BASHRC` (default `~/side-projects/openfoam/etc/bashrc`).

## Status

The primal is wired end to end and reproduces the standalone case (morphing to
the baseline shape is a no-op to within 0.3 percent on `Cd` and `Cl`). Absolute
accuracy carries the biases noted in `mesh/README.md`; the mesh-independence and
domain study is deferred.

The adjoint runs and both objectives produce point sensitivities. The
adjoint's own primal re-solve uses gentler relaxation and the
advection-diffusion wall distance, so it settles a few percent off the
`simpleFoam` primal; making the two fully consistent, and tightening the adjoint
past its ~1e-4 residual plateau, is open work. The adjoint fields are cheap once
the primal is warm (a couple of minutes for both objectives).

`tests/test_cfd_morph.py` covers the morph math, `tests/test_cfd_adjoint.py` the
morph transpose identity (fast) and the drag adjoint vs central FD (slow),
`tests/test_cfd_mesh.py` the generator and `blockMesh`, `tests/test_cfd_primal.py`
the full trimmed primal. The slow ones need `AEROSTEALTH_SLOW_TESTS=1`.
