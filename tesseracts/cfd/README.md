# cfd

`theta -> x_surf` (from `geom`) `-> Cd, Cl, Cm` at `Cl = cl_target`, via an
OpenFOAM incompressible RANS primal on a morphed reference mesh.

## Flow

1. `runner.prepare_mesh` copies the vendored reference mesh (`mesh/vendor.py`,
   see `mesh/README.md`) and morphs it onto `x_surf` with `mesh/morph.py`
   (compact-support RBF, far field fixed, never remeshed).
2. `runner.run_trim` runs `simpleFoam` (Spalart-Allmaras, freestream
   boundaries), reads `forceCoeffs`, and steps the angle of attack by a secant
   Newton iteration until `Cl` matches `cl_target`. Each iteration restarts from
   the previous angle's fields.
3. `tesseract_api.apply` hashes `x_surf` and `reynolds` to a run directory under
   `_run/` (gitignored) and returns `Cd, Cl, Cm`, the trimmed `alpha_deg`, and
   the trim iteration count.

Angle of attack enters through the freestream direction, not the mesh; the
section stays in the body frame.

## Adjoint (vector_jacobian_product)

`sensitivity.trimmed_sensitivity` assembles the gradient the optimizer needs:

1. `runner.run_adjoint` runs `adjointOptimisationFoam` (`singleRun`,
   differentiated Spalart-Allmaras) twice off the trimmed primal, once for the
   drag objective and once for lift, each restarted from the converged fields.
   Output is the point sensitivity vector `dJ/dx` on the airfoil.
2. `runner.alpha_derivatives` central-differences `dCd/dalpha` and `dCl/dalpha`
   with two warm-restarted primals.
3. `morph.morph_vjp` pulls each `dJ/dx` back through the linear RBF morph to
   `dJ/dx_surf`.
4. With the inner trim holding `Cl` at `cl_target`,
   `dCd/dx|_trim = dCd/dx - (dCd/dalpha / dCl/dalpha) dCl/dx`.

`vector_jacobian_product` returns `cot_Cd * dCd/dx|_trim + cot_Cl * dCl/dx`
(the moment cotangent is ignored). The driver chains this through `geom`'s VJP
to reach `dCd/dtheta`.

## Configuration

The primal and adjoint dictionaries are the ones from OpenFOAM's
`sensitivityMaps/naca0012/turbulent` tutorial, on that tutorial's own mesh.
One `system/fvSchemes` and one `system/fvSolution` serve both solvers, so the
adjoint cannot drift from the primal it is differentiating. Deviations from the
tutorial are deliberate and limited to:

- `SIMPLE/residualControl` in `fvSolution`, which `simpleFoam` needs for the
  trim runs and `adjointOptimisationFoam` ignores (it takes its own from
  `optimisationDict`).
- `includeSurfaceArea true` in `optimisationDict`. The tutorial leaves it
  `false`, which divides the sensitivity by the point dual area to produce a
  map for plotting; the chain rule needs the undivided `dJ/dx`.
- `nNonOrthogonalCorrectors 2` in both `solutionControls` blocks. Note that
  `adjointOptimisationFoam` reads this from `optimisationDict`, not from
  `fvSolution/SIMPLE`: `solverControl::solutionDict()` returns
  `solverDict().subDict("solutionControls")`, so the `fvSolution` entry that
  `simpleFoam` honours is inert here. The tutorial's zero leaves the drag
  adjoint's pressure residual stalled at 3.3e-3; 1, 2 and 4 correctors reach
  1.2e-4, 1.3e-5 and 3.6e-7. Two is the useful compromise, since the
  sensitivity itself stops moving after one.
- The objective patch, direction and reference values, which the runner
  substitutes per drag or lift adjoint.

Freestream speed is 60 and chord is 1, so `nu = 60 / Re`; the design point
`Re = 6e6` reproduces the tutorial's `nu = 1e-5`.

## Environment

Needs OpenFOAM on `PATH`. `runner` shells out through
`source $AEROSTEALTH_OF_BASHRC` (default `~/side-projects/openfoam/etc/bashrc`).

## Status

Primal and adjoint both converge on the reference mesh and on RBF-morphed CST
sections of it, the adjoint to residuals near 1e-8 with the differentiated
Spalart-Allmaras model.

Sensitivities are checked against central finite differences of the converged
primal on gaussian shape bumps at six chordwise stations
(`analysis/figures/cfd_adjoint_vs_fd.png`).

For the **lift** objective, whose adjoint converges to 1e-8, sign and shape
agree everywhere and the adjoint reads 0.68 to 0.92 of the finite difference.
That deficit is not a numerical artefact: it is unchanged by the finite
difference step (2e-3 down to 2.5e-4), by the RBF support radius (0.3 to 4.0,
which is expected since the sensitivity is nonzero only on boundary points), by
`includeDistance` (2 percent), and by the ATC formulation (`standard` 0.81,
`UaGradU` 0.78, `cancel` 0.54). `includeMeshMovement` is not optional: with it
off the formulation changes to `SI` and the ratios scatter over 1.3 to 22.
What remains is the continuous-adjoint discretization gap, which closes under
mesh refinement rather than under any dictionary setting.

For the **drag** objective the sensitivity scatters: 0.60, 0.40, 0.70, 1.46,
-1.35, 1.35 against the same finite differences. Two things were ruled out.
Adjoint convergence is not the cause: the pressure residual stalls near 3e-3
with no corrector, but 1, 2 and 4 correctors take it to 1.2e-4, 1.3e-5 and
3.6e-7, and the ratios are identical to three figures throughout (0.59, 0.30,
0.55, 1.50, -1.30, 1.17), so the adjoint has converged. Nor is the finite
difference at fault: at the
station where the two disagree in sign it reads 2.094, 2.144, 2.161 and 2.137
for steps 4e-3, 2e-3, 1e-3 and 5e-4, against an adjoint value of -2.91.

So the converged drag adjoint disagrees with a converged finite difference, on
the tutorial's own mesh and configuration. The lift objective on the same run
does not, which points at the viscous part of the drag force and the grid
sensitivity of the Spalding wall function rather than at anything in the
pipeline around it. This is the open item; the trimmed gradient needs it.

`tests/test_cfd_morph.py` covers the morph math, `tests/test_cfd_adjoint.py` the
morph transpose identity (fast) and the drag adjoint vs central FD (slow),
`tests/test_cfd_mesh.py` the own-grid generator and `blockMesh`,
`tests/test_cfd_primal.py` the full trimmed primal. The slow ones need
`AEROSTEALTH_SLOW_TESTS=1`.
