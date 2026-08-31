# cfd

`theta -> x_surf` (from `geom`) `-> Cd, Cl, Cm` at a fixed angle of attack, via
an OpenFOAM incompressible RANS primal on a morphed reference mesh.

## Flow

1. `runner.prepare_mesh` copies the vendored reference mesh (`mesh/vendor.py`,
   see `mesh/README.md`) and morphs it onto `x_surf` with `mesh/morph.py`
   (compact-support RBF, far field fixed, never remeshed).
2. `runner.run_primal` runs `simpleFoam` (Spalart-Allmaras, freestream
   boundaries) and reads `forceCoeffs`. It raises unless the forces are
   stationary, and caches the converged result beside the run so `apply` and the
   vjp share one solve.
3. `tesseract_api.apply` hashes `x_surf`, `alpha_deg` and `reynolds` to a run
   directory under `_run/` (gitignored) and returns `Cd, Cl, Cm`.

Angle of attack enters through the freestream direction and the force
directions, not the mesh; the section stays in the body frame.

## Adjoint (vector_jacobian_product)

`sensitivity.shape_sensitivity`:

1. `runner.run_adjoint` runs `adjointOptimisationFoam` (`singleRun`,
   differentiated Spalart-Allmaras) twice off the converged primal, once for
   drag and once for lift. Output is the point sensitivity on the airfoil.
2. `morph.morph_vjp` pulls each `dJ/dx` back through the linear RBF morph to
   `dJ/dx_surf`.

`vector_jacobian_product` returns `cot_Cd * dCd/dx_surf + cot_Cl * dCl/dx_surf`
(the moment cotangent is ignored). The driver chains this through `geom`'s VJP
to reach `dJ/dtheta`. Lift is a constraint in the outer problem, so there is no
trim and nothing here is finite-differenced.

Two details that are load-bearing:

- The sensitivity read is `pointSensNormalVecadjESI`, the normal-projected
  field, not the full vector `pointSensVecadjESI`. Only the normal component
  changes the shape; the raw vector is 45 to 60 percent tangential here, and the
  morph transpose cannot tell the two apart. With the normal field the whole
  geometry chain reproduces a finite difference of the morph to four figures;
  with the full vector it does not.
- `morph_vjp` builds its operator on the *reference* mesh points, not the
  case's own, which `prepare_mesh` has already displaced onto the target curve.
  It asserts the two files are in the same point order before using them.

`runner._check_adjoint` rejects a diverged run. `adjointOptimisationFoam` exits
0 and writes a full sensitivity field even when the adjoint mesh-movement solve
has run away, so the return code says nothing; healthy runs report `Max ma`
around 0.1, a runaway reports 1e80.

## Configuration

The primal and adjoint dictionaries are the ones from OpenFOAM's
`sensitivityMaps/naca0012/turbulent` tutorial, on that tutorial's own mesh.
One `system/fvSchemes` and one `system/fvSolution` serve both solvers, so the
adjoint cannot drift from the primal it is differentiating. Deviations from the
tutorial are deliberate and limited to:

- `residualControl` at 1e-8, primal and adjoint. At the tutorial's 1e-6 the
  primal stops with `Cd` still 4e-4 off its own converged value, drifting
  monotonically. `runner._converged` gates on stationary forces rather than on
  the residual banner, calibrated so a run that stops at 1e-6 fails and one that
  reaches 3e-6 relative passes.
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

Substitutions into the dictionaries match on the keyword and raise if they hit
nothing. Matching on an expected literal is how the force directions silently
stopped being rotated once a case was copied from an already-rotated one.

## Environment

Needs OpenFOAM on `PATH` with `adjointOptimisationFoam`. `runner` shells out
through `source $AEROSTEALTH_OF_BASHRC`.

## Status

Gradients are checked against central finite differences of the converged
primal on all twelve design variables by `analysis/gradient_check.py`
(`analysis/figures/cfd_adjoint_vs_fd.png`, `cfd_lift_adjoint_vs_fd.png`).

**Lift** agrees well: cosine 0.9904 against the finite-difference gradient, 7.9
degrees, with the magnitude a fairly uniform 0.8 of it. A uniform scale factor
costs a descent method nothing.

**Drag** does not: cosine 0.5007, 60 degrees, three of twelve signs flipped, and
a leading-edge lower-surface component 6.3x too large that dominates the norm.
It is still formally a descent direction and MMA does reduce `Cd` with it, at
about half efficiency.

The cause is that the drag adjoint never converges. Its pressure residual
plateaus near 6e-4 and burns the full 3000-iteration cap, while the lift adjoint
on the same primal converges on tolerance in 1829. Everything around it has been
ruled out: the geometry chain reproduces a finite difference of the morph to
1.0000, the primal reproduces its own converged value, tightening the adjoint
`residualControl` from 1e-6 to 1e-8 changes the ratios by nothing, and six ATC
variants leave `standard` the best of them (`cancel` is worse at 82.7 degrees,
`extraConvection 1` diverges outright). What remains is the viscous part of the
drag force and the grid sensitivity of the Spalding wall function. This is the
open item; the sweep uses the gradient as-is.

`tests/test_cfd_morph.py` covers the morph math, `tests/test_cfd_adjoint.py` the
morph transpose identity (fast) and both adjoints vs central FD (slow),
`tests/test_cfd_mesh.py` the own-grid generator and `blockMesh`,
`tests/test_cfd_primal.py` the full primal. The slow ones need
`AEROSTEALTH_SLOW_TESTS=1`.
