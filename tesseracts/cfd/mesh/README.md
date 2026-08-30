# cfd reference mesh

`cgrid.py` writes `../case_template/system/blockMeshDict`: a structured C-grid
around a closed airfoil section, for wall-resolved incompressible RANS.

## Topology

Six hex blocks. Four wrap the section from its surface to a circular far field,
split at the trailing edge, an upper crown, the leading edge, and a lower crown.
Two carry the grid from the trailing edge to the outlet; the wake cut at `y = 0`
is an internal interface, not a boundary. Patches: `airfoil` (wall), `farfield`
(freestream, includes the outlet), `front` and `back` (empty).

## Conventions

- Chord 1, leading edge at the origin, section in the body frame (no angle of
  attack baked into the mesh). Angle of attack enters through the freestream
  direction in `0/U` and the `forceCoeffs` lift and drag directions.
- The surface enters as four spline edges from `naca0012()`, cosine-clustered at
  the leading and trailing edges. `n_wrap` cells span each quarter, so the
  surface carries `4 * n_wrap` cells.
- Radial grading packs the first cell to `first_cell` for a wall-resolved first
  layer. The wake blocks inherit that spacing on the trailing-edge spoke.
- 2D: one cell across a `span`-thick slab with empty end patches.

## Current parameters and validation

Defaults: `n_per_side=161`, `n_wrap=48`, `n_radial=128`, `n_wake=64`,
`farfield_radius=20`, `wake_length=25`, `first_cell=8e-6`. 40960 cells.
`checkMesh`: non-orthogonality max 62 (avg 18), skewness max 0.83, no negative
volumes. Aspect ratio flags ~3600 cells in the far wake where the internal cut
keeps a fine first layer next to coarse streamwise cells; these sit in a benign
low-gradient region and do not affect convergence.

NACA 0012, Re 1e6, Uinf 1, nu 1e-6, Spalart-Allmaras, freestream boundaries,
`simpleFoam` SIMPLEC to 3000 iterations. y+ below 0.5 everywhere.

| alpha | Cd      | Cl     | Cm     |
|-------|---------|--------|--------|
| 0     | 0.01469 | 0.000  | 0.000  |
| 4     | 0.01654 | 0.397  | +0.006 |
| 8     | 0.02462 | 0.764  | +0.014 |

Lift curve slope 0.099 per degree against a reference near 0.105 to 0.11; Cl 0.4
is reached near alpha 4.0 against a reference near 3.7. Cd at Cl 0.4 is about
0.0165 against a fully turbulent reference near 0.013. Trends are correct (zero
lift at zero incidence, linear lift, quadratic drag rise); absolute levels run
high, dominated by the compact `farfield_radius=20` and the 192-cell surface.

## Refinement backlog

- Grow `farfield_radius` toward 50 to 100 with `n_radial` raised so the per-cell
  growth stays near 8 percent; the extreme grading at `farfield_radius=50`,
  `n_radial=128` diverged.
- Raise `n_wrap` toward 90 to 120 for the surface pressure and skin friction.
- Split the wake blocks radially, or use `edgeGrading`, so the fine first layer
  does not persist to the outlet.
- Mesh-independence study once the above land.
