# em conventions

`x_surf` (from geom) `-> sigma_agg`, the KS-aggregated monostatic echo width over
a frontal incidence sector. Pure JAX, so the EM shape sensitivity is reverse-mode
autodiff through `mom.py` and `bessel.py` (no separate adjoint).

## Physics

- 2D method-of-moments EFIE, TM-z polarization (electric field along the span),
  perfect electric conductor. Pulse basis, point matching, standard
  small-argument log self term. Off-diagonal terms use the midpoint rule.
- Validated against the analytic PEC circular cylinder series (Abramowitz and
  Stegun): within about 0.05 percent for ka up to 10 at 200 segments.

## Frame and scaling

- The contour is `x_surf[:-1]` (the closed loop with its duplicate trailing-edge
  point dropped), scaled from chord 1 to `chord_m` metres. This is the "scaled
  body": at the defaults, `chord_m = 0.05` and `frequency_hz = 10 GHz` give
  `k * chord ~ 10.5`, a few wavelengths across, where the shape drives the RCS
  without the pattern breaking into fine lobes.
- `incidence_deg` is the angle of the incident propagation direction from the
  chord axis. The frontal sector is `[-30, 30]` degrees, `incidence_count`
  samples. Echo width is taken in the monostatic (backscatter) direction.

## Aggregation

`sigma_agg` is a Kreisselmeier-Steinhauser smooth maximum of the per-angle echo
widths, taken in the log domain so it is scale invariant and `ks_rho` means the
same thing at any RCS level. It is an upper bound on the sector peak; `ks_rho`
is raised over the sweep (loose for conditioning, tight for peak fidelity).

The driver normalizes `sigma_agg` by its baseline value before the optimizer
sees it.
