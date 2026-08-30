# geom conventions

Settled interface that `cfd` and `em` both consume. Change here only with a
matching update to both adapters.

## Parameterization

- `theta` is a flat vector of length `2 * n`, ordered `[upper_0 .. upper_{n-1},
  lower_0 .. lower_{n-1}]`. Each half is a set of Bernstein weights for a CST
  (Kulfan) shape function with class exponents $N_1 = 0.5$ (round leading edge),
  $N_2 = 1.0$ (sharp trailing edge).
- Surface height: $y(\psi) = \psi^{N_1}(1-\psi)^{N_2}\sum_k w_k B_k(\psi)
  + \psi\, t_\mathrm{te}/2$, with $B_k$ the degree $n-1$ Bernstein basis.
- `te_thickness` adds a finite trailing-edge gap, split evenly between the
  surfaces. Zero by default (sharp trailing edge).

## Frame and orientation

- Chord 1, leading edge at $(0, 0)$, trailing edge at $(1, 0)$. Coordinates are
  dimensionless (chord units). No angle of attack is applied here: `cfd` rotates
  the inflow and `em` rotates the incidence sector, both in this body frame.
- Chordwise stations use cosine spacing, clustered at both edges.
- `x_surf` is a closed loop in Selig order: start at the trailing edge, run
  forward over the upper surface to the leading edge, then aft along the lower
  surface back to the trailing edge. Shape `(2 * n_surface - 1, 2)`; the shared
  leading-edge point appears once, the first and last points are the trailing
  edge.

## Rasterization

- `level_set` is a signed distance field on an `(raster_ny, raster_nx)` grid
  over `raster_bbox = (xmin, xmax, ymin, ymax)`, in the same units as `x_surf`.
- Sign convention: negative inside the airfoil, positive outside. Magnitude is
  the exact Euclidean distance to the boundary polygon. `em` maps this to a
  permittivity field with its own smoothing; the raw field is the common
  intermediate.

## Constraints

`g_geom` is a length-4 vector, all entries constrained `<= 0`:

1. `thickness_min - max_thickness`
2. `max_thickness - thickness_max`
3. `abs(te_gap) - trailing_edge_gap_max`
4. `enclosed_area_min - enclosed_area`

`max_thickness` is the maximum of $y_\mathrm{upper} - y_\mathrm{lower}$ over the
chordwise stations. Enclosed area is the shoelace area of `x_surf`.
