# aerostealth

Gradient-based RCS vs drag co-design. Four differentiable components composed
behind Tesseract's uniform `jvp`/`vjp` interface so that
`theta -> (Cd, Cl, sigma_agg, g_geom)` is one `jax.value_and_grad`-able program,
traced across a stealth-vs-aero Pareto front by an epsilon-constraint sweep.

- `geom` (JAX, reverse-mode autodiff): CST airfoil to boundary curve, differentiable
  rasterization, geometric constraints.
- `cfd` (OpenFOAM `adjointOptimisationFoam`): mesh morph, primal RANS, AoA trim,
  drag adjoint sensitivity.
- `em` (JAX method of moments): 2D EFIE on the PEC contour, monostatic echo width,
  KS-aggregated and differentiated by autodiff.
- `driver`: composes `geom -> {cfd, em}` with `tesseract-jax` and runs the sweep.

License: Apache-2.0.

## Environment

```
conda create -n aerostealth --clone ml
conda activate aerostealth
pip install tesseract-core tesseract-jax nlopt lz4
```

External solvers, sourced from the host:

- OpenFOAM ESI v2606 at `~/side-projects/openfoam` (`source ~/side-projects/openfoam/etc/bashrc`).
  Its adjoint-tutorial NACA 0012 mesh is the CFD reference grid, vendored by
  `tesseracts/cfd/mesh/vendor.py`.

## Layout

```
tesseracts/   geom/ cfd/ em/     one tesseract_api.py + tesseract_config.yaml each
driver/       optimize.py mgda.py objectives.py surrogate.py tesseracts.py
configs/      baseline.yaml sweep.yaml
analysis/     pareto.py plots.py
tests/        per-tesseract FD checks + end-to-end gradient check
paper/        writeup.md
```

## Run

```
pytest
python -m driver.optimize --baseline configs/baseline.yaml --sweep configs/sweep.yaml
```

The sweep writes one Pareto point per epsilon level with its convergence history.
