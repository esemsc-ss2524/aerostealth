# driver

Composes the three Tesseracts into one differentiable program and runs the
epsilon-constraint sweep over it.

## Composition

`forward.build_forward(config)` returns

```
theta -> {Cd, Cl, sigma_agg, g_geom}
```

`geom` is the shared spine: its `x_surf` feeds both physics legs, so a cotangent
on either objective comes back through the same CST jacobian. `em` is JAX all
the way down and reverse-mode differentiates itself; `cfd` is an OpenFOAM
continuous adjoint behind a `vector_jacobian_product` endpoint. `tesseract-jax`
makes that difference invisible to `jax.grad`, which is the point of the
project.

`build_forward(config, aero=False)` drops the OpenFOAM leg and leaves the
all-JAX path: same geometry, same EM, seconds instead of minutes. The gradient
checks and the stealth anchor run on it.

Every `apply_tesseract` call asks for reverse mode explicitly
(`materialize_jacobian=False`). Left to itself the default prefers a Tesseract's
`jacobian` endpoint, and `geom`'s jacobian includes the 96 by 96 level set that
nothing downstream reads: one constraint jacobian goes from 1.5 seconds to more
than two minutes.

## Sweep

`optimize.run_sweep` solves

```
min Cd  s.t.  sigma_agg <= eps_k,  g_geom <= 0
```

at a schedule of `eps_k` between two anchors, each level warm-started from the
previous solution. `Cl = Cl*` is absent from the outer problem because the `cfd`
tesseract trims to it internally and reports the gradient at fixed lift.

The anchors come from the two ends: `min Cd` with the echo width unconstrained,
and `min sigma_agg` subject to geometry only. Both objectives are divided by the
baseline values in `configs/baseline.yaml`, since `Cd` and the echo width have
comparable magnitudes but shape sensitivities orders of magnitude apart.

`Evaluator` memoizes one forward and one reverse pass per design vector, because
nlopt asks for the objective and the constraints separately at the same point.
Only `Cd` needs OpenFOAM, so the aero leg is differentiated on its own and the
cheap leg supplies the constraint jacobian; no wasted adjoint runs.

## Design variables

`theta` is `[upper weights, lower weights]`, bounded so the upper stay positive
and the lower negative. That guarantees positive thickness everywhere, which
keeps the RBF mesh morph valid, while still leaving camber free through the
difference in magnitudes.

## Cost

The cheap leg is seconds. Each `Cd` value-and-gradient is an OpenFOAM trim plus
two adjoint solves plus two angle-of-attack primals, so the sweep budget is set
almost entirely by how many aero gradient calls the schedule allows. Keep
`optimizer.max_iter` small and lean on warm starts between epsilon levels.

Aero gradients carry the caveat in `../tesseracts/cfd/README.md`: the drag
adjoint disagrees with converged finite differences. The sweep uses it as-is.
