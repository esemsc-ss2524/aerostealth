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
min Cd  s.t.  sigma_agg <= eps_k,  Cl >= Cl*,  g_geom <= 0
```

at a fixed angle of attack, over a schedule of `eps_k` between two anchors.
Lift is a genuine constraint carried by the optimizer and its gradient is the
lift adjoint, so no quantity in the problem is finite-differenced.

The anchors come from the two ends: `min Cd` with the echo width unconstrained,
and `min sigma_agg` subject to geometry only. Each intermediate level is
warm-started from the two anchor design vectors blended by where its `eps_k`
sits between them, which starts every level near its own solution rather than
marching in from a neighbour. Both objectives are divided by the baseline values
in `configs/baseline.yaml`, and the constraint rows by their own bounds, so the
optimizer sees a problem of order one throughout.

The optimizer is MMA. It asks for a gradient on every call, so no OpenFOAM
solve is spent on a line-search probe that gets discarded; SLSQP asks for a
value without a gradient roughly seventy percent of the time. A design the aero
leg will not solve or differentiate shrinks the move limit and restarts the
level rather than being papered over with a fabricated gradient.

`Evaluator` memoizes per design vector, and skips the adjoints entirely
(`grad=False`) for a point that is being reported rather than stepped from.

Levels run in separate processes, not threads: concurrent `apply_tesseract`
calls from several threads of one interpreter abort the process without a
traceback. Set `optimizer.jobs` to the number of physical cores. Throughput
scales sublinearly because the OpenFOAM linear solvers are memory-bandwidth
bound.

## Design variables

`theta` is `[upper weights, lower weights]`, bounded so the upper stay positive
and the lower negative. That keeps the surfaces apart in practice and the RBF
mesh morph valid, but it is a box, not a guarantee: `g_geom` bounds the
*maximum* thickness, so nothing forbids the surfaces crossing at some chordwise
station. Widening the box wants a pointwise thickness constraint first.

## Cost

The cheap leg is seconds. Each `Cd` value-and-gradient is one converged primal
plus two adjoint solves, and both coefficients share them, so asking for Cd and
Cl costs what asking for either would. The sweep budget is set almost entirely
by how many aero gradient calls the schedule allows.

Aero gradients carry the caveat in `../tesseracts/cfd/README.md`: the lift
adjoint agrees with finite differences to 7.9 degrees, the drag adjoint only to
60. Both are used as-is.
