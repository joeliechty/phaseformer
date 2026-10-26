# Skate-Model Motion Planner on Theseus

Plans a first-order nonholonomic ("skate") robot through a 2D obstacle field by solving a
nonlinear least-squares trajectory optimization with
[Theseus](https://github.com/facebookresearch/theseus).

## The model

State is an SE(2) pose, input is `u = (v, ω)`:

```
ẋ = v·cos θ      ẏ = v·sin θ      θ̇ = ω
```

subject to the no-side-slip constraint `ẋ·sin θ − ẏ·cos θ = 0`.

The whole design turns on one observation. Written as a **body-frame twist**, that model is

```
ξ = [v, 0, ω]
         ↑
   the literal zero IS the skate constraint
```

Because the lateral slot is a hard-coded zero rather than a decision variable, side-slip is
impossible **by construction** — there is no penalty weight to tune and no residual slip. That is
the difference between this and the usual formulation, which carries a lateral velocity variable
and merely penalizes it (available here as `twist_mode="penalty"`, see below).

Integration is exact, not an Euler step: holding `(v, ω)` constant over `dt` traces a circular
arc, and `pose.retract(dt·ξ) == pose ∘ Exp(dt·ξ)` is precisely that arc. The dynamics residual
vanishes to ~1e-16 on an exactly-integrated pair.

## The factor graph

`N+1` SE(2) poses and `N` controls, solved with Levenberg–Marquardt and a dense Cholesky solver.

| Factor | Source | dim | Error |
|---|---|---|---|
| `SkateDynamics` | **new** | 3 | `Log(Tᵢ⁻¹Tᵢ₊₁) − dt·[v, 0, ω]` |
| Collision | `th.eb.Collision2D` | 1 | `clamp(cost_eps − sdf(xyᵢ), min=0)` |
| Control limits | `th.eb.HingeCost` | 2 | box on `\|v\|`, `\|ω\|` |
| Smoothness | `th.Between` | 2 | `uᵢ₊₁ − uᵢ` |
| Effort | `th.Difference` | 2 | `uᵢ` |
| Start / goal | `th.Difference` | 3 | `Log(T*⁻¹T)`, weight 1e4 |

`SkateDynamics` is the only new cost function; everything else is reused from Theseus. Its row 1
compares how far the robot moved *sideways* against a commanded lateral velocity of zero — that
row is what forbids slip. Its jacobians are analytic and verified against autograd by
`theseus.utils.check_jacobians`.

## Install

```bash
conda env create -f environment.yml
conda activate fm_bc
pytest tests/ -v
```

Two constraints drive the whole environment, both worth knowing before you change it:

- **Python must be 3.10.** `theseus-ai` 0.2.3 publishes wheels for cp38/cp39/cp310 only. On 3.11+
  pip falls back to the sdist, which refuses to build without torch already installed and then
  fails compiling `scikit-sparse` against a missing `cholmod.h`.
- **numpy must stay below 2**, because torch 2.1.2 predates the NumPy 2 ABI break.

Also note `torchkin` imports `urdf_parser_py` without declaring it — `import theseus` fails on a
clean environment without it. `environment.yml` adds it explicitly.

> If `pytest` fails to start with a `dash` plugin import error, packages in
> `~/.local/lib/python3.10/site-packages` are leaking into the env. Run with
> `PYTHONNOUSERSITE=1`.

## Use

```python
from skate_planner import SkatePlanner, scenario, plot_solution

world, start, goal = scenario("slalom")
planner = SkatePlanner.for_world(world, cell_size=0.1)
solution = planner.plan(world, start, goal, n_starts=12)

print(solution.report())
plot_solution(world, solution, radius=planner.radius, save="plan.png")
```

```bash
python examples/skate/plan_single.py     --scenario slalom
python examples/skate/plan_multistart.py --scenario clutter --starts 12
```

### Batched multi-start

Theseus batches over dim 0, so `n_starts` restarts cost **one** solve rather than N. Seeds fan out
perpendicular to the straight line, alternating sides, because a straight line through an obstacle
carries no gradient telling it which way to go around.

Batching is the win here — not the GPU. Measured on the slalom map (horizon 45, RTX 4090):

| restarts | CPU | CUDA |
|---:|---:|---:|
| 1 | 0.31 s | 0.62 s |
| 8 | 0.74 s | 1.07 s |
| 32 | 2.29 s | **1.33 s** |
| 128 | 10.82 s | **2.07 s** |
| 512 | 49.58 s | **4.57 s** |

Below ~32 restarts the problem is small enough that kernel-launch overhead dominates and **CPU is
faster**; the examples therefore default to `--device cpu`. Past the crossover the GPU pulls away
to ~10× at 512 restarts. Use `--device cuda` when you are sweeping many restarts or many worlds
at once.

## Reading the diagnostics

Every solve reports metrics computed on the **rolled-out** trajectory — the controls re-integrated
from the true start pose — not on the optimizer's pose variables:

```
goal xy err 0.0060 m (optimizer thought 0.0001) | drift 0.0098 m | clearance 0.3790 m
```

That parenthetical is the point. Poses and controls are separate variables coupled only by a soft
residual, so the optimizer's final pose can sit exactly on the goal while the path the robot would
actually drive misses it. Here the gap is 60×. `min_clearance` is likewise measured on a densified
rollout (`substeps=8`) against the **analytic** SDF, so neither grid interpolation error nor
corner-cutting between knots can hide a collision.

A plan is `feasible` when the executed path both clears every obstacle and reaches the goal.

## `twist_mode`

- **`"structural"`** (default) — 2-dof controls `[v, ω]` plus `SkateDynamics`. Slip is impossible.
- **`"penalty"`** — 3-dof body twists with `th.eb.DoubleIntegrator` + `th.eb.Nonholonomic`, the
  library-native formulation, where slip is a variable that is merely penalized.

The second exists as a cross-check: it is built entirely from Theseus primitives, so agreement
between the two is independent evidence that the custom residual is correct. On the slalom map
they find paths within ~2% of each other in length with identical goal error.

`GPMotionModel` would be the more idiomatic choice for the penalty path, but `GPCostWeight`
whitens with the Cholesky of a block matrix coupling the position and velocity halves, so its
dynamics rows do not end up weighted at `dynamics` and the two modes stop being comparable.
`DoubleIntegrator` with a `DiagonalCostWeight` keeps the comparison honest.

## Tuning

`SkateWeights` are stated on the **squared** error; the planner passes `sqrt(w)` internally, since
Theseus cost weights are whitening factors (`ScaleCostWeight(s)` contributes `s²‖e‖²/2`).

`dynamics=2000` was chosen by sweep. At 400 the collision term wins trades against the dynamics
residual and the rolled-out path drifts ~0.17 m off the optimized poses; above ~50000 the problem
gets stiff enough that LM stalls after a few iterations.

## Limitations

- **Collision is checked at knots only.** Between knots the robot follows an arc that the cost
  function never sees, so a thin obstacle can be tunnelled. The safety margin absorbs this at
  normal speeds, and `evaluate(substeps=...)` will *detect* it after the fact, but the optimizer
  is not defending against it. Shorten `dt` in fast or tight scenes.
- **Local optimizer.** Multi-start mitigates poor minima; it does not eliminate them. On the
  cluttered map roughly half the restarts still fail.
- **`info.best_solution` is detached and CPU-resident** (Theseus clones it with `.cpu()`). The
  planner moves it back, but anyone extending this toward a differentiable `TheseusLayer` use —
  backpropagating through the solve to fit cost weights — must use the `solution` dict returned by
  `forward()` instead, and accept the last iterate rather than the best one.

