"""
Experiment 5 -- assimilating an initial state instead of a parameter.

Same twin idea as figures 3/4, but the unknown is now the whole initial
temperature field: tens of thousands of scalars against a handful of 2-d
observations, and reverse mode costs the same as it did for two parameters,
which is the point. The observation operator is figure 3's: temperature and
salinity on the top `N_LAYERS` layers, at the final step of an `n_steps`-day
rollout.

Two parts, using the same fit and the same step rule (below), which is why
they share one script:

  (a) identifiable.  The initial-state error is confined to the observed
      layers (a bump that is exactly zero below layer N_LAYERS). Everything
      unknown is observed, so the fit should recover the initial state nearly
      exactly.
  (b) non-identifiable. The initial-state error is confined to *below* the
      observed layers instead -- a perturbation the observation operator is
      structurally blind to. K_STARTS different such perturbations are fit
      from, and every one of them should drive the (fully observed) loss to
      the same floor while recovering a different deep temperature: the
      spread across fits, not any single one of them, is the result. Deep
      temperature is not "hard to recover" here, it is invisible to this
      observation operator -- forecasting from any of the K fits would
      silently carry a wrong deep state.

The step rule is steepest descent with a trust radius: each iteration moves
any ocean cell by at most `step_size` kelvin, in the gradient direction.
Two reasons. The gradient's magnitude changes by orders of magnitude with
rollout length, so any fixed learning rate is either inert or divergent,
while a step measured in kelvin is meaningful at every length. And Adam,
which would also be scale-free, normalises every cell separately, so it
would move the deep ocean in part (b) -- where the gradient is exactly zero,
not just small -- by the same clipped step as the surface, planting motion
where there is no signal at all. Keeping the gradient's own shape leaves the
null space alone.

    python test/paper_figures/exp5_assimilation.py

"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import common

BUMP_AMPLITUDE = 2.0  # K
BUMP_DEPTH = 300.0  # m, e-folding depth of the perturbation's horizontal envelope
N_LAYERS = 3  # observed layers, as in figure 3
K_STARTS = 4  # part (b): number of different deep perturbations fit from


def horizontal_envelope(model):
    x, y = model.grid.xt[:, None], model.grid.yt[None, :]
    return jnp.sin(2 * jnp.pi * x / 60.0) * jnp.exp(-((y - 10.0) ** 2) / (2 * 15.0**2))


def layer_mask(model, n_layers, below):
    """1.0 on the top n_layers levels (below=False) or everything strictly below them (below=True)."""
    nz = model.config.nz
    top = (jnp.arange(nz) >= nz - n_layers).astype(jnp.float32)
    return (1.0 - top) if below else top


def shallow_bump(model):
    """A smooth bump confined to the top N_LAYERS levels -- zero below them by construction."""
    envelope = horizontal_envelope(model)[:, :, None] * layer_mask(model, N_LAYERS, below=False)[None, None, :]
    return BUMP_AMPLITUDE * envelope * model.boundary_conditions.maskT


def deep_bump(model, key):
    """A smooth, random-phase bump confined strictly below the top N_LAYERS levels."""
    key_phase, key_y0 = jax.random.split(key)
    phase = jax.random.uniform(key_phase, (), minval=0.0, maxval=2 * jnp.pi)
    y0 = jax.random.uniform(key_y0, (), minval=-20.0, maxval=20.0)
    x, y, z = model.grid.xt[:, None, None], model.grid.yt[None, :, None], model.grid.zt[None, None, :]
    envelope = jnp.sin(2 * jnp.pi * x / 60.0 + phase) * jnp.exp(-((y - y0) ** 2) / (2 * 15.0**2))
    vertical = jnp.exp(z / BUMP_DEPTH) * layer_mask(model, N_LAYERS, below=True)[None, None, :]
    return BUMP_AMPLITUDE * envelope * vertical * model.boundary_conditions.maskT


def fit(model, state0, forcing_fn, truth, start, target_temp, target_salt, scale, n_steps, n_iterations, step_size):
    """Steepest descent with a kelvin trust radius on the full 3-d initial temperature field."""
    mask = model.boundary_conditions.maskT

    @eqx.filter_jit
    def final_state(temp):
        integrator_state = eqx.tree_at(lambda s: s.state.temp, state0, temp)
        return common.rollout(model, integrator_state, forcing_fn, n_steps).state

    def loss(temp):
        return common.upper_ts_misfit(model, final_state(temp), target_temp, target_salt, scale)

    schedule = optax.cosine_decay_schedule(step_size, n_iterations)

    @eqx.filter_jit
    def update(temp, radius):
        value, grads = eqx.filter_value_and_grad(loss)(temp)
        direction = grads * mask
        # Scale by a high percentile, not the maximum: the gradient can have a hot spot
        # (figure 2) orders of magnitude above the rest, and normalising by it would
        # freeze every other cell. Clipping keeps that one cell inside the radius.
        scale_g = jnp.maximum(jnp.percentile(jnp.abs(direction), 99.5), 1e-30)
        return temp - radius * jnp.clip(direction / scale_g, -1.0, 1.0), value

    def state_error(temp):
        return jnp.sqrt((((temp - truth) * mask) ** 2).sum() / jnp.maximum((mask > 0).sum(), 1))

    temp, history = start, []
    t0 = time.time()
    for i in range(n_iterations):
        temp, loss_i = update(temp, schedule(i))
        history.append((float(loss_i), float(state_error(temp))))
        if i % 20 == 0 or i == n_iterations - 1:
            print(f"    iter {i:4d}  loss={history[-1][0]:.4e}  rms(state err)={history[-1][1]:.4e}  "
                  f"({time.time() - t0:.0f}s)")
    return temp, np.array(history)


def main(n_steps, n_iterations, step_size):
    model, state0, forcing_fn = common.spinup()
    truth = state0.state.temp

    @eqx.filter_jit
    def final_state(temp):
        integrator_state = eqx.tree_at(lambda s: s.state.temp, state0, temp)
        return common.rollout(model, integrator_state, forcing_fn, n_steps).state

    true_final = final_state(truth)
    target_temp, target_salt = common.upper_ts(true_final, N_LAYERS)
    scale = common.upper_ts_scale(model, true_final, N_LAYERS)
    fit_args = (model, state0, forcing_fn, target_temp, target_salt, scale, n_steps, n_iterations, step_size)

    # (a) identifiable: the perturbation lives entirely inside the observed layers.
    print("part (a): identifiable")
    start_a = truth + shallow_bump(model)
    fitted_a, history_a = fit(*fit_args[:3], truth, start_a, *fit_args[3:])

    # (b) non-identifiable: K_STARTS perturbations, each confined below the observed layers.
    print("part (b): non-identifiable")
    keys = jax.random.split(jax.random.PRNGKey(0), K_STARTS)
    starts_b = [truth + deep_bump(model, k) for k in keys]
    fitted_b, history_b = [], []
    for k, start_k in enumerate(starts_b):
        print(f"  start {k}")
        fitted_k, history_k = fit(*fit_args[:3], truth, start_k, *fit_args[3:])
        fitted_b.append(fitted_k)
        history_b.append(history_k)
    fitted_b = jnp.stack(fitted_b)

    x, y, land = common.grid_arrays(model)
    common.save(
        "exp5_assimilation",
        n_steps=n_steps, n_layers=N_LAYERS, k_starts=K_STARTS, step_size=step_size,
        truth=common.interior(truth),
        start_a=common.interior(start_a), fitted_a=common.interior(fitted_a), history_a=history_a,
        starts_b=jnp.stack([common.interior(s) for s in starts_b]),
        fitted_b=jnp.stack([common.interior(f) for f in fitted_b]),
        history_b=np.stack(history_b),
        x=x, y=y, z=np.asarray(model.grid.zt), land=land,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=None,
                        help="defaults to common.HORIZON_DAYS // 2, once figure 1 has set it")
    parser.add_argument("--n-iterations", type=int, default=150)
    parser.add_argument("--step-size", type=float, default=0.05, help="trust radius in kelvin per iteration")
    args = parser.parse_args()
    n_steps = args.n_steps
    if n_steps is None:
        if common.HORIZON_DAYS is None:
            raise SystemExit("common.HORIZON_DAYS is not set yet -- run exp1 first, or pass --n-steps explicitly")
        n_steps = common.HORIZON_DAYS // 2
    main(n_steps, args.n_iterations, args.step_size)
