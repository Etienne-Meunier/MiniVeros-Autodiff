"""
Experiment 5b -- where initial-state assimilation stops working, across metric and length.

Companion to figure 5 (exp5_assimilation.py), which fixes the observation
operator (T,S on the top 3 layers) and the rollout length (H/2 = 40 days)
and only asks whether *that one* setup is identifiable -- and finds it
isn't quite: the whole-volume state error only falls to ~60% of its start
even in the "identifiable" case (see report/paper/gradients_and_limits.md).
This sweeps both axes that choice fixed: which layers are observed (five
operators, from SST alone to the whole column) and the rollout length
(10..320 days, H/8..4H), against one initial-state perturbation with no
particular depth structure of its own -- a bump of uniform amplitude at
every level, masked to the ocean. Recoverability is then purely a function
of the metric and the rollout, not of where the perturbation happened to
sit.

The step rule and the top-layer T,S misfit are exp5_assimilation.py's own;
only the observation operator (n_layers, and one non-scaled SST-MSE variant)
and the rollout length change across the grid. What's saved per (length,
metric) is the fitted field itself (not just its rms), so fig5b can plot
the recovery error by depth -- the whole point being *where* it stops
vanishing, not just by how much.

    python test/paper_figures/exp5b_assimilation_horizon.py

Cost is sum(lengths) * len(metrics) * n_iterations model steps (each length
also needs its own truth rollout), so it grows fast with the longest length
-- with the defaults (6 lengths, 5 metrics, 100 iterations) expect this to
run for hours on a GPU node, most of it spent at 160 and 320 days.
"""

import argparse
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

import common

BUMP_AMPLITUDE = 2.0  # K
LENGTHS = [10, 20, 40, 80, 160, 320]  # days: H/8 .. 4H, H = common.HORIZON_DAYS = 80
# name -> observed top layers; "sst_mse" additionally swaps the scaled T,S misfit
# for a plain mean-square SST error, so the sweep varies the loss itself, not
# only its depth coverage.
METRICS = ["sst_mse", "top1_TS", "top3_TS", "top6_TS", "full_TS"]


def horizontal_envelope(model):
    x, y = model.grid.xt[:, None], model.grid.yt[None, :]
    return jnp.sin(2 * jnp.pi * x / 60.0) * jnp.exp(-((y - 10.0) ** 2) / (2 * 15.0**2))


def full_depth_bump(model):
    """A smooth bump of uniform amplitude at every depth -- unlike figure 5's bumps, not
    confined to or excluded from any particular set of layers, so whether it's recovered
    is decided purely by the metric and rollout length being swept, not by the bump."""
    envelope = horizontal_envelope(model)[:, :, None]
    return BUMP_AMPLITUDE * envelope * model.boundary_conditions.maskT


def metric_n_layers(model, name):
    return {"sst_mse": 1, "top1_TS": 1, "top3_TS": 3, "top6_TS": 6, "full_TS": model.config.nz}[name]


def make_misfit(model, name, true_final):
    """(misfit_fn, n_layers) for one named metric, targeted at `true_final`."""
    n_layers = metric_n_layers(model, name)
    if name == "sst_mse":
        target = common.sst(true_final)
        return (lambda state: jnp.mean((common.sst(state) - target) ** 2)), n_layers

    target_temp, target_salt = common.upper_ts(true_final, n_layers)
    scale = common.upper_ts_scale(model, true_final, n_layers)
    return (lambda state: common.upper_ts_misfit(model, state, target_temp, target_salt, scale)), n_layers


def fit(model, state0, forcing_fn, truth, start, misfit_fn, n_steps, n_iterations, step_size):
    """Steepest descent with a kelvin trust radius -- exp5_assimilation.py's own step rule."""
    mask = model.boundary_conditions.maskT

    @eqx.filter_jit
    def final_state(temp):
        integrator_state = eqx.tree_at(lambda s: s.state.temp, state0, temp)
        return common.rollout(model, integrator_state, forcing_fn, n_steps).state

    def loss(temp):
        return misfit_fn(final_state(temp))

    schedule = optax.cosine_decay_schedule(step_size, n_iterations)

    @eqx.filter_jit
    def update(temp, radius):
        value, grads = eqx.filter_value_and_grad(loss)(temp)
        direction = grads * mask
        # See exp5_assimilation.py: a high percentile, not the max, so one hot spot
        # doesn't freeze every other cell's step.
        scale_g = jnp.maximum(jnp.percentile(jnp.abs(direction), 99.5), 1e-30)
        return temp - radius * jnp.clip(direction / scale_g, -1.0, 1.0), value

    def state_error(temp):
        return jnp.sqrt((((temp - truth) * mask) ** 2).sum() / jnp.maximum((mask > 0).sum(), 1))

    temp, history = start, []
    for i in range(n_iterations):
        temp, loss_i = update(temp, schedule(i))
        history.append((float(loss_i), float(state_error(temp))))
    return temp, np.array(history)


def main(lengths, metrics, n_iterations, step_size):
    model, state0, forcing_fn = common.spinup()
    truth = state0.state.temp
    start = truth + full_depth_bump(model)
    nz = model.config.nz
    x, y, land = common.grid_arrays(model)

    shape = (len(lengths), len(metrics))
    loss_start = np.full(shape, np.nan)
    loss_end = np.full(shape, np.nan)
    state_err_start = np.full(shape, np.nan)
    state_err_end = np.full(shape, np.nan)
    fitted = np.full(shape + land.shape + (nz,), np.nan)

    for i, n in enumerate(lengths):
        @eqx.filter_jit
        def true_final(n=n):
            return common.rollout(model, state0, forcing_fn, n).state
        tf = true_final()

        for j, name in enumerate(metrics):
            misfit_fn, n_layers = make_misfit(model, name, tf)
            t0 = time.time()
            fitted_ij, history = fit(model, state0, forcing_fn, truth, start, misfit_fn, n, n_iterations, step_size)
            loss_start[i, j], loss_end[i, j] = history[0, 0], history[-1, 0]
            state_err_start[i, j], state_err_end[i, j] = history[0, 1], history[-1, 1]
            fitted[i, j] = common.interior(fitted_ij)
            print(f"n_steps={n:4d}  metric={name:9s} (top {n_layers:2d}/{nz} layers)  "
                  f"loss {loss_start[i, j]:.2e} -> {loss_end[i, j]:.2e}  "
                  f"state err {state_err_start[i, j]:.3e} -> {state_err_end[i, j]:.3e}  "
                  f"({time.time() - t0:.0f}s)")

    common.save(
        "exp5b_assimilation_horizon",
        lengths=lengths, metrics=metrics, n_layers=[metric_n_layers(model, m) for m in metrics], nz=nz,
        truth=common.interior(truth), fitted=fitted,
        loss_start=loss_start, loss_end=loss_end,
        state_err_start=state_err_start, state_err_end=state_err_end,
        x=x, y=y, z=np.asarray(model.grid.zt), land=land,
        n_iterations=n_iterations, step_size=step_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--metrics", nargs="+", default=METRICS, choices=METRICS)
    parser.add_argument("--n-iterations", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=0.05, help="trust radius in kelvin per iteration")
    args = parser.parse_args()
    main(args.lengths, args.metrics, args.n_iterations, args.step_size)
