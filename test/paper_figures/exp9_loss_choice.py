"""
Experiment 9 -- does the loss choice buy horizon?

Figure 1 checks one loss, mean(SST^2) at the final step, against six model
parameters over growing rollouts. This repeats that same AD-vs-FD check for
one parameter (c_k) but six different losses, to see whether some losses stay
trustworthy longer than others:

  final_mse         mean(SST^2) at the final step -- figure 1's loss.
  avg_mse_full       mean(SST^2) averaged over every step of the rollout.
  avg_mse_last30     mean(SST^2) averaged over the last 30 days only.
  global_mean_sst_sq (mean(SST))^2 at the final step -- averaging before
                     squaring, so a smoother function of the state.
  upper_ts_misfit    figure 3's loss: T,S on the top N_LAYERS layers at the
                     final step, against a twin rollout at c_k * TARGET_OFFSET
                     (not at the base c_k itself -- that would make the loss
                     and its gradient exactly zero right where we differentiate,
                     turning the AD-vs-FD check into a comparison of two
                     round-off noise floors instead of a real gradient check).
  heat_content_top3  mean depth-weighted temperature of the top N_LAYERS
                     layers at the final step.

The two running-average losses accumulate their sum inside the checkpointed
scan (one extra scalar carried alongside the state) rather than storing the
whole trajectory, so their memory cost stays the same as the others'.

    python test/paper_figures/exp9_loss_choice.py

"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import common
from mini_veros import loop

PARAM = "c_k"
LENGTHS = [10, 20, 40, 80, 160, 320, 640]  # days
REL_STEP = 1e-5
LAST_WINDOW_DAYS = 30
N_LAYERS = 3
HORIZON_TOL = 1e-2
TARGET_OFFSET = 1.2  # upper_ts_misfit's target is rolled out at base*TARGET_OFFSET, not at base itself
LOSS_NAMES = ["final_mse", "avg_mse_full", "avg_mse_last30", "global_mean_sst_sq", "upper_ts_misfit", "heat_content_top3"]


def rollout_with_running_mse(model, state0, forcing_fn, n_steps, window, checkpoint_every=10):
    """(final_state, mean(SST^2) averaged over the last `window` steps of n_steps)."""
    assert n_steps % checkpoint_every == 0
    start_avg = n_steps - window

    def one_step(carry, step_idx):
        integrator_state, running = carry
        force = forcing_fn(model, integrator_state.state)
        integrator_state = loop.step(model, integrator_state, force)
        include = (step_idx >= start_avg).astype(running.dtype)
        return (integrator_state, running + include * common.mean_square_sst(integrator_state.state)), None

    @jax.checkpoint
    def block(carry, block_idx):
        base = block_idx * checkpoint_every
        carry, _ = lax.scan(one_step, carry, base + jnp.arange(checkpoint_every))
        return carry, None

    (final_state, running), _ = lax.scan(block, (state0, jnp.array(0.0)), jnp.arange(n_steps // checkpoint_every))
    return final_state, running / window


def heat_content_top3(model, state):
    dz = model.grid.zt[-N_LAYERS:]
    temp = common.interior(state.temp)[..., -N_LAYERS:]
    return (temp * dz).mean()


def loss_fn(name, model, state0, forcing_fn, n_steps, base_target=None):
    """A scalar function of {PARAM: value}, one of LOSS_NAMES."""
    if name == "final_mse":
        return lambda params: common.mean_square_sst(
            common.rollout(common.with_params(model, params), state0, forcing_fn, n_steps).state)
    if name == "avg_mse_full":
        return lambda params: rollout_with_running_mse(
            common.with_params(model, params), state0, forcing_fn, n_steps, n_steps)[1]
    if name == "avg_mse_last30":
        window = min(LAST_WINDOW_DAYS, n_steps)
        return lambda params: rollout_with_running_mse(
            common.with_params(model, params), state0, forcing_fn, n_steps, window)[1]
    if name == "global_mean_sst_sq":
        def f(params):
            final = common.rollout(common.with_params(model, params), state0, forcing_fn, n_steps).state
            return common.sst(final).mean() ** 2
        return f
    if name == "upper_ts_misfit":
        target_temp, target_salt, scale = base_target

        def f(params):
            final = common.rollout(common.with_params(model, params), state0, forcing_fn, n_steps).state
            return common.upper_ts_misfit(model, final, target_temp, target_salt, scale)
        return f
    if name == "heat_content_top3":
        return lambda params: heat_content_top3(
            model, common.rollout(common.with_params(model, params), state0, forcing_fn, n_steps).state)
    raise ValueError(name)


def main(lengths, loss_names):
    model, state0, forcing_fn = common.spinup()
    base = getattr(model.parameters, PARAM)
    h = REL_STEP * jnp.abs(base)

    grad_ad = np.full((len(lengths), len(loss_names)), np.nan)
    grad_fd = np.full((len(lengths), len(loss_names)), np.nan)

    for i, n in enumerate(lengths):
        # upper_ts_misfit's target must come from a *different* parameter value than the one
        # being differentiated -- built from `base` itself, loss(base) and its gradient would
        # be exactly zero (base is then the misfit's own global minimum), and checking AD vs FD
        # at a stationary point compares two numbers that are both round-off noise, blowing up
        # their ratio for reasons that have nothing to do with autodiff's correctness.
        target_final = common.rollout(
            common.with_params(model, {PARAM: base * TARGET_OFFSET}), state0, forcing_fn, n).state
        base_target = (*common.upper_ts(target_final, N_LAYERS), common.upper_ts_scale(model, target_final, N_LAYERS))

        for j, name in enumerate(loss_names):
            t0 = time.time()
            loss = eqx.filter_jit(loss_fn(name, model, state0, forcing_fn, n, base_target))
            grad = eqx.filter_jit(eqx.filter_grad(loss))
            grad_ad[i, j] = grad({PARAM: base})[PARAM]
            plus = loss({PARAM: base + h})
            minus = loss({PARAM: base - h})
            grad_fd[i, j] = (plus - minus) / (2 * h)
            print(f"n={n:5d}d  {name:20s}  ad={grad_ad[i, j]:+.4e}  fd={grad_fd[i, j]:+.4e}  "
                  f"({time.time() - t0:.0f}s)")

    rel_err = np.abs(grad_ad - grad_fd) / np.abs(grad_fd)  # (length, loss)
    horizon = np.zeros(len(loss_names))
    for j, name in enumerate(loss_names):
        inside = rel_err[:, j] < HORIZON_TOL
        horizon[j] = max((lengths[i] for i in range(len(lengths)) if inside[i]), default=0)
        print(f"horizon({name}) = {horizon[j]:.0f} d")

    common.save(
        "exp9_loss_choice",
        lengths=lengths, loss_names=loss_names, param=PARAM,
        grad_ad=grad_ad, grad_fd=grad_fd, horizon=horizon,
        dt_tracer=model.config.dt_tracer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--losses", nargs="+", default=LOSS_NAMES, choices=LOSS_NAMES)
    args = parser.parse_args()
    main(args.lengths, args.losses)
