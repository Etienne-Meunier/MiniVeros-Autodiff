"""
Experiment 9b -- does averaging quiet the finite difference without moving the horizon?

Figure 9 found that no loss among six -- including two that average mean(SST^2)
over time -- buys a longer horizon. The open question this follows up on:
averaging should smooth out chaos in the finite difference's own estimate
(each rollout's noise partly cancels across the steps or the space being
averaged), but there is no reason it should help the adjoint, which still
walks every step of the *unaveraged* dynamics on the way back regardless of
how the loss reduces the final state. This reuses figure 1's round-off
ensemble -- the same N_MEMBERS perturbed initial states -- across a handful
of exp9_loss_choice's own loss definitions, split into what should and
shouldn't respond to averaging:

  final_mse           figure 1/9's baseline: mean(SST^2) at the final step.
  avg_mse_full        mean(SST^2) averaged over the whole rollout (time avg).
  avg_mse_last30      mean(SST^2) averaged over the last 30 days (time avg).
  global_mean_sst_sq  (mean(SST))^2 at the final step (space avg, before
                      squaring rather than after -- smoother in space).

For each, member 0 (unperturbed) gives the usual AD-vs-FD horizon comparison;
all members together give the finite difference's own round-off spread. The
prediction: the spread (panel a) should shrink from final_mse to the
averaged losses; the AD-vs-FD disagreement (panel b) should not.

    python test/paper_figures/exp9b_loss_chaos.py

Cost: len(LENGTHS) * len(LOSS_NAMES) * N_MEMBERS * (1 backward + 2 FD rollouts); one jitted
(loss, grad) pair per (length, loss name), reused across every member.
"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import common
from exp9_loss_choice import LAST_WINDOW_DAYS, rollout_with_running_mse

PARAM = "tke_closure.c_k"
LENGTHS = [10, 20, 40, 80, 160, 320]  # days
LOSS_NAMES = ["final_mse", "avg_mse_full", "avg_mse_last30", "global_mean_sst_sq"]
REL_STEP = 1e-5
N_MEMBERS = 6
ROUNDOFF_STD = 1e-12  # K, as in exp1_gradient_check


def perturbed_state0(state0, model, key):
    """exp1_gradient_check's own perturbation: ocean temperature + ROUNDOFF_STD noise."""
    ocean = common.interior(model.boundary_conditions.maskT) != 0
    noise = jnp.zeros_like(state0.state.temp)
    interior_noise = ROUNDOFF_STD * jax.random.normal(key, ocean.shape) * ocean
    noise = noise.at[2:-2, 2:-2, :].set(interior_noise)
    new_state = eqx.tree_at(lambda s: s.temp, state0.state, state0.state.temp + noise)
    return eqx.tree_at(lambda s: s.state, state0, new_state)


def make_loss(name, model, forcing_fn, n_steps):
    """A function of (params, member_state0), unlike exp9_loss_choice's own loss_fn which bakes
    state0 into the closure -- state0 needs to be a traced argument here so one jitted function
    is reused across every round-off member instead of recompiling per member.
    """
    if name == "final_mse":
        return lambda params, state0: common.mean_square_sst(
            common.rollout(common.with_params(model, params), state0, forcing_fn, n_steps).state)
    if name == "avg_mse_full":
        return lambda params, state0: rollout_with_running_mse(
            common.with_params(model, params), state0, forcing_fn, n_steps, n_steps)[1]
    if name == "avg_mse_last30":
        window = min(LAST_WINDOW_DAYS, n_steps)
        return lambda params, state0: rollout_with_running_mse(
            common.with_params(model, params), state0, forcing_fn, n_steps, window)[1]
    if name == "global_mean_sst_sq":
        def f(params, state0):
            final = common.rollout(common.with_params(model, params), state0, forcing_fn, n_steps).state
            return common.sst(final).mean() ** 2
        return f
    raise ValueError(name)


def main(lengths, loss_names, n_members):
    model, state0, forcing_fn = common.spinup()
    base = common.get_param(model, PARAM)
    h = REL_STEP * jnp.abs(base)
    keys = jax.random.split(jax.random.PRNGKey(0), n_members)

    n_lengths, n_losses = len(lengths), len(loss_names)
    grad_ad = np.full((n_lengths, n_members, n_losses), np.nan)
    grad_fd = np.full((n_lengths, n_members, n_losses), np.nan)

    for i, n in enumerate(lengths):
        t0 = time.time()
        # One jitted (loss, grad) pair per loss name, compiled once and reused across every
        # member below -- member_state0 is a traced argument, not baked into the closure.
        losses = {name: eqx.filter_jit(make_loss(name, model, forcing_fn, n)) for name in loss_names}
        grads = {name: eqx.filter_jit(eqx.filter_grad(losses[name])) for name in loss_names}

        for mi in range(n_members):
            member_state0 = state0 if mi == 0 else perturbed_state0(state0, model, keys[mi])
            for j, name in enumerate(loss_names):
                grad_ad[i, mi, j] = grads[name]({PARAM: base}, member_state0)[PARAM]
                plus = losses[name]({PARAM: base + h}, member_state0)
                minus = losses[name]({PARAM: base - h}, member_state0)
                grad_fd[i, mi, j] = (plus - minus) / (2 * h)

        spread = np.nanstd(grad_fd[i], axis=0) / np.maximum(np.abs(np.nanmedian(grad_fd[i], axis=0)), 1e-30)
        rel_err = np.abs(grad_ad[i, 0] - grad_fd[i, 0]) / np.abs(grad_fd[i, 0])
        print(f"n={n:5d}d  fd relative spread={np.round(spread, 3)}  ad-vs-fd rel_err={np.round(rel_err, 3)}  "
              f"({time.time() - t0:.0f}s)")

    common.save(
        "exp9b_loss_chaos",
        lengths=lengths, loss_names=loss_names, param=PARAM, n_members=n_members,
        grad_ad=grad_ad, grad_fd=grad_fd, roundoff_std=ROUNDOFF_STD,
        dt_tracer=model.config.dt_tracer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--losses", nargs="+", default=LOSS_NAMES, choices=LOSS_NAMES)
    parser.add_argument("--n-members", type=int, default=N_MEMBERS)
    args = parser.parse_args()
    main(args.lengths, args.losses, args.n_members)
