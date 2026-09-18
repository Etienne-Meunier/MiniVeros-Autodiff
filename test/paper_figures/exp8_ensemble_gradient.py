"""
Experiment 8 -- past the horizon, is the gradient noise or is it wrong?

Figures 1 and 4 show the backpropagated gradient becoming useless somewhere past
figure 1's horizon H. That leaves the question a practitioner actually cares
about: is the blown-up gradient a noisy estimate of the right thing, in which
case averaging over an ensemble recovers it, or is it systematically wrong, in
which case no amount of averaging helps and the rollout length is a wall.

So we take K members, each the settled state plus a small temperature
perturbation -- same climate, different trajectory -- and for every member
compute both gradients of the same loss: reverse-mode autodiff, and a central
finite difference. The finite difference stays order 1 at every length (figure 1
panel b), so the ensemble-mean finite difference is the reference the ensemble-
mean autodiff gradient has to converge to. If it does, the blow-up is noise and
an ensemble is the remedy. If the running mean stays put while the spread grows,
it is not.

    python test/paper_figures/exp8_ensemble_gradient.py

About 20 min on a laptop CPU: one backward pass and two forward passes per
member per length.
"""

import argparse
import dataclasses
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import common

def default_lengths():
    if common.HORIZON_DAYS is None:
        raise SystemExit("common.HORIZON_DAYS is not set yet -- run exp1 first, or pass --lengths explicitly")
    H = common.HORIZON_DAYS
    return [H // 2, H, H * 2, H * 4]


PARAM = "tke_closure.c_k"
PERTURBATION = 1e-3  # K, small enough that every member is the same climate
REL_STEP = 1e-4  # finite-difference step, as a fraction of the parameter


def main(lengths, n_members, perturbation):
    model, state0, forcing_fn = common.spinup()
    mask = model.boundary_conditions.maskT
    base = common.get_param(model, PARAM)

    def member(key):
        """The settled state, nudged by a small temperature perturbation."""
        noise = jax.random.normal(key, state0.state.temp.shape) * mask * perturbation
        temp = state0.state.temp + noise
        return dataclasses.replace(state0, state=dataclasses.replace(state0.state, temp=temp))

    keys = jax.random.split(jax.random.PRNGKey(0), n_members)
    members = [member(k) for k in keys]

    grad_ad = np.zeros((len(lengths), n_members))
    grad_fd = np.zeros((len(lengths), n_members))

    for i, n in enumerate(lengths):

        @eqx.filter_jit
        def loss(value, integrator_state, n=n):
            model_p = common.with_params(model, {PARAM: value})
            return common.mean_square_sst(common.rollout(model_p, integrator_state, forcing_fn, n).state)

        grad = eqx.filter_jit(eqx.filter_grad(loss))
        h = REL_STEP * jnp.abs(base)

        t0 = time.time()
        for j, integrator_state in enumerate(members):
            grad_ad[i, j] = grad(base, integrator_state)
            grad_fd[i, j] = (loss(base + h, integrator_state) - loss(base - h, integrator_state)) / (2 * h)

        mean_ad, mean_fd = grad_ad[i].mean(), grad_fd[i].mean()
        print(f"n={n:5d}  mean AD={mean_ad:+.4e}  mean FD={mean_fd:+.4e}  "
              f"ratio={mean_ad / mean_fd:+.3e}  AD spread={grad_ad[i].std():.3e}  "
              f"({time.time() - t0:.0f}s)")

    common.save(
        "exp8_ensemble_gradient",
        lengths=lengths, param=PARAM, n_members=n_members, perturbation=perturbation,
        grad_ad=grad_ad, grad_fd=grad_fd, dt_tracer=model.config.dt_tracer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=None)
    parser.add_argument("--n-members", type=int, default=24)
    parser.add_argument("--perturbation", type=float, default=PERTURBATION)
    args = parser.parse_args()
    main(args.lengths or default_lengths(), args.n_members, args.perturbation)
