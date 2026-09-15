"""
Experiment 1 -- how long a rollout can a backpropagated gradient survive?

For a loss L = mean(SST^2) at the end of an n-step rollout, we compare
dL/dp from reverse-mode autodiff (grad_ad) and from forward-mode autodiff
(grad_jvp, one jax.jvp per parameter's unit direction -- mathematically the
same number as the matching grad_ad entry, computed by walking the tangent
equations forward instead of the adjoint backward) against a central finite
difference,

    (L(p + h) - L(p - h)) / 2h,

for a handful of model parameters p and a range of rollout lengths n (in
days: global_4deg's dt_tracer = 1 day). Rather than sweeping h to gauge the
finite difference's own uncertainty, we sweep the initial condition: N members
are built by perturbing the spin-up's temperature by float64 round-off
(~1e-12 K white noise) and integrated with the *same* h. Once the gradient's
sensitivity to a round-off-sized change in initial conditions is as large as
the gradient itself, no h can make the finite difference trustworthy -- the
rollout, not the differencing, is what has stopped being reproducible.

Member 0 is always the unperturbed spin-up state, and its (AD, FD, JVP) triple
is what panels (a)/(b) plot as *the* gradient; all N members feed panel (c),
the round-off ensemble's relative spread.

    python test/paper_figures/exp1_gradient_check.py

Runtime is dominated by the finite differences: 2 forward rollouts per
(parameter, member) pair per length, versus one backward pass for all
parameters at once per member (reverse mode) plus one forward-mode pass per
parameter per member (jvp) -- roughly doubling the non-FD cost.
"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import common

PARAMS = ["c_k", "c_eps", "A_h"]  # TKE production, TKE dissipation, horizontal viscosity
LENGTHS = [10, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640, 960, 1280, 1920, 2560]  # days
REL_STEP = 1e-5  # h = REL_STEP * p, fixed across the whole ensemble
N_MEMBERS = 8
N_MEMBERS_BEYOND = 640  # lengths past this use N_MEMBERS_REDUCED instead, to bound cost
N_MEMBERS_REDUCED = 4
ROUNDOFF_STD = 1e-12  # K, roughly float64 round-off on a ~10 K temperature
HORIZON_TOL = 1e-2  # relative error below which a length counts as "inside the horizon"


def perturbed_state0(state0, model, key):
    """state0 with ocean temperature perturbed by ROUNDOFF_STD * standard normal noise."""
    ocean = common.interior(model.boundary_conditions.maskT) != 0
    noise = jnp.zeros_like(state0.state.temp)
    interior_noise = ROUNDOFF_STD * jax.random.normal(key, ocean.shape) * ocean
    noise = noise.at[2:-2, 2:-2, :].set(interior_noise)
    new_state = eqx.tree_at(lambda s: s.temp, state0.state, state0.state.temp + noise)
    return eqx.tree_at(lambda s: s.state, state0, new_state)


def n_members_for(n_steps):
    return N_MEMBERS_REDUCED if n_steps > N_MEMBERS_BEYOND else N_MEMBERS


def main(lengths, checkpoint_every):
    model, state0, forcing_fn = common.spinup()
    base = {p: getattr(model.parameters, p) for p in PARAMS}
    keys = jax.random.split(jax.random.PRNGKey(0), N_MEMBERS)

    @eqx.filter_jit
    def loss(params, member_state0, n_steps):
        final = common.rollout(common.with_params(model, params), member_state0, forcing_fn, n_steps, checkpoint_every)
        return common.mean_square_sst(final.state)

    grad = eqx.filter_jit(eqx.filter_grad(loss))

    @eqx.filter_jit
    def jvp_component(params, tangent, member_state0, n_steps):
        """Forward-mode directional derivative along `tangent` (a one-hot dict picks out a
        single parameter's dL/dp, matching the corresponding grad_ad entry). One jitted
        function reused across every member and parameter for a given n_steps -- tangent,
        params and member_state0 are all traced dynamic inputs of the same shape regardless
        of which one-hot direction is active, so this compiles once per length rather than
        once per (member, parameter) call.
        """
        _, jv = jax.jvp(lambda p: loss(p, member_state0, n_steps), (params,), (tangent,))
        return jv

    n_lengths = len(lengths)
    losses = np.full(n_lengths, np.nan)
    grad_ad = np.full((n_lengths, len(PARAMS)), np.nan)  # member 0 only
    grad_fd = np.full((n_lengths, len(PARAMS)), np.nan)  # member 0 only
    grad_jvp = np.full((n_lengths, len(PARAMS)), np.nan)  # member 0 only
    grad_ad_members = np.full((n_lengths, N_MEMBERS, len(PARAMS)), np.nan)
    grad_fd_members = np.full((n_lengths, N_MEMBERS, len(PARAMS)), np.nan)
    grad_jvp_members = np.full((n_lengths, N_MEMBERS, len(PARAMS)), np.nan)
    n_members_used = np.zeros(n_lengths, dtype=int)

    for i, n in enumerate(lengths):
        t0 = time.time()
        m = n_members_for(n)
        n_members_used[i] = m

        for mi in range(m):
            member_state0 = state0 if mi == 0 else perturbed_state0(state0, model, keys[mi])
            if mi == 0:
                losses[i] = loss(base, member_state0, n)
            g = grad(base, member_state0, n)
            grad_ad_members[i, mi] = [g[p] for p in PARAMS]
            for j, p in enumerate(PARAMS):
                h = REL_STEP * jnp.abs(base[p])
                plus = loss({**base, p: base[p] + h}, member_state0, n)
                minus = loss({**base, p: base[p] - h}, member_state0, n)
                grad_fd_members[i, mi, j] = (plus - minus) / (2 * h)
                tangent = {pp: (jnp.ones_like(base[pp]) if pp == p else jnp.zeros_like(base[pp])) for pp in base}
                grad_jvp_members[i, mi, j] = jvp_component(base, tangent, member_state0, n)

        grad_ad[i] = grad_ad_members[i, 0]
        grad_fd[i] = grad_fd_members[i, 0]
        grad_jvp[i] = grad_jvp_members[i, 0]
        rel_err = np.abs(grad_ad[i] - grad_fd[i]) / np.abs(grad_fd[i])
        print(f"n={n:5d}d  members={m}  L={losses[i]:.6f}  "
              f"dL/dc_k: ad={grad_ad[i, 0]:+.4e} fd={grad_fd[i, 0]:+.4e}  "
              f"rel_err(max over params)={rel_err.max():.2e}  ({time.time() - t0:.0f}s)")

    rel_err_all = np.abs(grad_ad - grad_fd) / np.abs(grad_fd)  # (n_lengths, n_params)
    inside = np.all(rel_err_all < HORIZON_TOL, axis=1)
    horizon_days = int(max((lengths[i] for i in range(n_lengths) if inside[i]), default=0))
    print(f"horizon H = {horizon_days} days (relative error < {HORIZON_TOL:.0e} for every parameter)")
    print(f"-> copy this into common.HORIZON_DAYS by hand once you are happy with it")

    common.save(
        "exp1_gradient_check",
        lengths=lengths, params=PARAMS, rel_step=REL_STEP, base=[base[p] for p in PARAMS],
        losses=losses, grad_ad=grad_ad, grad_fd=grad_fd, grad_jvp=grad_jvp,
        grad_ad_members=grad_ad_members, grad_fd_members=grad_fd_members, grad_jvp_members=grad_jvp_members,
        n_members_used=n_members_used,
        roundoff_std=ROUNDOFF_STD, horizon_days=horizon_days,
        dt_tracer=model.config.dt_tracer, checkpoint_every=checkpoint_every,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    main(args.lengths, args.checkpoint_every)
