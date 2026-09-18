"""
Experiment 10b -- does the H=80-day horizon survive going from 2 scalar parameters to a NN?

The gating question for the whole NN-closure line of work. exp1_gradient_check.py checked
reverse-mode AD against central finite differences for 3 scalar parameters, one at a time (a
one-hot direction each). A NN closure has ~700 weights (two small MLPs, coefficient mode) --
one-hot FD per weight is 1400 extra rollouts per length, not worth it. Instead: N_DIRECTIONS
random unit vectors in the full (both nets') weight space, and for each, central FD along that
direction vs the reverse-mode gradient (one backward pass covers every weight) dotted with it.
Same rollout-length sweep as exp1.

The nets start at exp10a's constant-init (all weights zero, bias = c_k or c_eps) -- the
gradient is checked at the same point exp1/exp3 always calibrate from, and the perturbation
direction moves the weights *away* from that trivial (input-independent) point for the first
time, which is exactly the regime training would move through.

    python test/nn_tke/exp10b_gradient_check.py

Cost: len(LENGTHS) * N_DIRECTIONS * (1 backward [shared across directions] + 2 FD rollouts).
"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import common
from exp10a_identity_check import constant_mlp

LENGTHS = [10, 20, 30, 40, 60, 80, 120, 160, 240, 320]  # days
N_DIRECTIONS = 5
ABS_STEP = 1e-4  # fixed (not relative): weights start at exactly 0, no scale to be relative to
HORIZON_TOL = 1e-2


def random_unit_direction(nets, key):
    """An array-leaves-only pytree shaped like `nets`, drawn N(0,1) per-leaf and L2-normalized
    over ALL leaves combined (a single unit vector in the full weight space, not per-leaf).
    Non-array leaves (activation functions etc., eqx.nn.MLP's static fields) are left as None,
    matching what eqx.filter_grad's gradient pytree already looks like."""
    arrays, _ = eqx.partition(nets, eqx.is_array)
    leaves, treedef = jax.tree_util.tree_flatten(arrays)
    keys = jax.random.split(key, len(leaves))
    raw = [jax.random.normal(k, leaf.shape) for k, leaf in zip(keys, leaves)]
    norm = jnp.sqrt(sum(jnp.sum(r**2) for r in raw))
    unit = [r / norm for r in raw]
    return jax.tree_util.tree_unflatten(treedef, unit)


def pytree_dot(a, b):
    """a, b: array-leaves-only pytrees (matching structure -- e.g. both from eqx.filter_grad
    or random_unit_direction)."""
    return sum(jnp.sum(la * lb) for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def pytree_add(nets, direction, scale):
    """nets + scale * direction, direction being array-leaves-only (see random_unit_direction)."""
    arrays, static = eqx.partition(nets, eqx.is_array)
    moved = jax.tree_util.tree_map(lambda x, d: x + scale * d, arrays, direction)
    return eqx.combine(moved, static)


def main(lengths, n_directions, abs_step):
    model, state0, forcing_fn = common.spinup()
    key_ck, key_ceps, key_dirs = jax.random.split(jax.random.PRNGKey(0), 3)
    nets0 = (constant_mlp(model.parameters.tke_closure.c_k, key_ck), constant_mlp(model.parameters.tke_closure.c_eps, key_ceps))
    directions = [random_unit_direction(nets0, k) for k in jax.random.split(key_dirs, n_directions)]

    def with_nets(nets):
        return eqx.tree_at(
            lambda m: (m.parameters.tke_ck_net, m.parameters.tke_ceps_net), model, nets,
            is_leaf=lambda x: x is None,
        )

    @eqx.filter_jit
    def loss(nets, n_steps):
        return common.mean_square_sst(common.rollout(with_nets(nets), state0, forcing_fn, n_steps).state)

    grad = eqx.filter_jit(eqx.filter_grad(loss))

    n_lengths = len(lengths)
    ad_proj = np.full((n_lengths, n_directions), np.nan)
    fd_proj = np.full((n_lengths, n_directions), np.nan)

    for i, n in enumerate(lengths):
        t0 = time.time()
        g = grad(nets0, n)
        for di, direction in enumerate(directions):
            ad_proj[i, di] = float(pytree_dot(g, direction))
            plus = loss(pytree_add(nets0, direction, abs_step), n)
            minus = loss(pytree_add(nets0, direction, -abs_step), n)
            fd_proj[i, di] = float((plus - minus) / (2 * abs_step))

        rel_err = np.abs(ad_proj[i] - fd_proj[i]) / np.abs(fd_proj[i])
        print(f"n={n:5d}d  rel_err over directions={np.round(rel_err, 4)}  ({time.time() - t0:.0f}s)")

    rel_err_all = np.abs(ad_proj - fd_proj) / np.abs(fd_proj)
    inside = np.all(rel_err_all < HORIZON_TOL, axis=1)
    horizon_days = int(max((lengths[i] for i in range(n_lengths) if inside[i]), default=0))
    print(f"NN-closure horizon H_nn = {horizon_days} days (vs H = {common.HORIZON_DAYS} for the scalar closure)")

    common.save(
        "exp10b_gradient_check",
        lengths=lengths, n_directions=n_directions, abs_step=abs_step,
        ad_proj=ad_proj, fd_proj=fd_proj, horizon_days=horizon_days,
        scalar_horizon_days=common.HORIZON_DAYS, dt_tracer=model.config.dt_tracer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--n-directions", type=int, default=N_DIRECTIONS)
    parser.add_argument("--abs-step", type=float, default=ABS_STEP)
    args = parser.parse_args()
    main(args.lengths, args.n_directions, args.abs_step)
