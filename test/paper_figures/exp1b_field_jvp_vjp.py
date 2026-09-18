"""
Experiment 1b -- does figure 1's horizon generalise past a single scalar loss?

Figure 1 checks one loss (mean(SST^2)) against three parameters. This asks
two different follow-up questions:

(a) Field-loss AD-vs-FD check. Repeat figure 1's own reverse-mode-vs-finite-
    difference test, but with the loss built from a whole prognostic field's
    mean square (temperature, salinity) instead of SST alone, at the same
    growing rollout lengths. If the horizon is a property of the rollout
    (chaos in the dynamics) rather than of that one scalar diagnostic, it
    should show up here too, for every (field, parameter) pair.

(b) Forward-vs-reverse consistency, no finite difference involved. Reverse
    mode (vjp, one backward pass) and forward mode (jvp, one forward pass per
    direction) differentiate the *same* computation and are mathematically
    identical for any direction; the only thing that can make them disagree
    is floating point -- different operation order, no shared adjoint tape.
    For several random directions in the three-parameter space, this checks
    jvp(loss, direction) against vjp_gradient . direction, across the same
    growing lengths. This isolates how far the two exact-arithmetic-
    equivalent modes drift apart from each other, with no finite difference
    in the loop at all.

    python test/paper_figures/exp1b_field_jvp_vjp.py

Cost: part (a) is len(LENGTHS) * len(FIELDS) * len(PARAMS) * (1 backward +
2 FD evaluations); part (b) is len(LENGTHS) * N_DIRECTIONS * (1 backward +
1 forward).
"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import common

PARAMS = ["tke_closure.c_k", "tke_closure.c_eps", "A_h"]
FIELDS = ["temp", "salt"]
LENGTHS = [10, 20, 40, 80, 160, 320, 640]  # days
REL_STEP = 1e-5
N_DIRECTIONS = 5
DIRECTION_SEED = 0


def mean_square_field(state, name):
    """mean(field^2) over the whole interior volume -- figure 1's mean_square_sst, generalised
    from the surface layer alone to every layer of one prognostic field."""
    return (common.interior(getattr(state, name)) ** 2).mean()


def random_directions(n, param_names, seed):
    """n random unit vectors over param_names, as a list of {name: float} dicts."""
    key = jax.random.PRNGKey(seed)
    v = jax.random.normal(key, (n, len(param_names)))
    v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    return [{p: v[i, j] for j, p in enumerate(param_names)} for i in range(n)]


def main(lengths, checkpoint_every):
    model, state0, forcing_fn = common.spinup()
    base = {p: common.get_param(model, p) for p in PARAMS}

    @eqx.filter_jit
    def field_loss(params, n_steps, field_name):
        final = common.rollout(common.with_params(model, params), state0, forcing_fn, n_steps, checkpoint_every)
        return mean_square_field(final.state, field_name)

    @eqx.filter_jit
    def sst_loss(params, n_steps):
        final = common.rollout(common.with_params(model, params), state0, forcing_fn, n_steps, checkpoint_every)
        return common.mean_square_sst(final.state)

    grad_field = eqx.filter_jit(eqx.filter_grad(field_loss))
    grad_sst = eqx.filter_jit(eqx.filter_grad(sst_loss))

    @eqx.filter_jit
    def jvp_sst(params, tangent, n_steps):
        """One jitted function reused across every direction for a given n_steps, rather than
        recompiling per direction (params/tangent are traced dynamic inputs of fixed shape)."""
        _, jv = jax.jvp(lambda p: sst_loss(p, n_steps), (params,), (tangent,))
        return jv

    n_lengths = len(lengths)

    # (a) field-loss AD vs FD
    grad_ad = np.full((n_lengths, len(FIELDS), len(PARAMS)), np.nan)
    grad_fd = np.full((n_lengths, len(FIELDS), len(PARAMS)), np.nan)

    # (b) jvp vs vjp, no FD
    directions = random_directions(N_DIRECTIONS, PARAMS, DIRECTION_SEED)
    jvp_vals = np.full((n_lengths, N_DIRECTIONS), np.nan)
    vjp_proj = np.full((n_lengths, N_DIRECTIONS), np.nan)

    for i, n in enumerate(lengths):
        t0 = time.time()

        for fi, field_name in enumerate(FIELDS):
            g = grad_field(base, n, field_name)
            grad_ad[i, fi] = [g[p] for p in PARAMS]
            for j, p in enumerate(PARAMS):
                h = REL_STEP * jnp.abs(base[p])
                plus = field_loss({**base, p: base[p] + h}, n, field_name)
                minus = field_loss({**base, p: base[p] - h}, n, field_name)
                grad_fd[i, fi, j] = (plus - minus) / (2 * h)

        g_sst = grad_sst(base, n)
        for di, direction in enumerate(directions):
            vjp_proj[i, di] = sum(float(g_sst[p]) * float(direction[p]) for p in PARAMS)
            jvp_vals[i, di] = jvp_sst(base, direction, n)

        rel_err_a = np.abs(grad_ad[i] - grad_fd[i]) / np.abs(grad_fd[i])
        rel_err_b = np.abs(jvp_vals[i] - vjp_proj[i]) / np.abs(vjp_proj[i])
        print(f"n={n:5d}d  (a) field/param max rel_err={rel_err_a.max():.2e}  "
              f"(b) jvp-vs-vjp max rel_err={rel_err_b.max():.2e}  ({time.time() - t0:.0f}s)")

    common.save(
        "exp1b_field_jvp_vjp",
        lengths=lengths, fields=FIELDS, params=PARAMS, rel_step=REL_STEP,
        grad_ad=grad_ad, grad_fd=grad_fd,
        directions=np.array([[float(d[p]) for p in PARAMS] for d in directions]),
        jvp_vals=jvp_vals, vjp_proj=vjp_proj,
        dt_tracer=model.config.dt_tracer, checkpoint_every=checkpoint_every,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    main(args.lengths, args.checkpoint_every)
