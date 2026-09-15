"""
Experiment 3b -- does calibration's success depend on how wrong the starting guess is?

Figure 4 already sweeps rollout length; this adds a second axis, the
distance between the optimiser's starting c_k and the true value (c_eps
held fixed at truth, as in figure 4 -- fitting one identifiable parameter
keeps rollout length and starting distance as the only two variables). More
rollout points than figure 4 as well (LENGTHS below has twice as many).

For error bars, N_REPEATS members are built the way figure 1 builds its
round-off ensemble -- the spin-up's temperature perturbed by float64
round-off -- and for each repeat the truth and the fit both start from that
same perturbed state (so the only deliberate mismatch between them is the
parameter distance; the round-off is there to let the fit's own chaos,
figure 6's story, show up as within-cell spread rather than as a second
uncontrolled distance).

Step rule as in figures 3/4: steepest descent in log-parameter space with a
cosine-decayed trust radius, taking the sign of the gradient.

Saved raw (not just a plot): fitted[length, distance, repeat], so the matrix
can be re-visualised differently later without rerunning.

    python test/paper_figures/exp3b_calibration_distance.py

Cost is sum(LENGTHS) * len(DISTANCES) * N_REPEATS * n_iterations model steps
(each (length, repeat) pair also needs its own truth rollout, shared across
distances) -- expect this to need a GPU node for the full grid.
"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import common

TRUE_C_K = 0.10
FIXED = {"c_eps": 0.70}
LENGTHS = [10, 20, 40, 80, 120, 160, 240, 320]  # days: H/8 .. 4H, H = common.HORIZON_DAYS = 80
DISTANCES = [0.15, 0.3, 0.5, 0.7, 1.0]  # |log(start / true)|
N_REPEATS = 4
N_LAYERS = 3
ROUNDOFF_STD = 1e-12  # K, as in exp1_gradient_check
ROUNDOFF_SEED = 0


def perturbed_state0(state0, model, key):
    """state0 with ocean temperature perturbed by ROUNDOFF_STD * standard normal noise --
    exp1_gradient_check's own perturbation, reused here to seed error-bar repeats."""
    ocean = common.interior(model.boundary_conditions.maskT) != 0
    noise = jnp.zeros_like(state0.state.temp)
    interior_noise = ROUNDOFF_STD * jax.random.normal(key, ocean.shape) * ocean
    noise = noise.at[2:-2, 2:-2, :].set(interior_noise)
    new_state = eqx.tree_at(lambda s: s.temp, state0.state, state0.state.temp + noise)
    return eqx.tree_at(lambda s: s.state, state0, new_state)


def fit_one(final_state, loss, update, schedule, start_c_k, n_iterations):
    """(fitted_c_k, loss_start, loss_end), reusing (length, repeat)-scoped jitted closures
    across every distance -- one compile per (length, repeat), not per (length, repeat, distance)."""
    log_c_k = jnp.log(jnp.array(start_c_k))
    loss_start = loss_end = None
    for it in range(n_iterations):
        log_c_k, value = update(log_c_k, schedule(it))
        if it == 0:
            loss_start = float(value)
        if not np.isfinite(float(value)):
            break
        loss_end = float(value)

    return float(jnp.exp(log_c_k)), loss_start, loss_end


def main(lengths, distances, n_repeats, n_iterations, step_size):
    model, state0, forcing_fn = common.spinup()
    keys = jax.random.split(jax.random.PRNGKey(ROUNDOFF_SEED), n_repeats)
    schedule = optax.cosine_decay_schedule(step_size, n_iterations)

    shape = (len(lengths), len(distances), n_repeats)
    fitted = np.full(shape, np.nan)
    loss_start = np.full(shape, np.nan)
    loss_end = np.full(shape, np.nan)

    for i, n in enumerate(lengths):
        for r in range(n_repeats):
            t0 = time.time()
            member_state0 = state0 if r == 0 else perturbed_state0(state0, model, keys[r])

            @eqx.filter_jit
            def final_state(params, member_state0=member_state0, n=n):
                model_p = common.with_params(model, {**FIXED, **params})
                return common.rollout(model_p, member_state0, forcing_fn, n).state

            true_final = final_state({"c_k": jnp.array(TRUE_C_K)})
            target_temp, target_salt = common.upper_ts(true_final, N_LAYERS)
            scale = common.upper_ts_scale(model, true_final, N_LAYERS)

            def loss(log_c_k, final_state=final_state, target_temp=target_temp, target_salt=target_salt, scale=scale):
                return common.upper_ts_misfit(model, final_state({"c_k": jnp.exp(log_c_k)}), target_temp, target_salt, scale)

            @eqx.filter_jit
            def update(log_c_k, radius, loss=loss):
                value, grad = eqx.filter_value_and_grad(loss)(log_c_k)
                return log_c_k - radius * jnp.sign(grad), value

            for j, dist in enumerate(distances):
                start = TRUE_C_K * float(np.exp(dist))  # always the over-estimate side
                f, ls, le = fit_one(final_state, loss, update, schedule, start, n_iterations)
                fitted[i, j, r], loss_start[i, j, r], loss_end[i, j, r] = f, ls, le
            rel = np.abs(fitted[i, :, r] - TRUE_C_K) / TRUE_C_K
            print(f"n={n:5d}d  repeat={r}  param err over distances={np.round(rel, 4)}  ({time.time() - t0:.0f}s)")

    common.save(
        "exp3b_calibration_distance",
        lengths=lengths, distances=distances, n_repeats=n_repeats, true_c_k=TRUE_C_K,
        fitted=fitted, loss_start=loss_start, loss_end=loss_end,
        n_layers=N_LAYERS, roundoff_std=ROUNDOFF_STD,
        dt_tracer=model.config.dt_tracer, n_iterations=n_iterations, step_size=step_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--distances", type=float, nargs="+", default=DISTANCES)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-iterations", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=0.05,
                        help="trust radius per iteration, in log-parameter units")
    args = parser.parse_args()
    main(args.lengths, args.distances, args.n_repeats, args.n_iterations, args.step_size)
