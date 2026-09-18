"""
Experiment 4 -- how long a rollout can you still calibrate against?

The same twin experiment as figure 3, repeated over rollouts of growing length
and from several wrong starting points. What we record per (length, start) is
the distance between the fitted and the true parameter, next to the loss the
optimiser actually reached -- the interesting failure is the one where the loss
goes down and the parameter does not.

The observation operator is figure 3's -- temperature and salinity on the top
3 layers at the final step -- so the only thing changing across the sweep is
the rollout length. Only c_k is fitted, with c_eps held at its true value:
fitting one parameter that is identifiable on its own keeps the rollout length
as the only possible cause of failure.

Step rule as in figure 3: steepest descent in log-parameter space with a trust
radius, taking the sign of the gradient rather than its magnitude, on a cosine
schedule that ends at zero. The sign matters here more than anywhere -- past the
blow-up length the gradient is not merely large but wrong, and a rule that
trusts its magnitude produces a divergence rather than the failure to converge
this figure is about.

    python test/paper_figures/exp4_calibration_limit.py

Cost is sum(lengths) * n_iterations * len(STARTS) model steps (each fit needs
its own truth rollout at that length too), so it grows fast with the longest
length in LENGTHS.
"""

import argparse
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

import common

TRUE = {"tke_closure.c_k": 0.10, "tke_closure.c_eps": 0.70}
FIT = ["tke_closure.c_k"]  # the entries of TRUE the optimiser is allowed to move
STARTS = [{"tke_closure.c_k": 0.20}, {"tke_closure.c_k": 0.06}]
LENGTHS = [10, 20, 40, 80, 160, 320]  # days: H/8 .. 4H, H = common.HORIZON_DAYS = 80
N_LAYERS = 3


def main(lengths, n_iterations, step_size):
    model, state0, forcing_fn = common.spinup()
    truth = np.array([TRUE[k] for k in FIT])
    fixed = {k: jnp.array(v) for k, v in TRUE.items()}

    shape = (len(lengths), len(STARTS))
    fitted = np.full(shape + (len(FIT),), np.nan)
    loss_start = np.full(shape, np.nan)
    loss_end = np.full(shape, np.nan)

    for i, n in enumerate(lengths):

        @eqx.filter_jit
        def final_state(params, n=n):
            model_p = common.with_params(model, {**fixed, **params})
            return common.rollout(model_p, state0, forcing_fn, n).state

        true_final = final_state({})
        target_temp, target_salt = common.upper_ts(true_final, N_LAYERS)
        scale = common.upper_ts_scale(model, true_final, N_LAYERS)

        def loss(log_params):
            params = {k: jnp.exp(log_params[k]) for k in FIT}
            return common.upper_ts_misfit(model, final_state(params), target_temp, target_salt, scale)

        schedule = optax.cosine_decay_schedule(step_size, n_iterations)

        @eqx.filter_jit
        def update(log_params, radius):
            value, grads = eqx.filter_value_and_grad(loss)(log_params)
            return {k: log_params[k] - radius * jnp.sign(grads[k]) for k in FIT}, value

        for j, start in enumerate(STARTS):
            t0 = time.time()
            log_params = {k: jnp.log(jnp.array(start[k])) for k in FIT}
            values = []
            for it in range(n_iterations):
                log_params, value = update(log_params, schedule(it))
                values.append(float(value))
                if not np.isfinite(values[-1]):  # a diverged rollout ends this fit
                    break

            fitted[i, j] = [float(jnp.exp(log_params[k])) for k in FIT]
            loss_start[i, j], loss_end[i, j] = values[0], values[-1]
            rel = np.sqrt(np.mean(((fitted[i, j] - truth) / truth) ** 2))
            print(f"n={n:5d} start={j}  fitted={np.round(fitted[i, j], 5)}  param err={rel:.4f}  "
                  f"loss {values[0]:.2e} -> {values[-1]:.2e}  ({len(values)} iters, {time.time() - t0:.0f}s)")

    common.save(
        "exp4_calibration_limit",
        lengths=lengths, param_names=FIT, true=truth, n_layers=N_LAYERS,
        starts=[[start[k] for k in FIT] for start in STARTS],
        fitted=fitted, loss_start=loss_start, loss_end=loss_end,
        dt_tracer=model.config.dt_tracer, n_iterations=n_iterations,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--n-iterations", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=0.05,
                        help="trust radius per iteration, in log-parameter units")
    args = parser.parse_args()
    main(args.lengths, args.n_iterations, args.step_size)
