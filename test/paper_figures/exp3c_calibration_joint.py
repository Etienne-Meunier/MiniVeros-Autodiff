"""
Experiment 3c -- calibration success as a function of rollout length AND how far off (in 2D)
the starting guess is, fitting c_k and c_eps jointly.

Figure 4 (and its own follow-up, figure 3b) only ever fit c_k alone, with c_eps pinned at
truth. This fits both at once -- the two-parameter problem figure 3 actually posed -- and
places the starting guess on a circle of a given radius around the true (c_k, c_eps) point,
in *log*-parameter space (distance^2 = log(start_ck/true_ck)^2 + log(start_ceps/true_ceps)^2):
log space because c_k and c_eps differ in scale and gradient magnitude by an order of
magnitude, so a circle in raw units would be wildly lopsided in what it actually perturbs.
Rather than a full circle, each radius is checked at exactly ANGLES fixed, evenly-spaced
directions (0, 120, 240 degrees) -- enough to see whether the direction of the initial error
matters, without the cost of a continuous sweep or of repeats for error bars (there are none
here; see exp3b_calibration_distance.py for the round-off-ensemble version of this question).

The observation operator, step rule and truth are figure 3's own -- T,S on the top 3 layers at
the final step, steepest descent in log-parameter space with a sign-of-gradient step on a
cosine-decayed trust radius. Saved raw: the fitted (c_k, c_eps) themselves, not just an error
summary, so figure 3c can be re-plotted differently later without rerunning.

    python test/paper_figures/exp3c_calibration_joint.py

Cost is sum(LENGTHS) * len(DISTANCES) * len(ANGLES) * n_iterations model steps (one truth
rollout per length, shared across every distance and angle at that length).
"""

import argparse
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

import common

TRUE = {"tke_closure.c_k": 0.10, "tke_closure.c_eps": 0.70}
NAMES = list(TRUE)
LENGTHS = [10, 20, 40, 80, 120, 160, 240, 320]  # days: H/8 .. 4H, H = common.HORIZON_DAYS = 80
DISTANCES = [0.15, 0.3, 0.5, 0.7, 1.0]  # L2 radius in log-parameter space
ANGLES_DEG = [0.0, 120.0, 240.0]  # evenly spread, not random and not a full circle
N_LAYERS = 3


def start_point(distance, angle_deg):
    """(c_k, c_eps) at `distance` (log-space L2 radius) and `angle_deg` around TRUE."""
    theta = np.deg2rad(angle_deg)
    return {
        "tke_closure.c_k": TRUE["tke_closure.c_k"] * float(np.exp(distance * np.cos(theta))),
        "tke_closure.c_eps": TRUE["tke_closure.c_eps"] * float(np.exp(distance * np.sin(theta))),
    }


def main(lengths, distances, angles_deg, n_iterations, step_size):
    model, state0, forcing_fn = common.spinup()
    schedule = optax.cosine_decay_schedule(step_size, n_iterations)

    shape = (len(lengths), len(distances), len(angles_deg))
    fitted = np.full(shape + (len(NAMES),), np.nan)
    loss_start = np.full(shape, np.nan)
    loss_end = np.full(shape, np.nan)

    for i, n in enumerate(lengths):
        t0 = time.time()

        @eqx.filter_jit
        def final_state(params, n=n):
            return common.rollout(common.with_params(model, params), state0, forcing_fn, n).state

        true_final = final_state({k: jnp.array(v) for k, v in TRUE.items()})
        target_temp, target_salt = common.upper_ts(true_final, N_LAYERS)
        scale = common.upper_ts_scale(model, true_final, N_LAYERS)

        def loss(log_params, final_state=final_state, target_temp=target_temp, target_salt=target_salt, scale=scale):
            params = {k: jnp.exp(log_params[k]) for k in NAMES}
            return common.upper_ts_misfit(model, final_state(params), target_temp, target_salt, scale)

        @eqx.filter_jit
        def update(log_params, radius, loss=loss):
            value, grads = eqx.filter_value_and_grad(loss)(log_params)
            return {k: log_params[k] - radius * jnp.sign(grads[k]) for k in NAMES}, value

        for j, dist in enumerate(distances):
            for k, angle in enumerate(angles_deg):
                start = start_point(dist, angle)
                log_params = {name: jnp.log(jnp.array(start[name])) for name in NAMES}
                ls = le = None
                for it in range(n_iterations):
                    log_params, value = update(log_params, schedule(it))
                    if it == 0:
                        ls = float(value)
                    if not np.isfinite(float(value)):
                        break
                    le = float(value)
                fitted[i, j, k] = [float(jnp.exp(log_params[name])) for name in NAMES]
                loss_start[i, j, k], loss_end[i, j, k] = ls, le

            rel = np.abs(fitted[i, j] - np.array([TRUE[n_] for n_ in NAMES])) / np.array([TRUE[n_] for n_ in NAMES])
            print(f"n={n:5d}d  dist={dist:.2f}  param err by angle=\n{np.round(rel, 4)}  ({time.time() - t0:.0f}s)")

    common.save(
        "exp3c_calibration_joint",
        lengths=lengths, distances=distances, angles_deg=angles_deg, names=NAMES,
        true=[TRUE[n_] for n_ in NAMES],
        fitted=fitted, loss_start=loss_start, loss_end=loss_end,
        n_layers=N_LAYERS, dt_tracer=model.config.dt_tracer,
        n_iterations=n_iterations, step_size=step_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--distances", type=float, nargs="+", default=DISTANCES)
    parser.add_argument("--angles-deg", type=float, nargs="+", default=ANGLES_DEG)
    parser.add_argument("--n-iterations", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=0.05,
                        help="trust radius per iteration, in log-parameter units")
    args = parser.parse_args()
    main(args.lengths, args.distances, args.angles_deg, args.n_iterations, args.step_size)
