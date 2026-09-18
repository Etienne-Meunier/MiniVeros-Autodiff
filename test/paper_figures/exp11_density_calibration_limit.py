"""
Experiment 11 -- calibration limit, density-gradient version: rollout length vs starting
distance vs starting direction.

Figure 10's twin experiment (density-gradient loss, both c_k and c_eps jointly fitted),
repeated over 5 rollout lengths spanning two failure modes on either side of figure 1's
horizon H:

* too short (LENGTHS[0] = H/16): c_k and c_eps are only weakly identifiable from a snapshot
  this early -- several combinations should fit about as well, which should show up as a
  ridge rather than a bowl in the landscape.
* too long (LENGTHS[-1] = 4H): past H the gradient disagrees with finite differences (figure
  1); the fit can still drive the loss down while the parameters drift (figure 4's story).

LENGTHS[1:] = [H/2, H, 2H] bridge the two, H/2 matching figure 10's own rollout.

For each length: a (c_k, c_eps) loss landscape (one rollout per grid node, no gradient), and
for each of DISTANCES x 3 starting directions (ANGLES_DEG, as figure 10) one descent from
that start, keeping the full trajectory -- so a divergent descent's path is visibly frozen at
its blow-up point rather than continuing into NaN.

Step rule: Adam (optax.adam) on a cosine-decayed learning rate, gradient's global norm
clipped at `clip_factor` times its value at the start point before each descent -- same
recipe as the Vercor-Autodiff paper_calibration scripts (see exp10). Unlike figure 3/4's
sign-only steepest descent, this uses the gradient's magnitude too, so it also tests
whether the AD gradient's scale (not just its sign) is useful for calibration; the
per-start clip is what keeps one spiky step (figure 6) from throwing a fit off course.

Saved raw (not just a plot): landscape[length], trajectories[length, distance, direction],
fitted[length, distance, direction], so different figures can be made later without rerunning.

    python test/paper_figures/exp11_density_calibration_limit.py

Cost is dominated by the landscapes: len(LENGTHS) * n_grid^2 rollouts, the longest at 4H --
expect this to need a GPU node. The 5 lengths are independent (each does its own truth
rollout and landscape), so split one job per length with --lengths/--tag and merge:

    python test/paper_figures/exp11_density_calibration_limit.py --lengths 320 --tag len320
    ...one such job per length in LENGTHS...
    python test/paper_figures/exp11_merge.py
"""

import argparse
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

import common

TRUE = {"tke_closure.c_k": 0.10, "tke_closure.c_eps": 0.70}
C_K_RANGE = (0.04, 0.26)
C_EPS_RANGE = (0.30, 1.10)
LENGTHS = [5, 40, 80, 160, 320]  # days: H/16, H/2, H, 2H, 4H, H = common.HORIZON_DAYS = 80
DISTANCES = [0.15, 0.3, 0.5, 0.7, 1.0]  # |log(start / true)|
ANGLES_DEG = [40.0, 160.0, 280.0]  # as exp10_density_calibration
N_LAYERS = 3


def starts_from(true, distance, angles_deg, names):
    log_true = np.array([np.log(true[k]) for k in names])
    starts = []
    for angle in angles_deg:
        theta = np.deg2rad(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])
        log_start = log_true + distance * direction
        starts.append({k: float(np.exp(v)) for k, v in zip(names, log_start)})
    return starts


def fit_one(loss, value_and_grad, schedule, clip_factor, start, n_iterations, names):
    """(trajectory, losses, fitted). Adam with the gradient's global norm clipped at
    clip_factor times its start-point value (paper_calibration/calibrate.py) -- uses the
    gradient's magnitude, not just its sign, while the clip keeps one spiky step (figure 6)
    from throwing the descent off course. A non-finite loss freezes the trajectory at that
    point rather than letting NaN parameters overwrite it -- so a divergent descent's path
    still plots as a path that stops, not one that vanishes."""
    log_params = {k: jnp.log(jnp.array(start[k])) for k in names}
    value, grads = value_and_grad(log_params)
    start_norm = float(jnp.linalg.norm(jnp.array([grads[k] for k in names])))
    optimizer = optax.chain(optax.clip_by_global_norm(clip_factor * start_norm), optax.adam(schedule))
    opt_state = optimizer.init(log_params)

    trajectory = [[start[k] for k in names]]
    losses = []
    diverged = False
    for it in range(n_iterations):
        if not diverged:
            value = float(value)
            diverged = not np.isfinite(value)
            if not diverged:
                updates, opt_state = optimizer.update(grads, opt_state)
                log_params = optax.apply_updates(log_params, updates)
                value, grads = value_and_grad(log_params)
        losses.append(np.nan if diverged else value)
        trajectory.append([float(jnp.exp(log_params[k])) for k in names])
    fitted = [float(jnp.exp(log_params[k])) for k in names]
    return trajectory, losses, fitted


def main(lengths, distances, angles_deg, n_grid, n_iterations, step_size, clip_factor, tag):
    model, state0, forcing_fn = common.spinup()
    names = list(TRUE)
    c_k = np.linspace(*C_K_RANGE, n_grid)
    c_eps = np.linspace(*C_EPS_RANGE, n_grid)

    n_len, n_dist, n_dir = len(lengths), len(distances), len(angles_deg)
    landscape = np.full((n_len, n_grid, n_grid), np.nan)
    trajectories = np.full((n_len, n_dist, n_dir, n_iterations + 1, len(names)), np.nan)
    fitted = np.full((n_len, n_dist, n_dir, len(names)), np.nan)
    loss_start = np.full((n_len, n_dist, n_dir), np.nan)
    loss_end = np.full((n_len, n_dist, n_dir), np.nan)

    for i, n in enumerate(lengths):
        t0 = time.time()

        @eqx.filter_jit
        def final_state(params, n=n):
            model_p = common.with_params(model, params)
            return common.rollout(model_p, state0, forcing_fn, n)

        true_result = final_state({k: jnp.array(v) for k, v in TRUE.items()})
        target = common.density_gradient(true_result, N_LAYERS)
        scale = common.density_gradient_scale(model, true_result, N_LAYERS)

        def loss(params, final_state=final_state, target=target, scale=scale):
            return common.density_gradient_misfit(model, final_state(params), target, scale, N_LAYERS)

        schedule = optax.cosine_decay_schedule(step_size, n_iterations)

        @eqx.filter_jit
        def value_and_grad(log_params, loss=loss):
            return eqx.filter_value_and_grad(lambda lp: loss({k: jnp.exp(lp[k]) for k in names}))(log_params)

        for gi, a in enumerate(c_k):
            for gj, b in enumerate(c_eps):
                landscape[i, gi, gj] = loss({"tke_closure.c_k": jnp.array(a), "tke_closure.c_eps": jnp.array(b)})
        print(f"n={n:5d}d  landscape done ({time.time() - t0:.0f}s)")

        for di, dist in enumerate(distances):
            starts = starts_from(TRUE, dist, angles_deg, names)
            for ai, start in enumerate(starts):
                t1 = time.time()
                traj, losses, fit = fit_one(loss, value_and_grad, schedule, clip_factor, start, n_iterations, names)
                trajectories[i, di, ai] = traj
                fitted[i, di, ai] = fit
                loss_start[i, di, ai] = losses[0] if np.isfinite(losses[0]) else np.nan
                loss_end[i, di, ai] = next((l for l in reversed(losses) if np.isfinite(l)), np.nan)
                rel = np.abs(np.array(fit) - np.array([TRUE[k] for k in names])) / np.array([TRUE[k] for k in names])
                print(f"  dist={dist:.2f} dir={ai}  fitted={np.round(fit, 4)}  rel err={np.round(rel, 4)}  "
                      f"loss {loss_start[i, di, ai]:.2e} -> {loss_end[i, di, ai]:.2e}  ({time.time() - t1:.0f}s)")

    common.save(
        f"exp11_density_calibration_limit__{tag}" if tag else "exp11_density_calibration_limit",
        lengths=lengths, distances=distances, angles_deg=angles_deg, n_iterations=n_iterations,
        param_names=names, true=[TRUE[k] for k in names], n_layers=N_LAYERS,
        c_k=c_k, c_eps=c_eps, landscape=landscape,
        trajectories=trajectories, fitted=fitted, loss_start=loss_start, loss_end=loss_end,
        dt_tracer=model.config.dt_tracer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--distances", type=float, nargs="+", default=DISTANCES)
    parser.add_argument("--angles-deg", type=float, nargs="+", default=ANGLES_DEG)
    parser.add_argument("--n-grid", type=int, default=21)
    parser.add_argument("--n-iterations", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=0.05,
                        help="peak value of the cosine-decay learning-rate schedule (Adam)")
    parser.add_argument("--clip-factor", type=float, default=1.2,
                        help="clip the gradient's global norm at this multiple of its start-point value")
    parser.add_argument("--tag", default=None,
                        help="save to exp11_density_calibration_limit__<tag>.npz instead of the merged "
                             "name -- for splitting the sweep into one job per length; "
                             "see exp11_merge.py")
    args = parser.parse_args()
    main(args.lengths, args.distances, args.angles_deg, args.n_grid, args.n_iterations, args.step_size,
         args.clip_factor, args.tag)
