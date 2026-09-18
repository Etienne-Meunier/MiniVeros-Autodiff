"""
Experiment 10 -- recovering two hidden mixing parameters from a density-gradient snapshot.

Same twin experiment as figure 3 (c_k, c_eps against a snapshot of the settled model), but
the observable is potential density rather than temperature and salinity: the top-3-layer
density differences from the surface, prho[-3]-prho[-1] and prho[-2]-prho[-1]. c_k/c_eps set
mixing strength, which is felt directly as that density contrast; going through temperature
and salinity separately (figure 3) adds two fields' worth of noise the density combination
cancels. See common.density_gradient_misfit.

Three starting points rather than one, at the same distance from the truth in
log(c_k), log(c_eps) space but three different directions (ANGLES_DEG, 120 degrees apart,
none axis-aligned so every start moves both parameters) -- the same "does it converge
regardless of where it starts" story as the Vercor report's four-corner descents, with
three corners instead of four.

Step rule: Adam (optax.adam) on a cosine-decayed learning rate, gradient's global norm
clipped at `clip_factor` times its value at the start point before each descent -- same
recipe as the Vercor-Autodiff paper_calibration scripts. Unlike figure 3's sign-only
steepest descent, this uses the gradient's magnitude too, so it also tests whether the AD
gradient's scale (not just its sign) is useful for calibration; the per-start clip is what
keeps one spiky step (see figure 6) from throwing the fit off course.

    python test/paper_figures/exp10_density_calibration.py

The landscape dominates the cost (one rollout per grid point); the three descents are cheap
by comparison.
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
ANGLES_DEG = [40.0, 160.0, 280.0]  # directions of the 3 starts around the truth, log-parameter plane
N_LAYERS = 3


def starts_from(true, distance, angles_deg):
    """3 starting points at `distance` (|log(start/true)|) from `true`, one per angle in
    ANGLES_DEG -- direction (cos, sin) applied equally to log(c_k), log(c_eps)."""
    names = list(true)
    log_true = np.array([np.log(true[k]) for k in names])
    starts = []
    for angle in angles_deg:
        theta = np.deg2rad(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])
        log_start = log_true + distance * direction
        starts.append({k: float(np.exp(v)) for k, v in zip(names, log_start)})
    return starts


def main(n_steps, n_grid, n_iterations, step_size, clip_factor, distance, angles_deg):
    model, state0, forcing_fn = common.spinup()
    names = list(TRUE)
    starts = starts_from(TRUE, distance, angles_deg)

    @eqx.filter_jit
    def final_state(params):
        model_p = common.with_params(model, params)
        return common.rollout(model_p, state0, forcing_fn, n_steps)

    as_params = lambda log_params: {k: jnp.exp(log_params[k]) for k in names}

    true_result = final_state({k: jnp.array(v) for k, v in TRUE.items()})
    target = common.density_gradient(true_result, N_LAYERS)
    scale = common.density_gradient_scale(model, true_result, N_LAYERS)

    def loss(params):
        return common.density_gradient_misfit(model, final_state(params), target, scale, N_LAYERS)

    schedule = optax.cosine_decay_schedule(step_size, n_iterations)

    @eqx.filter_jit
    def value_and_grad(log_params):
        return eqx.filter_value_and_grad(lambda lp: loss(as_params(lp)))(log_params)

    trajectories, losses_all, fitted_all, guess_results = [], [], [], []
    for s, guess in enumerate(starts):
        log_params = {k: jnp.log(jnp.array(v)) for k, v in guess.items()}

        # Clip threshold measured at the start point rather than fixed, so one rule
        # transfers between starts whose gradient scale differs (paper_calibration/calibrate.py).
        value, grads = value_and_grad(log_params)
        start_norm = float(jnp.linalg.norm(jnp.array([grads[k] for k in names])))
        optimizer = optax.chain(optax.clip_by_global_norm(clip_factor * start_norm), optax.adam(schedule))
        opt_state = optimizer.init(log_params)

        trajectory, losses = [[guess[k] for k in names]], []

        t0 = time.time()
        for i in range(n_iterations):
            losses.append(float(value))
            updates, opt_state = optimizer.update(grads, opt_state)
            log_params = optax.apply_updates(log_params, updates)
            value, grads = value_and_grad(log_params)
            params = as_params(log_params)
            trajectory.append([float(params[k]) for k in names])
            if i % 20 == 0 or i == n_iterations - 1:
                print(f"start {s}  iter {i:4d}  " + "  ".join(f"{k}={float(params[k]):.5f}" for k in names)
                      + f"  loss={losses[-1]:.4e}  ({time.time() - t0:.0f}s)")

        fitted = as_params(log_params)
        print(f"start {s} fitted {[float(fitted[k]) for k in names]} vs true {[TRUE[k] for k in names]}")
        trajectories.append(trajectory)
        losses_all.append(losses)
        fitted_all.append([float(fitted[k]) for k in names])
        guess_results.append(final_state({k: jnp.array(v) for k, v in guess.items()}))

    fitted_results = [final_state({k: jnp.array(v) for k, v in zip(names, f)}) for f in fitted_all]

    # Loss landscape on a grid, for the figure's contours -- shared by all 3 descents.
    c_k = np.linspace(*C_K_RANGE, n_grid)
    c_eps = np.linspace(*C_EPS_RANGE, n_grid)
    landscape = np.zeros((n_grid, n_grid))
    t0 = time.time()
    for i, a in enumerate(c_k):
        for j, b in enumerate(c_eps):
            landscape[i, j] = loss({"tke_closure.c_k": jnp.array(a), "tke_closure.c_eps": jnp.array(b)})
        print(f"  landscape row {i + 1}/{n_grid} ({time.time() - t0:.0f}s)")

    x, y, land = common.grid_arrays(model)
    common.save(
        "exp10_density_calibration",
        n_steps=n_steps, n_layers=N_LAYERS, param_names=names, distance=distance, angles_deg=angles_deg,
        true=[TRUE[k] for k in names], guesses=[[g[k] for k in names] for g in starts], fitted=fitted_all,
        density_true=common.density_gradient_map(true_result),
        density_guess=np.stack([common.density_gradient_map(r) for r in guess_results]),
        density_fitted=np.stack([common.density_gradient_map(r) for r in fitted_results]),
        trajectories=np.array(trajectories), losses=np.array(losses_all),
        c_k=c_k, c_eps=c_eps, landscape=landscape,
        x=x, y=y, land=land,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=None,
                        help="defaults to common.HORIZON_DAYS // 2, once figure 1 has set it")
    parser.add_argument("--n-grid", type=int, default=21)
    parser.add_argument("--n-iterations", type=int, default=150)
    parser.add_argument("--step-size", type=float, default=0.05,
                        help="peak value of the cosine-decay learning-rate schedule (Adam)")
    parser.add_argument("--clip-factor", type=float, default=1.2,
                        help="clip the gradient's global norm at this multiple of its start-point value")
    parser.add_argument("--distance", type=float, default=0.5,
                        help="|log(start / true)| shared by the 3 starting points")
    parser.add_argument("--angles-deg", type=float, nargs=3, default=ANGLES_DEG,
                        help="direction of each start around the truth, in the "
                             "(log c_k, log c_eps) plane")
    args = parser.parse_args()
    n_steps = args.n_steps
    if n_steps is None:
        if common.HORIZON_DAYS is None:
            raise SystemExit("common.HORIZON_DAYS is not set yet -- run exp1 first, or pass --n-steps explicitly")
        n_steps = common.HORIZON_DAYS // 2
    main(n_steps, args.n_grid, args.n_iterations, args.step_size, args.clip_factor, args.distance, args.angles_deg)
