"""
Experiment 3 -- recovering two hidden mixing parameters from surface temperature and salinity.

A twin experiment. The "truth" is one rollout of the settled model with the
reference TKE parameters (c_k, c_eps); all we keep of it is temperature and
salinity on the top 3 layers, at the *final* step only -- the way a real
observing system would give you a snapshot of the upper ocean, not a
satellite-like time series of SST. Starting from a deliberately wrong pair, we
minimise

    L(c_k, c_eps) = upper_ts_misfit(final(c_k, c_eps), final(c_k_true, c_eps_true))

on the gradient autodiff gives us through the whole rollout, and record every
iterate. Separately the same loss is evaluated on a grid, so the figure can put
the optimiser's path on the landscape it walked.

Why top-3-layer T,S rather than SST: c_k and c_eps set the strength of
turbulent mixing, which is felt as upper-ocean *stratification* -- the density
contrast between the surface and the layers just below -- not as the surface
value alone. SST by itself does not see that contrast; observing salinity too,
and more than one layer, should.

Two choices worth explaining:

* The rollout is short (about half of figure 1's horizon H). Past H, figure 1
  shows the gradient disagreeing with finite differences; figure 4 is what
  happens as the rollout is pushed past that horizon instead.
* The step rule is steepest descent with a trust radius in log-parameter
  space: each iteration moves log(c_k) and log(c_eps) by `step_size`, in the
  direction of the gradient's sign, with a cosine schedule taking the radius to
  zero. Logs because the two constants differ by a factor 7 and their gradients
  by a factor 30. The sign rather than the magnitude because the gradient
  spikes (figure 1), and a step that trusts magnitude lets one bad gradient
  throw the fit off for a dozen iterations; with a bounded step it costs
  exactly one. The same rule fits the initial state in figure 5.

    python test/paper_figures/exp3_calibration.py

The landscape dominates the cost (one rollout per grid point).
"""

import argparse
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

import common

TRUE = {"tke_closure.c_k": 0.10, "tke_closure.c_eps": 0.70}
GUESS = {"tke_closure.c_k": 0.20, "tke_closure.c_eps": 0.45}
C_K_RANGE = (0.04, 0.26)
C_EPS_RANGE = (0.30, 1.10)
N_LAYERS = 3


def main(n_steps, n_grid, n_iterations, step_size):
    model, state0, forcing_fn = common.spinup()
    names = list(TRUE)

    @eqx.filter_jit
    def final_state(params):
        model_p = common.with_params(model, params)
        return common.rollout(model_p, state0, forcing_fn, n_steps).state

    as_params = lambda log_params: {k: jnp.exp(log_params[k]) for k in names}

    true_final = final_state({k: jnp.array(v) for k, v in TRUE.items()})
    target_temp, target_salt = common.upper_ts(true_final, N_LAYERS)
    scale = common.upper_ts_scale(model, true_final, N_LAYERS)
    guess_final = final_state({k: jnp.array(v) for k, v in GUESS.items()})

    def loss(params):
        return common.upper_ts_misfit(model, final_state(params), target_temp, target_salt, scale)

    schedule = optax.cosine_decay_schedule(step_size, n_iterations)

    @eqx.filter_jit
    def update(log_params, radius):
        value, grads = eqx.filter_value_and_grad(lambda lp: loss(as_params(lp)))(log_params)
        return {k: log_params[k] - radius * jnp.sign(grads[k]) for k in names}, value

    log_params = {k: jnp.log(jnp.array(v)) for k, v in GUESS.items()}
    trajectory, losses = [[GUESS[k] for k in names]], []

    t0 = time.time()
    for i in range(n_iterations):
        log_params, value = update(log_params, schedule(i))
        params = as_params(log_params)
        trajectory.append([float(params[k]) for k in names])
        losses.append(float(value))
        if i % 20 == 0 or i == n_iterations - 1:
            print(f"iter {i:4d}  " + "  ".join(f"{k}={float(params[k]):.5f}" for k in names)
                  + f"  loss={losses[-1]:.4e}  ({time.time() - t0:.0f}s)")

    fitted = as_params(log_params)
    print(f"fitted {[float(fitted[k]) for k in names]} vs true {[TRUE[k] for k in names]}")
    fitted_final = final_state(fitted)

    # Loss landscape on a grid, for the figure's contours.
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
        "exp3_calibration",
        n_steps=n_steps, n_layers=N_LAYERS, param_names=names,
        true=[TRUE[k] for k in names], guess=[GUESS[k] for k in names],
        fitted=[float(fitted[k]) for k in names],
        sst_true=common.sst(true_final), sst_guess=common.sst(guess_final), sst_fitted=common.sst(fitted_final),
        trajectory=trajectory, losses=losses,
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
                        help="trust radius per iteration, in log-parameter units")
    args = parser.parse_args()
    n_steps = args.n_steps
    if n_steps is None:
        if common.HORIZON_DAYS is None:
            raise SystemExit("common.HORIZON_DAYS is not set yet -- run exp1 first, or pass --n-steps explicitly")
        n_steps = common.HORIZON_DAYS // 2
    main(n_steps, args.n_grid, args.n_iterations, args.step_size)
