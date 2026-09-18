"""
Experiment 10d -- train the full-function closure by backpropagating through the rollout.

The centerpiece: starting from exp10c's offline-pretrained kappaM_net/diss_net (a working
approximation of the analytic closure), fine-tune both nets' weights via BPTT against a twin
experiment -- exactly exp3_calibration.py's own setup (T,S on the top 3 layers at the final
step of a short rollout), except what's being fit is ~1500 NN weights instead of 2 scalars, and
the "true" state comes from the *analytic* closure rather than a different parameter value.
This is a genuine replacement of the closure's functional form, trained end-to-end through the
ocean model, not a coefficient fit.

N_STEPS defaults conservatively short (inside the horizon exp1/exp10b measured for the scalar
and NN closures respectively) -- past it, BPTT gradients aren't trustworthy for calibration
regardless of what is being fit (see report/paper/gradients_and_limits.md and
exp10b_gradient_check.py's own result).

Optimizer is Adam with global-norm gradient clipping, not exp3/exp4's sign-of-gradient trust
region: that rule exists because it's cheap insurance against a rollout-length gradient spike
wrecking a 2-scalar fit for a dozen iterations; for ~1500 weights the sign-only step direction
is a much cruder approximation of the true (high-dimensional) gradient, so clipping (not sign)
is the better mitigation here (see exp10d2_regularized_finetune.py's docstring: an unclipped
run of that experiment diverged to NaN on its very last iteration after 280 clean ones -- the
same late-spike failure mode exp6_gradient_spikes.py documents for the scalar closure). The
last-finite weights are kept separately from the training loop's running `nets` for the same
reason: a late divergence must not silently overwrite good weights with NaN.

    python test/nn_tke/exp10d_online_finetune.py

Cost: n_iterations rollouts of N_STEPS days, each needing one backward pass through ~1500
weights (dominated by the rollout itself, same cost class as exp3_calibration.py's own fit).
"""

import argparse
import time

import equinox as eqx
import jax
import numpy as np
import optax

import common
from nn_utils import build_full_function_net

N_STEPS = 40  # days -- H/2, matching exp3_calibration.py's own convention; revisit after exp10b
N_LAYERS = 3
N_ITERATIONS = 300
LEARNING_RATE = 3e-4
GRAD_CLIP_NORM = 1.0


def with_nets(model, kappaM_net, diss_net):
    return eqx.tree_at(
        lambda m: (m.parameters.tke_kappaM_net, m.parameters.tke_diss_net), model, (kappaM_net, diss_net),
        is_leaf=lambda x: x is None,
    )


def main(n_steps, n_iterations, lr, grad_clip_norm):
    model, state0, forcing_fn = common.spinup()
    kappaM_net = eqx.tree_deserialise_leaves(
        common.DATA_DIR / "exp10c_kappaM_net.eqx", build_full_function_net(jax.random.PRNGKey(0))
    )
    diss_net = eqx.tree_deserialise_leaves(
        common.DATA_DIR / "exp10c_diss_net.eqx", build_full_function_net(jax.random.PRNGKey(1))
    )

    @eqx.filter_jit
    def analytic_final():
        return common.rollout(model, state0, forcing_fn, n_steps).state

    true_final = analytic_final()
    target_temp, target_salt = common.upper_ts(true_final, N_LAYERS)
    scale = common.upper_ts_scale(model, true_final, N_LAYERS)

    @eqx.filter_jit
    def loss(nets):
        model_nn = with_nets(model, *nets)
        final = common.rollout(model_nn, state0, forcing_fn, n_steps).state
        return common.upper_ts_misfit(model, final, target_temp, target_salt, scale)

    opt = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr))
    nets = (kappaM_net, diss_net)
    opt_state = opt.init(eqx.filter(nets, eqx.is_array))

    @eqx.filter_jit
    def step(nets, opt_state):
        value, grads = eqx.filter_value_and_grad(loss)(nets)
        updates, opt_state = opt.update(grads, opt_state)
        nets = eqx.apply_updates(nets, updates)
        return nets, opt_state, value

    start_loss = float(loss(nets))
    print(f"n_steps={n_steps}  grad_clip_norm={grad_clip_norm}  start loss (pretrained, untrained-online) = {start_loss:.4e}")

    # Last-finite weights, tracked separately from the loop's running `nets` -- see this
    # script's own docstring for why (exp10d2 lost a good run to exactly this).
    best_nets = nets
    losses = [start_loss]
    t0 = time.time()
    for it in range(n_iterations):
        nets, opt_state, value = step(nets, opt_state)
        v = float(value)
        losses.append(v)
        if it % 20 == 0 or it == n_iterations - 1:
            print(f"iter {it:4d}  loss={v:.4e}  ({time.time() - t0:.0f}s)")
        if not np.isfinite(v):
            print(f"loss diverged at iter {it}, stopping -- keeping the last finite weights")
            break
        best_nets = nets

    kappaM_net_trained, diss_net_trained = best_nets
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(common.DATA_DIR / "exp10d_kappaM_net.eqx", kappaM_net_trained)
    eqx.tree_serialise_leaves(common.DATA_DIR / "exp10d_diss_net.eqx", diss_net_trained)
    final_loss = float(loss(best_nets))
    common.save("exp10d_online_finetune", losses=np.array(losses), n_steps=n_steps, n_iterations=n_iterations, lr=lr)
    print(f"wrote exp10d_{{kappaM,diss}}_net.eqx  final loss={final_loss:.4e} (start {start_loss:.4e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--n-iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM)
    args = parser.parse_args()
    main(args.n_steps, args.n_iterations, args.lr, args.grad_clip_norm)
