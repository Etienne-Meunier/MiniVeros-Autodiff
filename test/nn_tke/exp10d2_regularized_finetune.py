"""
Experiment 10d2 -- does regularizing toward the pretrained closure fix the null-space drift?

fig10f (exp10d's own diagnostic) shows the plain BPTT fine-tune drifted the closure's *deep*
profile far from the analytic one -- kappaM/dissipation both collapse toward a nearly-uniform
value with depth -- even though the training loss (T,S on the top 3 layers) dropped 16.7x. This
is the same null space exp5_assimilation.py already found for initial-state assimilation from
shallow observations: the loss barely sees what happens below the observed layers, so nothing
stops the optimizer from moving the closure's behavior there to whatever's locally convenient.

This repeats exp10d's exact setup with one addition: an L2 penalty pulling every weight back
toward its offline-pretrained value (weight space, not output space -- cheap, no need to carry
exp10c's dataset along). REG_LAMBDA=0 exactly reproduces exp10d; the question is whether a
lambda exists that both improves the training loss AND keeps the closure physically sensible
where the loss doesn't constrain it -- an explicit prior standing in for the missing deep
observations, not a fix to the null space itself (which is a property of the data, not the
optimizer).

    python test/nn_tke/exp10d2_regularized_finetune.py --reg-lambda 1.0

Cost: same as exp10d (the regularizer is O(n_weights), negligible next to the rollout).

The first run of this (reg_lambda=1.0, no clipping) diverged to NaN on its *last* iteration
(299/300) after 280 iterations of clean, steadily-decreasing loss -- a late gradient spike, the
same kind of thing exp6_gradient_spikes.py documents for the scalar closure, generalized to NN
weights. Without keeping the last finite weights separately, that overwrote 5+ minutes of good
training with garbage; now fixed (best_nets below) plus optax.clip_by_global_norm as the actual
mitigation, matching this experiment's own original speculation about gradient clipping.
"""

import argparse
import time

import equinox as eqx
import jax
import numpy as np
import optax

import common
from exp10d_online_finetune import with_nets
from nn_utils import build_full_function_net

N_STEPS = 40
N_LAYERS = 3
N_ITERATIONS = 300
LEARNING_RATE = 3e-4
REG_LAMBDA = 1.0
GRAD_CLIP_NORM = 1.0  # a late-training gradient spike (exp6's own story, generalized to NN
                       # weights) can blow up Adam's momentum state; clip rather than let it


def pytree_sq_dist(a, b):
    arrays_a, _ = eqx.partition(a, eqx.is_array)
    arrays_b, _ = eqx.partition(b, eqx.is_array)
    return sum(jax.numpy.sum((la - lb) ** 2)
               for la, lb in zip(jax.tree_util.tree_leaves(arrays_a), jax.tree_util.tree_leaves(arrays_b)))


def main(n_steps, n_iterations, lr, reg_lambda, grad_clip_norm):
    model, state0, forcing_fn = common.spinup()
    kappaM_net0 = eqx.tree_deserialise_leaves(
        common.DATA_DIR / "exp10c_kappaM_net.eqx", build_full_function_net(jax.random.PRNGKey(0))
    )
    diss_net0 = eqx.tree_deserialise_leaves(
        common.DATA_DIR / "exp10c_diss_net.eqx", build_full_function_net(jax.random.PRNGKey(1))
    )
    nets0 = (kappaM_net0, diss_net0)

    @eqx.filter_jit
    def analytic_final():
        return common.rollout(model, state0, forcing_fn, n_steps).state

    true_final = analytic_final()
    target_temp, target_salt = common.upper_ts(true_final, N_LAYERS)
    scale = common.upper_ts_scale(model, true_final, N_LAYERS)

    @eqx.filter_jit
    def data_loss(nets):
        model_nn = with_nets(model, *nets)
        final = common.rollout(model_nn, state0, forcing_fn, n_steps).state
        return common.upper_ts_misfit(model, final, target_temp, target_salt, scale)

    @eqx.filter_jit
    def loss(nets):
        return data_loss(nets) + reg_lambda * pytree_sq_dist(nets, nets0)

    opt = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr))
    nets = nets0
    opt_state = opt.init(eqx.filter(nets, eqx.is_array))

    @eqx.filter_jit
    def step(nets, opt_state):
        value, grads = eqx.filter_value_and_grad(loss)(nets)
        updates, opt_state = opt.update(grads, opt_state)
        nets = eqx.apply_updates(nets, updates)
        return nets, opt_state, value

    start_data_loss = float(data_loss(nets))
    print(f"n_steps={n_steps}  reg_lambda={reg_lambda}  grad_clip_norm={grad_clip_norm}  start data_loss={start_data_loss:.4e}")

    # Track the last *finite* nets separately from `nets` itself -- a late divergence must not
    # overwrite the good weights that got us there (see this experiment's own docstring update
    # after the first run hit exactly this on its final iteration).
    best_nets = nets
    data_losses = [start_data_loss]
    t0 = time.time()
    for it in range(n_iterations):
        nets, opt_state, value = step(nets, opt_state)
        dl = float(data_loss(nets))
        data_losses.append(dl)
        if it % 20 == 0 or it == n_iterations - 1:
            print(f"iter {it:4d}  total_loss={float(value):.4e}  data_loss={dl:.4e}  ({time.time() - t0:.0f}s)")
        if not np.isfinite(dl):
            print(f"loss diverged at iter {it}, stopping -- keeping the last finite weights")
            break
        best_nets = nets

    kappaM_net_trained, diss_net_trained = best_nets
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(common.DATA_DIR / "exp10d2_kappaM_net.eqx", kappaM_net_trained)
    eqx.tree_serialise_leaves(common.DATA_DIR / "exp10d2_diss_net.eqx", diss_net_trained)
    final_loss = float(data_loss(best_nets))
    common.save("exp10d2_regularized_finetune", data_losses=np.array(data_losses),
                n_steps=n_steps, n_iterations=n_iterations, lr=lr, reg_lambda=reg_lambda)
    print(f"wrote exp10d2_{{kappaM,diss}}_net.eqx  final data_loss={final_loss:.4e} (start {start_data_loss:.4e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--n-iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--reg-lambda", type=float, default=REG_LAMBDA)
    parser.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM)
    args = parser.parse_args()
    main(args.n_steps, args.n_iterations, args.lr, args.reg_lambda, args.grad_clip_norm)
