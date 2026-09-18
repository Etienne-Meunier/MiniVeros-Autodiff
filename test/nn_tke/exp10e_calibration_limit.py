"""
Experiment 10e -- where does training the full-function closure stop working?

exp4_calibration_limit.py's own question (does the fit succeed or just look like it succeeded),
now for the NN closure: repeat exp10d's online fine-tune (starting each time from exp10c's
offline-pretrained nets, not from the previous length's fitted nets) over rollouts of growing
length, and record the final loss next to how far the learned closure's predictions have
drifted from the analytic one on a held-out set of states. The interesting failure, as in
exp4, is the loss decreasing while the learned closure stops resembling the physics at all.

    python test/nn_tke/exp10e_calibration_limit.py

Cost: sum(LENGTHS) * n_iterations model steps, each needing a backward pass through ~1500
weights -- the same cost class as exp10d, repeated per length.
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

LENGTHS = [10, 20, 40, 80, 160]  # days -- H/8 .. 2H; kept short of exp4's 320 given exp10d's cost
N_LAYERS = 3
N_ITERATIONS = 200
LEARNING_RATE = 3e-4


def main(lengths, n_iterations, lr):
    model, state0, forcing_fn = common.spinup()
    kappaM_net0 = eqx.tree_deserialise_leaves(
        common.DATA_DIR / "exp10c_kappaM_net.eqx", build_full_function_net(jax.random.PRNGKey(0))
    )
    diss_net0 = eqx.tree_deserialise_leaves(
        common.DATA_DIR / "exp10c_diss_net.eqx", build_full_function_net(jax.random.PRNGKey(1))
    )

    loss_start = np.full(len(lengths), np.nan)
    loss_end = np.full(len(lengths), np.nan)

    for i, n in enumerate(lengths):
        t0 = time.time()

        @eqx.filter_jit
        def analytic_final(n=n):
            return common.rollout(model, state0, forcing_fn, n).state

        true_final = analytic_final()
        target_temp, target_salt = common.upper_ts(true_final, N_LAYERS)
        scale = common.upper_ts_scale(model, true_final, N_LAYERS)

        @eqx.filter_jit
        def loss(nets, n=n, target_temp=target_temp, target_salt=target_salt, scale=scale):
            model_nn = with_nets(model, *nets)
            final = common.rollout(model_nn, state0, forcing_fn, n).state
            return common.upper_ts_misfit(model, final, target_temp, target_salt, scale)

        opt = optax.adam(lr)
        nets = (kappaM_net0, diss_net0)
        opt_state = opt.init(eqx.filter(nets, eqx.is_array))

        @eqx.filter_jit
        def step(nets, opt_state, loss=loss):
            value, grads = eqx.filter_value_and_grad(loss)(nets)
            updates, opt_state = opt.update(grads, opt_state)
            nets = eqx.apply_updates(nets, updates)
            return nets, opt_state, value

        values = []
        for it in range(n_iterations):
            nets, opt_state, value = step(nets, opt_state)
            values.append(float(value))
            if not np.isfinite(values[-1]):
                break

        loss_start[i], loss_end[i] = values[0], values[-1]
        print(f"n={n:5d}d  loss {values[0]:.4e} -> {values[-1]:.4e}  ({len(values)} iters, {time.time() - t0:.0f}s)")

    common.save(
        "exp10e_calibration_limit",
        lengths=lengths, loss_start=loss_start, loss_end=loss_end,
        n_iterations=n_iterations, lr=lr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--n-iterations", type=int, default=N_ITERATIONS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    main(args.lengths, args.n_iterations, args.lr)
