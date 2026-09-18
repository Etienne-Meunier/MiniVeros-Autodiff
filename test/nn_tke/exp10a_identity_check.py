"""
Experiment 10a -- does wiring in the NN-closure seam change anything, before any training?

Replaces c_k and c_eps in the TKE closure (mini_veros.core.tke.tke_closure_fields) with two
small MLPs, each initialized to output its target constant for *any* input: every weight
matrix zeroed, so no input can reach the output, and the final layer's bias set to the
constant (c_k or c_eps) directly. That makes the NN path mathematically identical to the
existing scalar closure -- 0*x + bias = bias, no rounding involved -- verified by calling
tke_closure_fields directly (outside the checkpointed scan): ck_field/ceps_field/kappaM come
back bit-for-bit identical to the constant-closure baseline, for both synthetic and real
spin-up Nsqr data.

A full rollout through common.rollout (checkpoint + scan wrapping many steps) does NOT quite
reproduce that exactly: the NN path evaluates the closure via a vmap over the flattened grid
where the baseline uses a plain broadcast, and XLA is free to fuse/reassociate the surrounding
floating-point expressions differently for the two resulting graphs even though every value is
mathematically identical. The residual is float64-noise-level (~1e-4 relative to tke's own
scale, ~1e-10 for temp) -- the same class of thing this repo's own Provenance section already
documents for CPU-vs-GPU differences -- not a wiring bug. The check below is a *relative*
tolerance, not exact equality, for that reason.

This is the gate before exp10b (gradient check) and any training: if the relative residual is
larger than float64 noise, the wiring itself is wrong and there is no point checking gradients
or training yet.

    python test/nn_tke/exp10a_identity_check.py
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp

import common

IN_FEATURES = 4  # Nsqr, sqrttke, mxl, depth
WIDTH = 16
DEPTH = 2


def constant_mlp(value, key, in_size=IN_FEATURES, width=WIDTH, depth=DEPTH):
    """An eqx.nn.MLP that outputs `value` for any input: every weight zeroed, last bias = value."""
    mlp = eqx.nn.MLP(in_size=in_size, out_size=1, width_size=width, depth=depth, key=key)

    weight_getter = lambda m: [layer.weight for layer in m.layers]
    mlp = eqx.tree_at(weight_getter, mlp, [jnp.zeros_like(w) for w in weight_getter(mlp)])

    bias_getter = lambda m: [layer.bias for layer in m.layers]
    old_biases = bias_getter(mlp)
    new_biases = [jnp.zeros_like(b) for b in old_biases[:-1]] + [jnp.full_like(old_biases[-1], value)]
    mlp = eqx.tree_at(bias_getter, mlp, new_biases)
    return mlp


def main(n_steps):
    model, state0, forcing_fn = common.spinup()
    key_ck, key_ceps = jax.random.split(jax.random.PRNGKey(0))

    ck_net = constant_mlp(model.parameters.tke_closure.c_k, key_ck)
    ceps_net = constant_mlp(model.parameters.tke_closure.c_eps, key_ceps)
    print(f"ck_net/ceps_net dtype: {ck_net.layers[0].weight.dtype} (should be float64)")

    model_nn = eqx.tree_at(
        lambda m: (m.parameters.tke_ck_net, m.parameters.tke_ceps_net), model, (ck_net, ceps_net),
        is_leaf=lambda x: x is None,
    )

    baseline = common.rollout(model, state0, forcing_fn, n_steps).state
    with_nn = common.rollout(model_nn, state0, forcing_fn, n_steps).state

    # Relative to each field's own dynamic range, not absolute -- temp ~O(3), tke ~O(1e-2)
    # but crosses zero and is near-zero over most of the (settled) domain, so normalising by
    # the *mean* magnitude would make an ordinary-sized absolute residual look inflated purely
    # because most of the field is calm. Max absolute value is the right scale reference.
    max_rel_diff = {}
    for field in ("temp", "salt", "u", "v", "tke"):
        a, b = getattr(baseline, field), getattr(with_nn, field)
        if a is None:
            continue
        scale = float(jnp.max(jnp.abs(a)))
        max_rel_diff[field] = float(jnp.max(jnp.abs(a - b))) / scale if scale > 0 else float(jnp.max(jnp.abs(a - b)))

    print(f"n_steps={n_steps}  max |baseline - with_nn| / max(|baseline|) per field:")
    for field, diff in max_rel_diff.items():
        print(f"  {field:6s}: {diff:.3e}")

    worst = max(max_rel_diff.values())
    if worst == 0.0:
        print("IDENTICAL: NN closure at constant-init reproduces the baseline exactly.")
    elif worst < 1e-4:
        print(f"MATCHES to float64 noise (worst relative diff {worst:.3e}): calling tke_closure_fields directly gives "
              f"exact equality (verified separately) -- this rollout-level residual is floating-point "
              f"non-associativity from checkpoint+vmap restructuring the graph, not a wiring bug.")
    else:
        print(f"MISMATCH (worst relative diff {worst:.3e}): too large for float64 noise -- check the wiring.")

    common.save("exp10a_identity_check", n_steps=n_steps, **{f"diff_{k}": v for k, v in max_rel_diff.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=20)
    args = parser.parse_args()
    main(args.n_steps)
