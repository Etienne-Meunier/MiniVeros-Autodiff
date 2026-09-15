"""
GPU smoke test -- does global_4deg run correctly and fast on the server's GPU?

Not one of the eight paper experiments. Checks three things in one short job:
  1. jax actually sees and uses the GPU (fails loudly otherwise).
  2. A short spin-up and a short rollout+gradient are numerically sane: the
     autodiff gradient of mean(SST^2) w.r.t. c_k should match a central finite
     difference to several digits, the same check exp1 does at full scale.
  3. Real per-step wall time on this GPU, to size the real spin-up and exp1
     before spending a full reservation on them.

    python test/paper_figures/gpu_smoke_test.py
"""

import time

import equinox as eqx
import jax
import jax.numpy as jnp

import common

SPINUP_STEPS = 500
ROLLOUT_STEPS = 40  # days
PARAM = "c_k"
REL_STEP = 1e-5


def main():
    devices = jax.devices()
    print(f"jax devices: {devices}")
    if devices[0].platform != "gpu":
        raise RuntimeError(f"expected a GPU device, jax sees {devices[0].platform}")

    model, state0, forcing_fn = common.build()

    t0 = time.time()
    state0, _ = eqx.filter_jit(__import__("mini_veros").loop.run)(
        model, state0, forcing_fn, lambda s: None, SPINUP_STEPS, SPINUP_STEPS
    )
    jax.block_until_ready(state0)
    spinup_s = time.time() - t0
    print(f"spin-up: {SPINUP_STEPS} steps in {spinup_s:.1f}s ({1000 * spinup_s / SPINUP_STEPS:.1f} ms/step, "
          f"includes one-time compile)")

    base = getattr(model.parameters, PARAM)

    @eqx.filter_jit
    def loss(value):
        model_p = common.with_params(model, {PARAM: value})
        return common.mean_square_sst(common.rollout(model_p, state0, forcing_fn, ROLLOUT_STEPS).state)

    grad = eqx.filter_jit(eqx.filter_grad(loss))

    t0 = time.time()
    ad = grad(base)
    jax.block_until_ready(ad)
    compile_and_grad_s = time.time() - t0

    t0 = time.time()
    ad = grad(base)
    jax.block_until_ready(ad)
    grad_s = time.time() - t0
    print(f"gradient (compiled): {ROLLOUT_STEPS} steps in {grad_s:.2f}s "
          f"({1000 * grad_s / ROLLOUT_STEPS:.1f} ms/step, first call incl. compile was {compile_and_grad_s:.1f}s)")

    h = REL_STEP * jnp.abs(base)
    fd = (loss(base + h) - loss(base - h)) / (2 * h)
    rel_err = abs(float(ad) - float(fd)) / abs(float(fd))
    print(f"d(mean_square_sst)/d({PARAM}) over {ROLLOUT_STEPS} days: ad={float(ad):+.6e} fd={float(fd):+.6e} "
          f"rel_err={rel_err:.2e}")

    if rel_err > 1e-2:
        raise RuntimeError(f"GPU gradient disagrees with finite differences by {rel_err:.2e} -- investigate before trusting a full run")
    print("OK: GPU gradient matches finite differences.")


if __name__ == "__main__":
    main()
