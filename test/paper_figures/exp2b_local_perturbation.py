"""
Experiment 2b -- how does a local perturbation spread, and how fast?

Figure 2 asks how a *parameter* shows up across the whole field. This asks
the opposite question: pin a delta at a single ocean grid point of one
prognostic field (temperature, at the surface) at t=0, and use forward mode
to propagate d(field at time t)/d(that one point) forward -- a spatial
sensitivity kernel, watched spreading outward snapshot by snapshot. One
jax.jvp gives the whole time series at about the cost of a second forward
integration, no tape.

The rollout is short (N_STEPS days, well inside figure 1's horizon) since the
question is about the early spreading footprint, not about long-horizon
validity.

    python test/paper_figures/exp2b_local_perturbation.py

A few minutes on a laptop CPU.
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import common
from mini_veros import loop

N_STEPS = 20  # days -- short, well inside the horizon
N_SNAPSHOTS = 5
PERTURBED_FIELD = "temp"
PERTURB_LEVEL = -1  # top level (surface)


def find_ocean_point(model):
    """An interior (i, j) grid index whose PERTURB_LEVEL is ocean, near the domain centre."""
    mask = np.asarray(common.interior(model.boundary_conditions.maskT)[..., PERTURB_LEVEL])
    nx, ny = mask.shape
    i0, j0 = nx // 2, ny // 2
    ocean = np.argwhere(mask != 0)
    d2 = (ocean[:, 0] - i0) ** 2 + (ocean[:, 1] - j0) ** 2
    i, j = ocean[np.argmin(d2)]
    return int(i), int(j)


def main(n_steps, n_snapshots):
    model, state0, forcing_fn = common.spinup()
    assert n_steps % n_snapshots == 0, "n_steps must be a multiple of n_snapshots"
    i, j = find_ocean_point(model)
    print(f"perturbing {PERTURBED_FIELD} at interior grid point (i={i}, j={j}), level={PERTURB_LEVEL}")

    def snapshots(state0):
        def block(integrator_state, _):
            def one_step(integrator_state, _):
                force = forcing_fn(model, integrator_state.state)
                return loop.step(model, integrator_state, force), None

            integrator_state, _ = lax.scan(one_step, integrator_state, length=n_steps // n_snapshots)
            return integrator_state, common.interior(integrator_state.state.temp)

        _, temps = lax.scan(block, state0, length=n_snapshots)
        return temps

    zero_tangent = jax.tree_util.tree_map(jnp.zeros_like, state0)
    field = getattr(state0.state, PERTURBED_FIELD)
    delta = jnp.zeros_like(field).at[i + 2, j + 2, PERTURB_LEVEL].set(1.0)
    tangent0 = eqx.tree_at(lambda s: getattr(s.state, PERTURBED_FIELD), zero_tangent, delta)

    temp, sensitivity = jax.jvp(eqx.filter_jit(snapshots), (state0,), (tangent0,))
    temp, sensitivity = np.asarray(temp), np.asarray(sensitivity)

    steps = np.arange(1, n_snapshots + 1) * (n_steps // n_snapshots)
    x, y, land = common.grid_arrays(model)
    for k, s in enumerate(steps):
        surf = sensitivity[k, ..., -1]
        print(f"step {s:4d}d  max|d(surface temp)/d(point perturbation)|={np.abs(surf).max():.3e}  "
              f"footprint (|.|>1% of max) = {int((np.abs(surf) > 0.01 * np.abs(surf).max()).sum())} cells")

    common.save(
        "exp2b_local_perturbation",
        n_steps=n_steps, n_snapshots=n_snapshots, steps=steps,
        perturbed_field=PERTURBED_FIELD, perturb_level=PERTURB_LEVEL, point_ij=[i, j],
        temp=temp, sensitivity=sensitivity,
        x=x, y=y, land=land, dt_tracer=model.config.dt_tracer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--n-snapshots", type=int, default=N_SNAPSHOTS)
    args = parser.parse_args()
    main(args.n_steps, args.n_snapshots)
