"""
Experiment 2 -- where does a parameter's influence show up, and when?

Forward mode is the natural direction for this question: one scalar quantity
in, a whole 3-d field out. A single `jax.jvp` propagates dT(x, y, z)/dp
alongside the temperature itself, for about the cost of a second forward
integration and with no tape to store. We take snapshots along the rollout, so
the figure can show the sensitivity appearing and spreading.

Four rows, chosen for a physical response that is easy to recognise on a map:

  wind_stress_scale  -- a multiplier alpha on (surface_taux, surface_tauy),
                         tangent taken at alpha=1. Expect an Ekman-pumping
                         pattern.
  kappaH_min         -- background vertical diffusivity. Expect a thermocline
                         response concentrated in the tropics.
  heat_flux_scale    -- a multiplier alpha on forc_temp_surface, tangent at
                         alpha=1. Expect a response near deep-water formation.
  eke_c_k            -- the EKE closure's production constant. Expect a
                         response where eddy mixing is active (the ACC, the
                         western boundary currents).

The two forcing multipliers are *not* model.parameters entries -- they wrap
forcing_fn, so the tangent is w.r.t. a quantity that does not exist in the
model at all until this experiment adds it, and mini-veros itself is
untouched.

Rollout length defaults to common.HORIZON_DAYS (figure 1's horizon): past that
length figure 1 shows the derivative has stopped meaning anything about the
ocean, so a sensitivity map beyond it would show the tangent equations, not
the physics.

    python test/paper_figures/exp2_sensitivity_map.py
    python test/paper_figures/exp2_sensitivity_map.py --rows kappaH_min eke_c_k

A few minutes per row on a laptop CPU.
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import common
from mini_veros import loop

TARGET_DEPTH = -1000.0  # m, the second snapshot level ("roughly 1000 m")

ROWS = {
    "wind_stress_scale": dict(kind="forcing", which="wind"),
    "kappaH_min": dict(kind="param", param="kappaH_min"),
    "heat_flux_scale": dict(kind="forcing", which="heat"),
    "eke_c_k": dict(kind="param", param="eke_c_k"),
}


def scale_forcing(forcing_fn, which):
    """forcing_fn wrapped so its result is scaled by `alpha` on one of its fields."""

    def scaled(model, state, alpha):
        force = forcing_fn(model, state)
        if which == "wind":
            return eqx.tree_at(lambda f: (f.surface_taux, f.surface_tauy), force,
                                (force.surface_taux * alpha, force.surface_tauy * alpha))
        elif which == "heat":
            return eqx.tree_at(lambda f: f.forc_temp_surface, force, force.forc_temp_surface * alpha)
        raise ValueError(which)

    return scaled


def run_row(model, state0, forcing_fn, row, n_steps, n_snapshots):
    """(base_value, temp_snapshots, sensitivity_snapshots) for one row of ROWS."""
    kind = row["kind"]

    if kind == "param":
        base_value = getattr(model.parameters, row["param"])

        def snapshots(value):
            model_p = common.with_params(model, {row["param"]: value})

            def block(integrator_state, _):
                def one_step(integrator_state, _):
                    force = forcing_fn(model_p, integrator_state.state)
                    return loop.step(model_p, integrator_state, force), None

                integrator_state, _ = lax.scan(one_step, integrator_state, length=n_steps // n_snapshots)
                return integrator_state, common.interior(integrator_state.state.temp)

            _, temps = lax.scan(block, state0, length=n_snapshots)
            return temps

    elif kind == "forcing":
        base_value = jnp.asarray(1.0)
        scaled_forcing_fn = scale_forcing(forcing_fn, row["which"])

        def snapshots(alpha):
            def block(integrator_state, _):
                def one_step(integrator_state, _):
                    force = scaled_forcing_fn(model, integrator_state.state, alpha)
                    return loop.step(model, integrator_state, force), None

                integrator_state, _ = lax.scan(one_step, integrator_state, length=n_steps // n_snapshots)
                return integrator_state, common.interior(integrator_state.state.temp)

            _, temps = lax.scan(block, state0, length=n_snapshots)
            return temps

    else:
        raise ValueError(kind)

    temp, sensitivity = jax.jvp(eqx.filter_jit(snapshots), (base_value,), (jnp.ones_like(base_value),))
    return base_value, temp, sensitivity


def main(row_names, n_steps, n_snapshots):
    model, state0, forcing_fn = common.spinup()
    assert n_steps % n_snapshots == 0, "n_steps must be a multiple of n_snapshots"

    x, y, land = common.grid_arrays(model)
    z = np.asarray(model.grid.zt)
    z_deep_idx = int(np.argmin(np.abs(z - TARGET_DEPTH)))
    steps = np.arange(1, n_snapshots + 1) * (n_steps // n_snapshots)

    base_values, temps, sensitivities = {}, {}, {}
    for name in row_names:
        base_value, temp, sensitivity = run_row(model, state0, forcing_fn, ROWS[name], n_steps, n_snapshots)
        base_values[name] = float(base_value)
        temps[name] = np.asarray(temp)
        sensitivities[name] = np.asarray(sensitivity)
        surface_max = np.abs(sensitivities[name][..., -1]).max(axis=(1, 2))
        print(f"{name}: base={base_values[name]:.4g}  max|d(surface temp)/dp| per snapshot = {surface_max}")

    common.save(
        "exp2_sensitivity_map",
        rows=row_names, base_values=[base_values[r] for r in row_names],
        steps=steps, z=z, z_deep_idx=z_deep_idx, x=x, y=y, land=land,
        dt_tracer=model.config.dt_tracer, n_steps=n_steps,
        **{f"temp__{r}": temps[r] for r in row_names},
        **{f"sensitivity__{r}": sensitivities[r] for r in row_names},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", default=list(ROWS), choices=list(ROWS))
    parser.add_argument("--n-steps", type=int, default=common.HORIZON_DAYS)
    parser.add_argument("--n-snapshots", type=int, default=4)  # divides common.HORIZON_DAYS's default, 80
    args = parser.parse_args()
    if args.n_steps is None:
        raise SystemExit("common.HORIZON_DAYS is not set yet -- run exp1 first, or pass --n-steps explicitly")
    main(args.rows, args.n_steps, args.n_snapshots)
