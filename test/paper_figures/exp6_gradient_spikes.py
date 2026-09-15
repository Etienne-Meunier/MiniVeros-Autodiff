"""
Experiment 6 -- where do the gradient spikes come from?

Figure 1 shows that even inside the trustworthy window, autodiff occasionally
disagrees sharply with finite differences for a single rollout length. This
asks three questions about that: how often, where, and why.

  (a) How often. Continuing the spin-up gives M_MEMBERS "start dates" 7 days
      apart. For each one, at two lengths (H/2 and H), we compute the autodiff
      and finite-difference gradients of the same loss w.r.t. c_k and record
      their ratio r. A spike is |r - 1| large or the wrong sign.

  (b) Where and when. Taking one spiking start date and one normal one, we
      replay the rollout by hand: the forward states are all kept, then walked
      backward one `jax.vjp` of a single model step at a time (not
      `common.rollout`'s checkpointed reverse-mode, so every intermediate
      adjoint is visible, not just the ones at checkpoint boundaries). At each
      backward step we record the norm of the adjoint of every prognostic
      field, and the grid cell where |adjoint of temp| is largest.

  (c) Why. With a spiking and a normal start date in hand, we re-measure (a)'s
      spike fraction at length H after switching off one likely source of
      non-smoothness at a time -- the min/max in a flux limiter, or a closure
      switching between two regimes -- via `build(overrides=...)`. A config
      whose spike fraction drops to ~0 names the culprit. Each override needs
      its own spin-up (cached separately by common.spinup, same climatology,
      slightly different equilibrium), so this part uses a smaller ensemble
      than (a).

    python test/paper_figures/exp6_gradient_spikes.py

"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import common
from mini_veros import loop

PARAM = "c_k"
REL_STEP = 1e-5
M_MEMBERS = 50
START_SPACING_DAYS = 7
SPIKE_THRESHOLD = 1.0  # |r - 1| beyond this (or a sign flip) counts as a spike
N_MEMBERS_WHY = 10  # smaller ensemble for part (c), one spin-up per override

# Non-smooth suspects, applied one at a time via build(overrides=...).
ABLATIONS = {
    "default": {},
    "no_tke_superbee": {"enable_tke_superbee_advection": False},
    "no_eke_superbee": {"enable_eke_superbee_advection": False},
    "no_eke": {"enable_eke": False},
}


def start_dates(model, state0, forcing_fn, n_members, spacing_days):
    """n_members states, `spacing_days` apart, continuing on from the spin-up."""

    @eqx.filter_jit
    def block(integrator_state, _):
        integrator_state = common.rollout(model, integrator_state, forcing_fn, spacing_days, spacing_days)
        return integrator_state, integrator_state

    _, states = lax.scan(block, state0, length=n_members)
    return states  # a single IntegratorState pytree with a leading n_members axis


def ratios(model, forcing_fn, members, n_steps):
    """(r,) autodiff/finite-difference ratio of dL/d(PARAM) for every member, at this length."""
    base = getattr(model.parameters, PARAM)

    @eqx.filter_jit
    def loss(value, member_state0):
        model_p = common.with_params(model, {PARAM: value})
        return common.mean_square_sst(common.rollout(model_p, member_state0, forcing_fn, n_steps).state)

    grad = eqx.filter_jit(eqx.filter_grad(loss))
    h = REL_STEP * jnp.abs(base)

    n_members = jax.tree_util.tree_leaves(members)[0].shape[0]
    out = np.zeros(n_members)
    for i in range(n_members):
        member_i = jax.tree_util.tree_map(lambda x: x[i], members)
        ad = grad(base, member_i)
        fd = (loss(base + h, member_i) - loss(base - h, member_i)) / (2 * h)
        out[i] = ad / fd
    return out


def backward_sweep(model, forcing_fn, state0, n_steps):
    """Per-backward-step adjoint norm of every prognostic field, and where |adjoint of temp| peaks.

    Returns (field_names, norms (n_steps, n_fields), peak_xy (n_steps, 2)).
    """

    def step_fn(integrator_state):
        force = forcing_fn(model, integrator_state.state)
        return loop.step(model, integrator_state, force)

    step_fn = eqx.filter_jit(step_fn)

    states = [state0]
    for _ in range(n_steps):
        states.append(step_fn(states[-1]))

    field_names = [f for f in ("u", "v", "temp", "salt", "psi", "tke", "eke") if getattr(states[-1].state, f) is not None]

    loss_grad = eqx.filter_grad(common.mean_square_sst)(states[-1].state)
    cotangent = jax.tree_util.tree_map(jnp.zeros_like, states[-1])
    cotangent = eqx.tree_at(lambda s: s.state, cotangent, loss_grad)

    norms = np.zeros((n_steps, len(field_names)))
    peak_xyz = np.zeros((n_steps, 3), dtype=int)  # (x, y, z) grid index of max |adjoint of temp|

    for i in range(n_steps - 1, -1, -1):
        _, vjp_fn = jax.vjp(step_fn, states[i])
        (cotangent,) = vjp_fn(cotangent)
        for j, f in enumerate(field_names):
            norms[i, j] = float(jnp.linalg.norm(getattr(cotangent.state, f)))
        temp_adj = common.interior(cotangent.state.temp)
        flat_idx = int(jnp.argmax(jnp.abs(temp_adj)))
        peak_xyz[i] = np.unravel_index(flat_idx, temp_adj.shape)

    return field_names, norms, peak_xyz


def main(n_members, spacing_days):
    model, state0, forcing_fn = common.spinup()
    if common.HORIZON_DAYS is None:
        raise SystemExit("common.HORIZON_DAYS is not set yet -- run exp1 first")
    H = common.HORIZON_DAYS
    lengths = [max(H // 2, 1), H]

    print(f"(a) how often -- {n_members} members, {spacing_days} days apart, lengths {lengths}")
    members = start_dates(model, state0, forcing_fn, n_members, spacing_days)
    ratios_by_length = {}
    for n in lengths:
        r = ratios(model, forcing_fn, members, n)
        spike = np.abs(r - 1.0) > SPIKE_THRESHOLD
        wrong_sign = r < 0
        ratios_by_length[n] = r
        print(f"  n={n:5d}d  spike fraction={spike.mean():.2f}  wrong-sign fraction={wrong_sign.mean():.2f}")

    print("(b) where and when")
    r_at_H = ratios_by_length[H]
    spiking_idx = int(np.argmax(np.abs(r_at_H - 1.0)))
    normal_idx = int(np.argmin(np.abs(r_at_H - 1.0)))
    print(f"  spiking member {spiking_idx} (r={r_at_H[spiking_idx]:.3e}), "
          f"normal member {normal_idx} (r={r_at_H[normal_idx]:.3e})")

    def member_state(i):
        return jax.tree_util.tree_map(lambda x: x[i], members)

    field_names, norms_spike, peak_spike = backward_sweep(model, forcing_fn, member_state(spiking_idx), H)
    _, norms_normal, peak_normal = backward_sweep(model, forcing_fn, member_state(normal_idx), H)

    print("(c) why -- ablations")
    ablation_names = list(ABLATIONS)
    ablation_spike_fraction = np.zeros(len(ablation_names))
    for k, name in enumerate(ablation_names):
        overrides = ABLATIONS[name]
        m, s, f = common.spinup(overrides=overrides or None)
        m_ablation = start_dates(m, s, f, N_MEMBERS_WHY, spacing_days)
        r = ratios(m, f, m_ablation, H)
        ablation_spike_fraction[k] = np.mean(np.abs(r - 1.0) > SPIKE_THRESHOLD)
        print(f"  {name:18s}  spike fraction={ablation_spike_fraction[k]:.2f}")

    x, y, land = common.grid_arrays(model)
    common.save(
        "exp6_gradient_spikes",
        lengths=lengths, param=PARAM, n_members=n_members, spacing_days=spacing_days,
        **{f"ratios__{n}": ratios_by_length[n] for n in lengths},
        spiking_idx=spiking_idx, normal_idx=normal_idx,
        field_names=field_names, norms_spike=norms_spike, norms_normal=norms_normal,
        peak_spike=peak_spike, peak_normal=peak_normal,
        ablation_names=ablation_names, ablation_spike_fraction=ablation_spike_fraction,
        x=x, y=y, land=land,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-members", type=int, default=M_MEMBERS)
    parser.add_argument("--spacing-days", type=int, default=START_SPACING_DAYS)
    args = parser.parse_args()
    main(args.n_members, args.spacing_days)
