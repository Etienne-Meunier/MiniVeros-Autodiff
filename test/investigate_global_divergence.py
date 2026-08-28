#!/usr/bin/env python3
"""
Investigation script: why does *every* global_4deg variant diverge from real
veros (report/matrix_report.md), while most acc variants match to ~1e-9?

Findings (see conclusion printed at the end, and TODO.md):
  - Static setup (kbot/maskT/land_map/ht/nisle) matches EXACTLY between
    mini_veros and real veros -- topology/masking is not the source.
  - The island streamfunction basis (psin) matches to ~5e-8 absolute,
    i.e. exactly the bicgstab solve's own atol=1e-8 -- solver-tolerance
    noise, not a setup bug.
  - GSW/TEOS-10 (eq_of_state_type=5, used only by global_4deg -- acc uses
    type 3, and test_eos_types.py never exercises 5) matches bit-for-bit
    (see the direct EOS comparison this script also runs).
  - Per-step, per-field error growth (u/v/temp/salt/tke/eke) is tiny in
    absolute terms (~1e-9 to ~3e-8 by step 60), does NOT grow unboundedly
    (it plateaus), and its worst-error location DRIFTS step to step
    (tracks wherever the local field gradient is steepest) rather than
    sitting at one fixed grid cell -- the signature of floating-point
    reduction-order noise from the elliptic (bicgstab) solve, amplified
    by grid size/condition number, not a fixed indexing/masking bug.
  - global_4deg's grid (90x40, 6 islands, real bathymetry) is ~3x bigger
    and much more heavily conditioned than acc's (30x42, 2 islands,
    idealized channel) -- same solver code, same atol=1e-8, but a bigger/
    worse-conditioned system lets mini_veros's and real veros's
    independently-rounding bicgstab iterations settle to solutions that
    differ by more within that same residual tolerance. That's enough to
    cross the test suite's very tight atol=1e-8/rtol=1e-8 pass bound for
    `temp` specifically (the field this script shows actually flips the
    final-step pass/fail for global_default) -- it is not enough to
    reflect a real physics/port divergence.
  - `psi` "fails" raw allclose from step 0 onward -- expected and already
    handled by plot_matrix_report.py's scale-normalized check, not a bug.

This does NOT explain global_surface_pressure (NaN blowup, separate
pressure-solver code path) -- that one is a real, already-flagged bug.

Run: python test/investigate_global_divergence.py [--veros-path PATH] [--steps N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from util import configure_veros_runtime, compare_field, compute_error_evolution
from variant_util import build_mini_variant, build_real_variant


def compare(name, a, b, rtol=1e-8, atol=1e-10):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"  {name:28s} SHAPE MISMATCH mini={a.shape} real={b.shape}")
        return
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom == 0, 1.0, denom)
    rel = diff / denom
    ok = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    argmax = np.unravel_index(np.nanargmax(diff), diff.shape) if diff.size else None
    flag = "ok" if ok else "DIVERGES"
    print(
        f"  {name:28s} {flag:9s} max_abs={float(np.nanmax(diff)):.3e} "
        f"max_rel(@argmax)={float(rel[argmax]):.3e} at {argmax} "
        f"(mini={a[argmax]:.6g} real={b[argmax]:.6g})"
    )


def compare_eos_type5(veros_path):
    """Direct comparison of the GSW/TEOS-10 (eq_of_state_type=5) pure functions -- used only by
    global_4deg (acc uses type 3), and never covered by test_eos_types.py (which only checks 1/2/4)."""
    sys.path.insert(0, str(REPO_ROOT / "mini-veros"))
    sys.path.insert(0, str(veros_path))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from mini_veros.core.density import gsw as mini_gsw
    from veros.core.density import gsw as real_gsw

    rng = np.random.default_rng(0)
    n = 5000
    salt = jnp.asarray(rng.uniform(30, 40, n))
    temp = jnp.asarray(rng.uniform(-2, 30, n))
    press = jnp.asarray(rng.uniform(0, 500, n))

    print("\n--- GSW/TEOS-10 (eq_of_state_type=5) pure-function check (untested by test_eos_types.py) ---")
    for fn in ["gsw_rho", "gsw_drhodT", "gsw_drhodS", "gsw_dyn_enthalpy"]:
        mv = getattr(mini_gsw, fn)(salt, temp, press)
        rv = getattr(real_gsw, fn)(salt, temp, press)
        diff = float(jnp.max(jnp.abs(mv - rv)))
        print(f"  {fn:20s} max_abs_diff={diff:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veros-path", type=Path, default=REPO_ROOT / "veros")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--record-interval", type=int, default=5)
    args = ap.parse_args()

    configure_veros_runtime(args.veros_path)

    compare_eos_type5(args.veros_path)

    family = "global_default"
    print(f"\n=== building mini_veros/{family} ===")
    model, mini_state0, mini_final, _, mini_ts, mini_states = build_mini_variant(
        "global_default", family, {}, args.steps, args.veros_path, record_interval=args.record_interval
    )
    print(f"=== building real veros/{family} ===")
    sim, real_state0, real_final, _, real_ts, real_states = build_real_variant(
        "global_default", family, {}, args.steps, args.veros_path, record_interval=args.record_interval
    )

    # --- 1. static setup comparison -----------------------------------
    print("\n--- topology / island setup (built once, before any timestep) ---")
    bc = model.boundary_conditions
    vs = sim.state.variables
    print(f"  nisle: mini={bc.psin.shape[-1]} real={vs.psin.shape[-1]}")
    compare("kbot", bc.kbot, vs.kbot)
    compare("maskT", bc.maskT.astype(float), vs.maskT.astype(float))
    compare("land_map", bc.land_map, vs.land_map)
    compare("psin (island basis fn)", np.asarray(bc.psin), np.asarray(vs.psin))
    compare("ht", bc.ht, vs.ht)

    # --- 2. per-field error evolution, matching the matrix report's own
    #        criterion (compare_field's np.allclose(atol=1e-8, rtol=1e-8)) --
    print(f"\n--- per-field error evolution (atol=1e-8, rtol=1e-8, matching generate_matrix_data.py) ---")
    errors = compute_error_evolution(mini_ts, mini_states, real_states, atol=1e-8, rtol=1e-8)
    for field in sorted(errors):
        d = errors[field]
        line = " ".join("P" if p else "F" for p in d["passes"])
        print(f"  {field:6s} pass/fail per step [{line}]  final max_abs={d['max_abs_errors'][-1]:.3e}"
              f"  final max_rel={d['max_rel_errors'][-1]:.3e}")

    # --- 3. does the worst-error location drift (floating-point noise
    #        tracking local gradients) or sit fixed (structural bug)? ---
    print("\n--- temp: does the worst-error grid cell drift over time? ---")
    for t, ms, rs in zip(mini_ts, mini_states, real_states):
        r = compare_field("temp", ms["temp"], rs["temp"], atol=1e-8, rtol=1e-8)
        print(f"  t={t:4d} argmax={r['argmax']} max_abs={r['max_abs']:.3e}")

    print(
        "\nSee this file's module docstring for the conclusion this evidence supports: "
        "global_default's failure is bicgstab solver-tolerance noise (amplified by the "
        "bigger/6-island/real-bathymetry grid), not a physics/port bug -- except "
        "global_surface_pressure, which is a separate, real, already-flagged issue."
    )


if __name__ == "__main__":
    main()
