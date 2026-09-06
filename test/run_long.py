#!/usr/bin/env python3
"""
Long single-code integration, for watching a setup's own evolution rather than
comparing two codes. `--code` selects mini_veros or veros.

Why not generate_matrix_data.py: that runs both codes (veros is ~2x mini's
cost here, so it would more than double the wall time for nothing) and its
log_select_fn records every prognostic field in full 3D. At 360x160x60 that
is ~28 MB per field per record held on device -- a decade at monthly records
would be tens of GB of device memory. This logs surface slices only.

The run is split into segments, each one a separate `loop.run` call, with the
cumulative .npz rewritten after every segment. That buys three things a
single long scan does not:

  - partial results survive a walltime kill, with no checkpoint machinery
  - progress is visible in the job log rather than silent for hours
  - `loop.run`'s NaN guard raises per segment, so a blow-up is caught with
    every earlier record already on disk

Both codes stop themselves on a blow-up and the runner records where: mini via
loop.run's NaN guard, veros via numerics.sanity_check raising "solution
diverged at iteration N". That makes this the tool for asking whether a
configuration is stable in *both* codes or only one.

Usage:
    python test/run_long.py --code mini  --years 10 --device gpu
    python test/run_long.py --code veros --years 5  --device gpu
    python test/run_long.py --code mini --years 1 --nx 90 --ny 40 --nz 15
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
RESULTS_DIR = STORE / "MiniVeros-Autodiff" / "results"

SECONDS_PER_DAY = 86400.0
DAYS_PER_YEAR = 365.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="global_1deg")
    parser.add_argument("--code", default="mini", choices=("mini", "veros"))
    parser.add_argument("--years", type=float, default=10.0)
    parser.add_argument("--log-days", type=float, default=30.4, help="model days between records")
    parser.add_argument("--segment-years", type=float, default=1.0, help="model years per saved segment")
    parser.add_argument("--device", default="gpu", choices=("cpu", "gpu"))
    parser.add_argument("--solver-atol", type=float, default=None, help="default: leave both codes as shipped")
    parser.add_argument("--veros-path", type=Path, default=REPO_ROOT / "veros")
    parser.add_argument("--run-timestamp", default=None)
    for dim in ("nx", "ny", "nz"):
        parser.add_argument(f"--{dim}", type=int, default=None, help=f"override {dim} (for a cheap smoke test)")
    args = parser.parse_args()

    from util import configure_veros_runtime

    configure_veros_runtime(args.veros_path, device=args.device)
    sys.path.insert(0, str(REPO_ROOT / "mini-veros"))

    import equinox as eqx
    import importlib
    import jax

    from setups_matrix import FAMILIES, VARIANTS_BY_NAME
    from variant_util import DEFAULT_SOLVER_ATOL, forced_solver_atol

    variant = VARIANTS_BY_NAME[args.variant]
    setup_mod = importlib.import_module(FAMILIES[variant["family"]]["mini_module"])

    overrides = dict(variant["overrides"])
    for dim in ("nx", "ny", "nz"):
        if getattr(args, dim) is not None:
            overrides[dim] = getattr(args, dim)

    from mini_veros import loop

    atol = args.solver_atol if args.solver_atol is not None else DEFAULT_SOLVER_ATOL


    def build_veros():
        """NoIO veros sim for this variant, plus its surface-slice logger."""
        import importlib as _il

        from veros.routines import veros_routine

        spec = FAMILIES[variant["family"]]
        RealSetup = getattr(_il.import_module(spec["real_module"]), spec["real_class"])

        class NoIOSetup(RealSetup):
            @veros_routine
            def set_diagnostics(self, state):
                state.diagnostics.clear()

        sim = NoIOSetup(override=dict(runlen=0, **overrides))
        sim.setup()

        def log():
            vs = sim.state.variables
            tau = vs.tau
            return {
                "temp": np.asarray(vs.temp[:, :, -1, tau]),
                "salt": np.asarray(vs.salt[:, :, -1, tau]),
                "psi": np.asarray(vs.psi[..., tau]),
            }

        return sim, log
    with forced_solver_atol(atol):
        if args.code == "mini":
            model, state, forcing_fn = setup_mod.build(overrides)
            dt = float(model.config.dt_tracer)
            nx, ny, nz = model.config.nx, model.config.ny, model.config.nz
        else:
            sim, veros_log = build_veros()
            dt = float(sim.state.settings.dt_tracer)
            nx, ny, nz = (sim.state.settings.nx, sim.state.settings.ny, sim.state.settings.nz)
        log_every = max(1, int(round(args.log_days * SECONDS_PER_DAY / dt)))
        seg_steps = max(log_every, int(round(args.segment_years * DAYS_PER_YEAR * SECONDS_PER_DAY / dt)))
        seg_steps -= seg_steps % log_every  # loop.run requires n_steps % log_every == 0
        n_segments = max(1, int(round(args.years * DAYS_PER_YEAR * SECONDS_PER_DAY / dt / seg_steps)))
        total = n_segments * seg_steps

        print(f"--- {args.variant} {args.code}-only: nx={nx} ny={ny} nz={nz}, "
              f"dt={dt:g}s, device={args.device}, solver atol={atol:g}", flush=True)
        print(f"    {total} steps = {total * dt / SECONDS_PER_DAY / DAYS_PER_YEAR:.2f} model years, "
              f"{n_segments} segments of {seg_steps}, record every {log_every} steps "
              f"({log_every * dt / SECONDS_PER_DAY:.1f} days)", flush=True)

        def log_select_fn(integrator_state):
            # surface slices only -- the full 3D state would be ~28 MB per
            # field per record on device at this resolution
            s = integrator_state.state
            return {"temp": s.temp[:, :, -1], "salt": s.salt[:, :, -1], "psi": s.psi}

        run_fn = eqx.filter_jit(loop.run)
        run_timestamp = args.run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"longrun_{args.code}_{args.variant}__{run_timestamp}.npz"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        steps_done, timesteps, logs, stopped_at, partial = 0, [], [], None, []
        t_start = time.time()

        def save(segments_done):
            """Rewrite the cumulative .npz. Called after every segment, and
            again when one fails, so a blow-up keeps everything before it."""
            out = dict(
                timesteps=np.asarray(timesteps),
                variant=np.asarray(args.variant),
                code=np.asarray(args.code),
                device=np.asarray(args.device),
                solver_atol=np.asarray(atol),
                dt=np.asarray(dt),
                generated_at=np.asarray(run_timestamp),
                overrides_json=np.asarray(json.dumps(overrides)),
                sec_per_step=np.asarray((time.time() - t_start) / max(steps_done, 1)),
                stopped_at=np.asarray(-1 if stopped_at is None else stopped_at),
                segments_done=np.asarray(segments_done),
                n_records=np.asarray(len(timesteps)),
            )
            for f in (logs[0] if logs else {}):
                out[f"{f}_frames"] = np.stack([r[f] for r in logs])
            np.savez(out_path, **out)

        for seg in range(n_segments):
            t0 = time.time()
            try:
                if args.code == "mini":
                    state, seg_logs = run_fn(model, state, forcing_fn, log_select_fn, seg_steps, log_every)
                    seg_logs = jax.tree_util.tree_map(np.asarray, seg_logs)
                else:
                    # veros steps from Python, so its records can be collected
                    # as we go -- a segment that diverges keeps everything up
                    # to the failing step, unlike mini's compiled scan
                    collected = []
                    try:
                        for k in range(seg_steps):
                            sim.step(sim.state)
                            if (k + 1) % log_every == 0:
                                collected.append(veros_log())
                    except RuntimeError:
                        # keep the records this segment did produce before
                        # sanity_check tripped, then re-raise to the handler
                        partial = collected
                        raise
                    seg_logs = {f: np.stack([c[f] for c in collected]) for f in collected[0]}
            except (eqx.EquinoxRuntimeError, RuntimeError) as exc:
                # loop.run's guard tripped: state is frozen and the logs for
                # this segment are lost, but every earlier segment is on disk
                # veros steps from Python, so whatever the failing segment
                # logged before it tripped is still usable; mini's compiled
                # scan gives us nothing back
                for rec in locals().get("partial", []):
                    steps_done += log_every
                    timesteps.append(steps_done)
                    logs.append(rec)
                stopped_at = steps_done
                print(f"    segment {seg + 1}: model stopped after {steps_done} steps "
                      f"({str(exc).splitlines()[0][:80]})", flush=True)
                save(seg)
                break

            for k in range(seg_steps // log_every):
                steps_done_k = steps_done + (k + 1) * log_every
                timesteps.append(steps_done_k)
                logs.append({f: seg_logs[f][k] for f in seg_logs})
            steps_done += seg_steps

            finite = all(np.all(np.isfinite(v)) for v in logs[-1].values())
            elapsed = time.time() - t0
            print(f"    segment {seg + 1}/{n_segments}: {steps_done} steps "
                  f"({steps_done * dt / SECONDS_PER_DAY / DAYS_PER_YEAR:.2f} yr), "
                  f"{elapsed:.0f}s ({elapsed / seg_steps * 1000:.1f} ms/step), finite={finite}", flush=True)

            save(seg + 1)

            if not finite:
                print("    non-finite record; stopping", flush=True)
                stopped_at = steps_done
                break

    print(f"    saved {out_path}  ({len(timesteps)} records, "
          f"{'completed' if stopped_at is None else f'stopped at step {stopped_at}'})", flush=True)


if __name__ == "__main__":
    main()
