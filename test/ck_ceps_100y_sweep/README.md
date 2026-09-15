# c_k / c_eps 100-year full-state sweep

Wider and longer than `test/ck_ceps_sweep/`'s 5x5 grid (factor-16 range around
the defaults 0.10/0.70, log2-spaced): a **10x10 grid, factor-64 range around
each default, log2-spaced, 100 points total** --
c_k in [0.0125, 0.8], c_eps in [0.0875, 5.6] (`common.C_K_GRID`/`C_EPS_GRID`,
also the source of `test/sweep/ck_ceps_100y.yaml`'s values) -- plus two more
changes:

- **100 model-years instead of 30** (`common.N_STEPS` = 73020 steps =
  ~100.03 model-years -- 730 steps/model-year doesn't divide evenly into a
  30-day/60-step cadence at 100 years the way it does at 30, so the step
  count is rounded to the nearest whole number of monthly snapshots instead
  of an exact 100y).
- **Full `PrognosticState` (u, v, temp, salt, tke, eke, psi, time) saved every
  ~30 days, for all 100 points** -- not just the 5 largest-deviation points at
  30 days like `run_top5_full_state.py`.
- **Resumable**: each point also checkpoints its full `IntegratorState` at the
  final step, so a run can be continued later without starting back at year 0
  (see Resuming below).

Run in parallel via a wandb grid sweep, one (c_k, c_eps) point per agent pull,
one GPU per point -- not the 30-year sweep's single sequential OAR job.

## Layout

- `common.py` -- grid, `N_STEPS`/`N_LOGS` sizing, `run_one_point(c_k, c_eps)`,
  `save_point(...)`, and the checkpoint/resume pieces below.
- `sweep_ck_ceps_100y.py` -- the wandb-agent entrypoint: one process pulls one
  `(c_k, c_eps)` off the grid queue, runs it, and writes
  `$STORE/MiniVeros-Autodiff/results/ck_ceps_100y_sweep/full_state/{sweep_id}/ck{c_k}_eps{c_eps}.npz`
  plus a checkpoint (see below). A divergent point is logged as
  `status=failed` rather than crashing the agent, so the queue keeps draining.
- `resume_point.py` -- standalone CLI to continue one point from its
  checkpoint (see Resuming below).
- `../sweep/ck_ceps_100y.yaml` -- the sweep config (`method: grid` over two
  10-value `values` lists, `c_k` x `c_eps` -- wandb's grid method takes the
  full cartesian product, so this is 100 points).

Nothing in mini-veros is modified: only `Parameters` entries (`c_k`, `c_eps`)
are overridden via `full.build`'s overrides dict, same as `test/ck_ceps_sweep/`.

## Launch

```
g5k sync code
wandb sweep test/sweep/ck_ceps_100y.yaml       # prints SWEEP_ID
g5k agent <entity>/<project>/<sweep_id> --site grenoble --n-agents 25 \
    --walltime 6:00:00 --no-besteffort
```

`--n-agents 25` runs the whole grid at once (one GPU per point, no
memory-sharing between runs); fewer agents just drain the same 25-point queue
in waves. Watch with `g5k sweep inspect <sweep_path>` / `g5k check --site grenoble`.

```
g5k sync model   # or a targeted scp of results/ck_ceps_100y_sweep/full_state/<sweep_id>/*.npz
```

## Resuming

`save_point`'s monthly snapshots only keep the physical fields
(`PrognosticState`) -- enough to plot, not enough to continue the run: the
AB2 tracer blend needs the last two tendency lags, the pressure solver needs
its warm-start `dpsi`, and the TKE budget reads the previous step's
dissipation, none of which are in a `PrognosticState`. `full.build`'s cold
start always re-zeros/re-derives those from t=0 initial conditions
(`setup.init_integrator_state`), so rebuilding a run from a monthly snapshot
plus a fresh `full.build` call would restart with fields 100 years in but
tendency history and carried diagnostics from year 0 -- a real inconsistency,
not just the ordinary cold-start transient.

`save_checkpoint`/`load_checkpoint` avoid that: every point also writes its
full final `IntegratorState` (`state` + `tendency_m1`/`tendency_m2` +
`statefuldiag_m1`) to
`$STORE/MiniVeros-Autodiff/results/ck_ceps_100y_sweep/checkpoints/{sweep_id}/ck{c_k}_eps{c_eps}.npz`.
`resume_point.py` loads one of these, continues it `--extra-years` further
(model/grid/topology rebuilt fresh via `full.build` -- deterministic given
the same `(c_k, c_eps)`, only the dynamic state needs to come from the
checkpoint), and writes both a continuation full-state `.npz` and a fresh
checkpoint so it can be chained again:

```
python test/ck_ceps_100y_sweep/resume_point.py \
    --run-id <sweep_id> --c-k 0.4 --c-eps 0.175 --extra-years 100
```

Writes to run id `<sweep_id>_resumed` by default (`--out-run-id` to pick
another). Only wired up per-point so far -- resuming the whole grid in
parallel would mean a second wandb sweep whose config carries `run_id` and
`extra_years` alongside `c_k`/`c_eps`; add that if/when resuming more than a
couple of points at once is actually needed.

## Sizing (estimated, not yet measured at this length)

- **Runtime**: the 30-year sweep's `acc_full`-shape grid ran forward-only at
  ~9ms/step combined mini+veros on GPU (`report/matrix_report.md`), mini
  alone faster. 73020 steps/point at that rate is on the order of ~10 min
  alone on a GPU; run in parallel across 25 GPUs the whole sweep should land
  well inside the 6h walltime above with margin for a slower per-step cost at
  this closure's actual settings.
- **Storage**: `run_top5_full_state.py` measured ~1.1 MiB/snapshot for this
  same field set. At ~1217 monthly snapshots x 25 points, that's
  ~1217 x 25 x 1.1 MiB =~ 33 GB total (float64) written to `$STORE` -- confirm
  there's room before launching all 25 at once. Checkpoints add one snapshot's
  worth per point (~1.1 MiB x 25 =~ 28 MB) plus the extra tendency/diagnostic
  fields (`tendency_m1`/`m2`, `statefuldiag_m1`) roughly doubling that single
  snapshot's size -- negligible next to the 33 GB above.
