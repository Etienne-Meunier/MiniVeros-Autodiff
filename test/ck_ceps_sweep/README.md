# c_k / c_eps ACC density sweep

Does the TKE closure's production/dissipation constants (`c_k`, `c_eps`,
`mini_veros.model.Parameters` defaults 0.10/0.70) leave a visible mark on the
ACC channel's potential density profile? 25 runs of `acc/full`
(idealized channel, TKE + EKE closures on, `dt_tracer` = 43200s so one step is
12h and 730 steps is a model year), a 5x5 grid spanning a factor of 16 around
each default (`c_k`: 0.025-0.4, `c_eps`: 0.175-2.8, log2-spaced), each
integrated 30 model-years forward from a cold start (`common.N_YEARS`) -- long
enough to settle: `acc_full` in the matrix comparison (`report/matrix_report.md`)
already reaches climatology ratio ~0.00 for temperature/streamfunction by 15
model-years, so 30 has margin.

Rather than storing full 3D snapshots, each run logs a horizontally-averaged
(area-weighted, ocean cells only) potential density profile (referenced to the
surface, `eq_of_state_type=3`) every 30 days -- 365 rows over 30 years, ~6KB per
run. The reported "final" profile per `(c_k, c_eps)` is the mean of the last 12
of those rows (~the last model-year).

- `common.py` -- grid definition, `build_sweep_reductor()` (the shared
  log_select_fn/jitted run_fn, built once so the 25 runs share one JIT trace),
  `run_one(c_k, c_eps, ...)`.
- `run_sweep.py` -- runs all 25 points, writes
  `$STORE/MiniVeros-Autodiff/results/ck_ceps_sweep/ck_ceps_density_sweep.npz`.
- `fig_density_profiles.py` -- reads that back, writes
  `report/ck_ceps_density_figures/*.{pdf,png}`: profile lines vs `c_k` (`c_eps`
  held at default) and vs `c_eps` (`c_k` held at default), plus a
  surface-minus-bottom density heatmap over the full grid. Never re-runs a
  simulation.

```
python test/ck_ceps_sweep/run_sweep.py
python test/ck_ceps_sweep/fig_density_profiles.py
```

## Running on the server

Same GPU host as the paper figures (`test/paper_figures/README.md`), but this
is a single sequential script (25 forward-only rollouts, no autodiff, no
checkpointing needed) -- one OAR job, no wandb sweep/multi-agent required.

```
g5k sync code
oarsub -l gpu=1,walltime=6:00:00 -p "gpu_count > 0" -n mv-ck-ceps \
  -O ~/oar_logs/ck_ceps.out -E ~/oar_logs/ck_ceps.err \
  "bash ~/code/MiniVeros-Autodiff/test/sweep/run_ck_ceps_oar.sh"
# watch: g5k check --site grenoble ; g5k log <job_id> --site grenoble -f
g5k sync model   # or a targeted scp of results/ck_ceps_sweep/*.npz
python test/ck_ceps_sweep/fig_density_profiles.py
```

## Sizing

Forward-only (no gradient/checkpointing), acc-size grid (90x40x15 in the
matrix runs' terms, here the smaller channel grid): matrix_report's
`acc_full` did 10950 steps at ~9ms/step combined mini+veros on GPU, mini alone
faster still. 30 years = 21900 steps/run x 25 runs is expected to finish in
well under the 6h walltime requested above; if a run diverges (unstable
`c_k`/`c_eps` combination -- plausible at the low-`c_eps`/high-`c_k` corner,
where TKE production outruns dissipation) it's marked `diverged=True` in the
.npz and skipped in the figures, not treated as a crash.
