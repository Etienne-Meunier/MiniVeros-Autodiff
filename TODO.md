# TODO — investigate real divergence, not "expected noise"

Same physics, same numbers as veros. Any mini_veros/veros field diff beyond
run-to-run float noise is a bug, full stop. Below: concrete numbers from
`test/setups_matrix.py`'s 31-variant sweep (`$STORE/MiniVeros-Autodiff/`).
"Documented as expected" in existing comments is not proof — recheck.

Full write-up of the items closed below: `report/divergence_report.md`
(regenerate with `test/investigate_divergence.py` + `test/plot_divergence_report.py`;
gated by `test/test_divergence.py`).

- [x] **global_surface_pressure: mini_veros goes NaN, real veros doesn't.**
      Real bug, fixed. `mini_veros/setup.py:init_barotropic_velocity` ran an
      initial `solve_pressure` whenever `enable_streamfunction` was False.
      Real veros runs no initial barotropic solve in that case at all —
      `VerosSetup.setup` guards `external.streamfunction_init` on
      `enable_streamfunction` — so it starts from psi = 0. On global_4deg
      mini therefore started with psi in [-12.7, +10.0] and u/v up to
      2.5e-2/3.1e-2 where veros had exactly zero: a different initial
      condition, not a solver problem. `init_barotropic_velocity` now
      returns S0 unchanged under the surface-pressure formulation. After the
      fix, global_surface_pressure is bit-identical to veros at step 0 and
      stays at solver-noise level (~1e-7 psi, ~1e-10 u/v) over the first
      steps. Regression test: `test_divergence.py::test_surface_pressure_init_is_not_run`.

- [x] **acc_surface_pressure: "near-singular cells, bounded drift" claim
      (test_pressure_solver.py docstring) is unverified.**
      The claim is wrong about the mechanism but harmless in effect. acc's
      surface-pressure path is not near-singular: at step 0 mini and veros
      are bit-identical, and with the solver's stopping rule tightened they
      agree to 1.4e-11 (normalized) after 10 steps. The u/v "100% relative
      error" is the metric, not the model — see the item below.

- [x] **Establish the actual noise floor first.** (Report on noise floor)
  -> 0.0 no difference between 2 veros run
      Correct but the wrong floor to compare against. veros vs veros is
      bit-identical because it is the same program; that says nothing about
      how far two *different* programs for the same equations may sit. The
      floor that matters is the external mode's stopping rule: both codes
      call `bicgstab(..., tol=0, atol=1e-8)`, an absolute residual bound, so
      the two solvers stop at points ~1e-8 apart in the preconditioned
      residual. That is ~1e-9 relative in psi at step 1 — seven orders above
      float64 eps. Force atol=1e-14 and the step-1 gap drops to ~1e-13.

- [x] **u/v relative error climbs from ~1e-4 to ~1e-2 over the run.**
      Metric artifact. `compare_field`'s `max_rel` divides by
      `max(|mini|,|real|)`, so any cell holding two small values of opposite
      sign scores ~2.0 regardless of how small they are. That is why nearly
      every 30-year row in `matrix_report.md` reports a "worst error" near
      2.0. Use max/rms absolute differences, and compare the run's time-mean
      against veros's own sampling spread (report section 3) — on that
      measure every variant sits below veros compared with itself.

- [x] **acc_maximal / global_maximal fail only when many flags are stacked.**
      Not a stacking effect. With the solver tolerance tightened, every one
      of the 31 variants agrees to 1e-11..1e-14 (acc) or 1e-9..1e-7 (global,
      which inherits a 1.6e-7 step-0 seed from the streamfunction init's own
      elliptic solve) after 10 steps. The only two outliers,
      `acc_tke_superbee_advection` and `acc_maximal`, share
      `enable_tke_superbee_advection`: the superbee limiter is discontinuous
      (`where(vel > 0)`, `clip`, the `abs(rj) < 1e-20` guard), so a
      roundoff-level input difference produces a finite flux difference.
      Real veros run against itself from a 1e-15 relative temperature kick
      reproduces the same ~3e-7 tke jump at the same step.

## Still open

- [x] **`matrix_report.md` mixes snapshots** and its "worst error" column is
      a saturating metric. Both fixed:

      * `test/metrics.py` (new) replaces `max_rel` with metrics that stay
        interpretable after the trajectories separate — scale-normalized max
        (`max|mini-veros| / rms(veros)`), relative L2, pattern correlation,
        an agreement horizon, and the climatology-vs-veros's-own-spread
        ratio. `plot_matrix_report.py` reports these; the per-variant figure
        is now three panels instead of an absolute/relative pair.
      * The report's status is `ok` / `chaotic` / `FAIL` / `diverged` /
        `error`. `chaotic` is the expected long-run outcome and is not a
        failure; `FAIL` means the climatology ratio exceeded 1.
      * `generate_matrix_data.py` always writes an .npz, including for a
        variant that failed outright, and truncates rather than aborts when
        veros's sanity check trips.
      * `--run-timestamp` pins one stamp across a split sweep, and
        `plot_matrix_report.py --strict` refuses to render a mixed one
        (the default now prints a warning banner naming the stale rows).

- [ ] **acc_minimal, global_minimal and global_biharmonic_friction are
      reported as the matrix's only three passing variants, but never
      completed a long run.** Diagnosed: all three are physically unstable
      over 30 years, and *real veros* is the side that rejects them —
      `VerosSetup.step` runs `numerics.sanity_check` every step and raises
      (`veros/veros.py:312`). Reproduced at the full horizon:

        acc_minimal:                solution diverged at iteration 8293
        global_minimal:             solution diverged at iteration 7044
        global_biharmonic_friction: solution diverged at iteration 7505

      mini_veros agrees the configuration blows up (on acc_minimal it first
      goes non-finite between steps 8325 and 8350); it just has no check, so
      it returns a NaN trajectory instead of raising. All three share
      `enable_hor_friction=False`.

      The bug was the reporting, not the physics: `generate_matrix_data.py`'s
      `except Exception` only printed, writing no .npz, and
      `plot_matrix_report.py:resolve_npz` with `timestamp="latest"` then
      fell back to that variant's newest *older* file — the 4-step smoke run
      from 2026-08-28. Those 4-step rows passed the tolerance gate because 4
      steps is not enough time to diverge, which was the entire content of
      the report's "3/31 variants within tolerance". Fixed as described
      above; these three variants should now come back as `diverged` with
      the step recorded.

- [x] **Rerun the matrix after the surface-pressure init fix.** Done:
      snapshot `20260902T072723Z`, all 31 variants on one timestamp,
      rendered with
      `python test/plot_matrix_report.py --strict --timestamp 20260902T072723Z`.

      Result: **28 chaotic, 3 diverged, 0 FAIL**. Every non-diverging
      variant's 30-year mean sits below veros's own sampling spread — worst
      ratio 0.35 (`acc_biharmonic_friction`), and 21 of 28 are at or below
      0.01. The three `diverged` rows are the unstable configurations, all
      sharing `enable_hor_friction=False`; they are reported with the step
      veros's sanity check tripped and how long the two codes tracked before
      it, and their rel L2 / corr / climatology are suppressed because the
      last kept records are mid-blow-up.

      Two defects in the new metrics surfaced during the first sweep and are
      fixed: `util.compare_field`'s `np.nanargmax` raises on an all-NaN
      record (the comparison now stops before mini's first non-finite one),
      and squaring the huge-but-finite values just before a blow-up
      overflows float64, so `rel_l2` / `pattern_corr` are computed on inputs
      rescaled by their peak (both are scale-invariant, so this is exact for
      healthy runs). A third: the climatology ratio was meaningless on a
      field the reference holds constant — acc's `salt` scored 41 on
      roundoff over roundoff — so such fields are now excluded from it.
