# TODO — investigate real divergence, not "expected noise"

Same physics, same numbers as veros. Any mini_veros/veros field diff beyond
run-to-run float noise is a bug, full stop. Below: concrete numbers from
`test/setups_matrix.py`'s 31-variant sweep (`$STORE/MiniVeros-Autodiff/`).
"Documented as expected" in existing comments is not proof — recheck.

- [ ] **global_surface_pressure: mini_veros goes NaN, real veros doesn't.**
      Checked raw fields directly: step 0 both ~29.7, step 5 mini all-NaN,
      real still ~30.1, stays fine through step 60. Not noise, not
      approximation — mini's `solve_pressure.py` diverges on the global
      grid. Root-cause in the bicgstab solve (matrix conditioning on
      real bathymetry vs acc's channel).

- [ ] **acc_surface_pressure: "near-singular cells, bounded drift" claim
      (test_pressure_solver.py docstring) is unverified, not proven.**
      u/v relative error reaches ~100% by step 300. Confirm whether mini's
      bicgstab actually handles those cells identically to real veros's
      solver (same pivoting/regularization), or whether the docstring is
      rationalizing a real gap.

- [ ] **Establish the actual noise floor first.** No baseline exists for
      "how much do two veros runs of the *same* setup differ" (solver
      tolerance, threading, whatever). Run real veros twice, diff against
      itself, use *that* as the pass bound — not an assumed rtol=1e-6.

- [ ] **u/v relative error climbs from ~1e-4 to ~1e-2 over the run in
      several nominally-passing variants** (acc_basic, global_default,
      acc_no_hor_friction). Currently written off as "near-zero denominator
      artifact" — verify: does it plateau (rounding) or keep growing
      (real divergence)? Rerun a couple at 3-5x the horizon and check slope.

- [ ] **acc_maximal / global_maximal fail only when many flags are stacked**
      (small margin over tolerance). Bisect: turn overrides on one at a time
      to find which single flag actually introduces the mismatch, instead of
      shrugging at "more terms, more noise."
