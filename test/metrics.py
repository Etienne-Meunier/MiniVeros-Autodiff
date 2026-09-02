#!/usr/bin/env python3
"""
Comparison metrics for mini_veros vs veros.

Why not just `util.compare_field`: its `max_rel` divides by
`max(|mini|, |real|)` per cell, so a single grid point holding two tiny
values of opposite sign scores ~2.0 no matter how small they are. On a
30-year run that happens somewhere on the grid every time, which is why
`matrix_report.md`'s "worst error" column reads ~2.0 for nearly every
variant and carries no information. The metrics here are built so that a
number stays interpretable once the two trajectories have decorrelated:

  max_norm      max |mini - real| / rms(real)
                the old max_abs, made scale-free. Still a tail statistic,
                so still spiky, but comparable across fields and setups.

  rel_l2        ||mini - real||_2 / ||real||_2
                whole-field relative error. Does not blow up on near-zero
                cells and is the number to quote for "how far apart are the
                two states".

  pattern_corr  correlation of the two fields after removing their spatial
                means. 1.0 means same pattern at different amplitude, 0
                means unrelated. Separates "same solution, small offset"
                from "different weather".

Run-level (not per step):

  agreement_horizon  first recorded step where max_norm exceeds a threshold.
                     One number for "how long do the two codes track".

  climatology        time-mean over the run's 2nd half, mini vs real, next
                     to the same statistic measured on real alone (its
                     3rd-quarter mean vs its 4th-quarter mean). The ratio
                     says whether the models differ by more than the
                     reference model differs from itself over a window of
                     the same length -- the only long-horizon statement that
                     survives chaotic separation.
"""

import numpy as np

# Threshold for `agreement_horizon`: comfortably above the elliptic solver's
# own stopping slack (~1e-9 relative in psi at step 1 with the shipped
# atol=1e-8) so the horizon measures divergence rather than solver noise.
AGREEMENT_THRESHOLD = 1e-6

PER_STEP_METRICS = ("max_abs", "max_norm", "rel_l2", "pattern_corr")

# Below this many recorded steps the run has not produced two independent
# halves to average over, and the climatology comparison is noise. A short
# smoke run should be judged point-wise instead.
MIN_CLIMATOLOGY_RECORDS = 20

# A field whose reference run varies by less than this fraction of its own
# magnitude over the run is effectively constant; its climatology ratio is
# roundoff over roundoff. See `climatology`.
MIN_SELF_SPREAD_FRACTION = 1e-8


def _finite_pair(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    good = np.isfinite(a) & np.isfinite(b)
    return a[good], b[good]


def field_metrics(mini, real):
    """
    Per-step metrics for one field. NaN in either input is dropped, not
    propagated.

    rel_l2 and pattern_corr are computed on inputs rescaled by their largest
    magnitude. Both are scale-invariant, so this changes nothing for a
    healthy run -- but on the last records before a diverging variant blows
    up the raw values are large enough that squaring them overflows float64,
    and every metric comes back inf or NaN just when the interesting part of
    the trajectory is being measured.
    """
    a, b = _finite_pair(mini, real)
    if a.size == 0:
        return dict(max_abs=np.nan, max_norm=np.nan, rel_l2=np.nan, pattern_corr=np.nan)

    max_abs = float(np.max(np.abs(a - b)))

    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    if peak == 0:
        return dict(max_abs=max_abs, max_norm=0.0, rel_l2=0.0, pattern_corr=1.0)
    a, b = a / peak, b / peak

    rms_real = float(np.sqrt(np.mean(b**2)))
    max_norm = (max_abs / peak) / rms_real if rms_real > 0 else np.nan

    norm_real = float(np.linalg.norm(b))
    rel_l2 = float(np.linalg.norm(a - b) / norm_real) if norm_real > 0 else 0.0

    # anomaly correlation: constant fields (and a constant offset between
    # them) carry no pattern, so report 1.0 rather than a 0/0
    am, bm = a - a.mean(), b - b.mean()
    denom = float(np.linalg.norm(am) * np.linalg.norm(bm))
    pattern_corr = float(np.dot(am, bm) / denom) if denom > 0 else 1.0

    return dict(max_abs=max_abs, max_norm=max_norm, rel_l2=rel_l2, pattern_corr=pattern_corr)


def evolution(timesteps, mini_states, real_states):
    """
    {field: {metric: array over recorded steps}} for every field both sides
    recorded. Shorter of the two state lists wins, so a run truncated by
    veros's divergence check still yields metrics over its valid prefix.
    """
    n = min(len(mini_states), len(real_states), len(timesteps))
    fields = sorted(set(mini_states[0]) & set(real_states[0])) if n else []

    out = {}
    for field in fields:
        rows = [field_metrics(mini_states[i][field], real_states[i][field]) for i in range(n)]
        out[field] = {m: np.asarray([r[m] for r in rows]) for m in PER_STEP_METRICS}
    return out


def agreement_horizon(timesteps, max_norm, threshold=AGREEMENT_THRESHOLD):
    """
    First recorded step whose scale-normalized max error exceeds `threshold`,
    or the last recorded step if it never does (i.e. "at least this long").

    Returns (step, exceeded). `exceeded=False` means the run ended still in
    agreement, so the step is a lower bound.
    """
    n = min(len(timesteps), len(max_norm))
    for i in range(n):
        if np.isfinite(max_norm[i]) and max_norm[i] > threshold:
            return int(timesteps[i]), True
    return (int(timesteps[n - 1]) if n else 0), False


def climatology(mini_frames, real_frames):
    """
    Time-mean difference over the second half of the run, against the same
    statistic measured on the reference run alone.

    real's 3rd-quarter mean vs its 4th-quarter mean is a same-model,
    same-length sample of the same quantity, so it is how much of
    `mean_diff` a finite averaging window produces with no model difference
    at all. Needs a run long enough for the halves to be independent
    samples; returns None below MIN_CLIMATOLOGY_RECORDS.

    `comparable` is False when the reference run has essentially no internal
    variability in this field, which makes the ratio noise divided by noise.
    acc's salt is the case that forced this: it is a constant field, so both
    the model difference (2.8e-10) and veros's own spread (6.7e-12) are pure
    roundoff, and their ratio reads 41 -- a meaningless number that would
    otherwise dominate the max over fields. Judge such a field on `rel_l2`
    instead.
    """
    mini = np.asarray(mini_frames, dtype=np.float64)
    real = np.asarray(real_frames, dtype=np.float64)
    n = min(len(mini), len(real))
    if n < MIN_CLIMATOLOGY_RECORDS:
        return None
    mini, real = mini[:n], real[:n]

    # same overflow guard as field_metrics: on a diverging run the late
    # frames are huge but still finite, and squaring them overflows. Every
    # number below is a ratio or is reported alongside `field_rms`, so
    # rescaling is exact.
    peak = float(np.nanmax(np.abs(real)))
    if peak > 0:
        mini, real = mini / peak, real / peak
    else:
        peak = 1.0

    half, quarter = n // 2, n // 4
    mean_diff = np.nanmean(mini[half:], axis=0) - np.nanmean(real[half:], axis=0)
    self_diff = np.nanmean(real[2 * quarter : 3 * quarter], axis=0) - np.nanmean(real[3 * quarter :], axis=0)

    def stats(d):
        return float(np.nanmax(np.abs(d))), float(np.sqrt(np.nanmean(d**2)))

    mean_max, mean_rms = (v * peak for v in stats(mean_diff))
    self_max, self_rms = (v * peak for v in stats(self_diff))
    field_rms = float(np.sqrt(np.nanmean(real[half:] ** 2))) * peak
    comparable = bool(field_rms > 0 and self_rms > MIN_SELF_SPREAD_FRACTION * field_rms)
    return dict(
        mean_max=mean_max,
        mean_rms=mean_rms,
        self_max=self_max,
        self_rms=self_rms,
        field_rms=field_rms,
        comparable=comparable,
        # >1 means the two models' climatologies differ by more than the
        # reference model differs from itself over an equally long window
        ratio_max=mean_max / self_max if self_max > 0 else np.inf,
        ratio_rms=mean_rms / self_rms if self_rms > 0 else np.inf,
        internal_std=float(np.nanmax(real[half:].std(axis=0))) * peak,
    )


def first_nonfinite(timesteps, states):
    """First recorded step at which any field stopped being finite, else None."""
    for i, step in enumerate(timesteps[: len(states)]):
        for field, value in states[i].items():
            if not np.all(np.isfinite(np.asarray(value))):
                return int(step), field
    return None
