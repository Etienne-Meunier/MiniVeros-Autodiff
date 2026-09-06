#!/usr/bin/env python3
"""
Renders the trust report from $STORE/MiniVeros-Autodiff/results/*.npz
(produced by generate_matrix_data.py): per-variant error-evolution curves,
a mini_veros/veros side-by-side field-evolution gif, a timing summary
across the whole matrix, and a short matrix_report.md tying it together.
Data stays in $STORE (results/*.npz can be regenerated); the report and
its figures are committed to the repo, under report/matrix_figures/ and
report/matrix_report.md.

Results live in "results/{run_id}/{variant}.npz" -- one directory per run --
so --run-id says which run to render. It is required, and it is matched exactly: a
variant with no file for that id becomes an empty row and a warning in the
banner, never a row filled in from some other run. That fallback is what once
made three crashed variants the report's only three "passing" rows, so the
resolution rule is gone rather than guarded -- there is no "latest".

Run ids come from wherever the run was launched: a wandb sweep id for a swept
matrix (every agent shares it without coordinating), or the "local-..." id
generate_matrix_data.py assigns otherwise. `plot_matrix_report.py --run-id
nosuch` lists the ids actually present.

Kept separate from data generation so re-plotting doesn't require
re-running the (slow) simulations.

The report is half hand-written: only the regions fenced by
"<!-- AUTO:key -->" markers (run banner, timing figure, summary table,
per-variant sections) are regenerated.
Prose outside them -- the intro, the metric definitions, any commentary
added by hand -- is copied through untouched, and a block whose markers
were deleted is not written back. --rewrite discards all of it and
regenerates the report from scratch.

Usage:
    python test/plot_matrix_report.py --run-id be7b0pze          # a sweep
    python test/plot_matrix_report.py --run-id local-20260828T143000Z
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

import metrics
from run_ids import require_run_id, result_path
from setups_matrix import FAMILIES, VARIANTS
from variant_util import DEFAULT_SOLVER_ATOL

STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
BASE_DIR = STORE / "MiniVeros-Autodiff"
RESULTS_DIR = BASE_DIR / "results"
REPORT_DIR = REPO_ROOT / "report"
FIG_DIR = REPORT_DIR / "matrix_figures"

# Agreement gate. Both numbers are scale-free (see test/metrics.py), so the
# same threshold means the same thing for psi in m^3/s and temp in K --
# unlike the old point-wise max_rel, which saturates at 2.0 as soon as one
# cell holds two tiny values of opposite sign and so read ~2.0 for almost
# every 30-year row regardless of how close the runs were.
# Both sit at the floor the shipped elliptic solver imposes anyway:
# bicgstab(tol=0, atol=1e-8) leaves psi about 1e-8 relative loose, so nothing
# can agree better than that no matter how faithful the port is.
MAX_NORM_OK = 1e-6   # max |mini - veros| / rms(veros), used for the agreement-horizon column

SNAPSHOT_FIELDS = ("temp", "psi")


def plot_error_evolution(name, data):
    """
    Three panels, one per metric that stays meaningful once the runs separate:
    scale-normalized max, relative L2, and pattern correlation. The old
    point-wise max_rel panel is gone -- it read ~2.0 for every long run and
    told you nothing (see test/metrics.py).
    """
    timesteps = data["timesteps"]
    fields = sorted(k[2:-len("_max_norm")] for k in data.files if k.startswith("m_") and k.endswith("_max_norm"))
    if not fields or len(timesteps) == 0:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for field in fields:
        axes[0].plot(timesteps, _positive(data[f"m_{field}_max_norm"]), marker="o", ms=3, label=field)
        axes[1].plot(timesteps, _positive(data[f"m_{field}_rel_l2"]), marker="o", ms=3, label=field)
        axes[2].plot(timesteps, data[f"m_{field}_pattern_corr"], marker="o", ms=3, label=field)

    for ax in axes[:2]:
        ax.set_yscale("log")
        ax.set_ylim(1e-17, 1e2)
        ax.axhline(MAX_NORM_OK, color="k", ls=":", lw=1)
        ax.grid(alpha=0.3)
    axes[0].set_title("max |mini - veros| / rms(veros)")
    axes[1].set_title(r"$\||$mini - veros$\||_2\, /\, \||$veros$\||_2$")
    axes[2].set_title("pattern correlation (1 = same field)")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].axhline(1.0, color="k", ls=":", lw=1)
    axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel("step")
    axes[0].legend(fontsize=8, ncol=2)

    fig.suptitle(f"{name}: mini_veros vs veros")
    fig.tight_layout()
    out = FIG_DIR / f"{name}_error_evolution.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _positive(values):
    """Zeros as NaN so a log axis skips an exact match instead of bottoming out."""
    arr = np.asarray(values, dtype=np.float64).copy()
    arr[arr <= 0] = np.nan
    return arr


def _field_frames(data, field):
    """(mini_frames, real_frames, timesteps, level_note) as 2D-per-step arrays,
    or None if this field's frames weren't recorded. Shared by make_gif and
    make_diff_gif so both slice the uppermost level the same way."""
    key_mini, key_real = f"{field}_mini_frames", f"{field}_real_frames"
    if key_mini not in data:
        return None

    mini_frames = data[key_mini]
    real_frames = data[key_real]
    timesteps = data["timesteps"]

    # 3D field (x, y, z): uppermost level as a 2D "surface" slice. A run with
    # --snapshot-surface-only already stored just that slice, so the label has
    # to come from the flag rather than from the array's rank.
    surface_only = bool(data["snapshot_surface_only"]) if "snapshot_surface_only" in data.files else False
    if mini_frames.ndim == 4:
        mini_frames = mini_frames[:, :, :, -1]
        real_frames = real_frames[:, :, :, -1]
        level_note = " (uppermost level)"
    else:
        level_note = " (uppermost level)" if surface_only and field != "psi" else ""

    return mini_frames, real_frames, timesteps, level_note


def make_gif(name, data, field):
    prepared = _field_frames(data, field)
    if prepared is None:
        return None
    mini_frames, real_frames, timesteps, level_note = prepared

    t_width = len(str(int(timesteps[-1])))

    gif_frames = []
    for i, t in enumerate(timesteps):
        # Per-frame vmin/vmax (shared between the two panels, so they stay
        # comparable) instead of one fixed range for the whole animation --
        # a range picked once from the last frame's magnitude washes out
        # early, smaller-amplitude steps. turbo instead of a diverging
        # colormap since the range is no longer forced symmetric about zero.
        vmin = min(np.nanmin(mini_frames[i]), np.nanmin(real_frames[i]))
        vmax = max(np.nanmax(mini_frames[i]), np.nanmax(real_frames[i]))
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0

        # Fixed subplots_adjust instead of per-frame tight_layout(): tight_layout()
        # recomputes margins from the rendered title/tick text extents each call, and
        # the step number's growing digit count (step 0 vs step 300) shifted those
        # margins slightly frame to frame -- reads as a flickering background when
        # played as a gif. A constant rect makes every frame pixel-aligned except
        # the actual data. Fixed-width step number (zero-padded) for the same reason.
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
        axes[0].imshow(mini_frames[i].T, origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
        axes[0].set_title("mini_veros")
        im = axes[1].imshow(real_frames[i].T, origin="lower", cmap="turbo", vmin=vmin, vmax=vmax)
        axes[1].set_title("veros")
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02)
        fig.suptitle(f"{name}: {field}{level_note}  (step {int(t):0{t_width}d})")
        fig.subplots_adjust(left=0.06, right=0.88, top=0.82, bottom=0.08, wspace=0.15)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        gif_frames.append(frame)
        plt.close(fig)

    out = FIG_DIR / f"{name}_{field}_evolution.gif"
    imageio.mimsave(out, gif_frames, duration=0.4, loop=0)
    return out


def make_diff_gif(name, data, field):
    """mini - veros, one panel, animated. Diverging colormap centered on
    zero (unlike the turbo state gif, sign matters here); vmax is per-frame
    so a variant whose disagreement grows over the run doesn't wash out its
    own early frames."""
    prepared = _field_frames(data, field)
    if prepared is None:
        return None
    mini_frames, real_frames, timesteps, level_note = prepared
    diff_frames = mini_frames - real_frames

    t_width = len(str(int(timesteps[-1])))

    gif_frames = []
    for i, t in enumerate(timesteps):
        vmax = np.nanmax(np.abs(diff_frames[i]))
        vmax = vmax if vmax > 0 else 1.0

        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        im = ax.imshow(diff_frames[i].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{name}: {field}{level_note} diff (mini - veros)\nstep {int(t):0{t_width}d}")
        fig.subplots_adjust(left=0.1, right=0.86, top=0.78, bottom=0.08)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        gif_frames.append(frame)
        plt.close(fig)

    out = FIG_DIR / f"{name}_{field}_diff.gif"
    imageio.mimsave(out, gif_frames, duration=0.4, loop=0)
    return out


def _fields_with(data, suffix):
    return sorted(k[2:-len(suffix)] for k in data.files if k.startswith("m_") and k.endswith(suffix))


def variant_summary(name, data):
    """
    Metrics worth printing for one variant, read off its .npz.

    No pass/fail verdict here -- just the numbers (max_norm, rel_l2,
    horizon, pattern_corr, clim_ratio) at the final recorded step. A
    variant that couldn't be run, or blew up mid-run, gets a `message`
    key and (for the mid-run case) the numbers that would just measure
    the explosion left out.
    """
    status_field = str(data["status"]) if "status" in data.files else "ok"
    if status_field == "error":
        message = str(data["error_message"]) if "error_message" in data.files else "unknown error"
        return dict(message=message)

    fields = _fields_with(data, "_max_norm")
    if not fields:
        return dict(message="no metrics recorded (regenerate with the current generate_matrix_data.py)")

    worst_max_norm = max(float(data[f"m_{f}_max_norm"][-1]) for f in fields)
    worst_rel_l2 = max(float(data[f"m_{f}_rel_l2"][-1]) for f in fields)
    horizon = min(int(data[f"m_{f}_agreement_horizon"]) for f in fields)
    horizon_exceeded = any(bool(data[f"m_{f}_agreement_exceeded"]) for f in fields)
    worst_corr = min(float(data[f"m_{f}_pattern_corr"][-1]) for f in fields)

    # only fields where veros itself varies enough for the ratio to mean
    # something -- see metrics.climatology's `comparable`
    ratios = {
        f: float(data[f"c_{f}_ratio_rms"])
        for f in fields
        if f"c_{f}_ratio_rms" in data.files and bool(data.get(f"c_{f}_comparable", True))
    }
    skipped = [
        f for f in fields
        if f"c_{f}_ratio_rms" in data.files and not bool(data.get(f"c_{f}_comparable", True))
    ]
    worst_ratio = max(ratios.values()) if ratios else None

    summary = dict(
        max_norm=worst_max_norm,
        rel_l2=worst_rel_l2,
        horizon=horizon,
        horizon_exceeded=horizon_exceeded,
        pattern_corr=worst_corr,
        clim_ratio=worst_ratio,
        clim_field=max(ratios, key=ratios.get) if ratios else None,
        clim_skipped=skipped,
    )

    if status_field == "diverged":
        # The last kept records are mid-blow-up: the fields are growing
        # without bound, so "rel L2 at the final step" is 1e23 and the
        # climatology averages over an explosion. Neither number says
        # anything about the port. The horizon -- how long the two codes
        # tracked before the configuration went unstable -- does.
        summary.update(rel_l2=None, clim_ratio=None, clim_field=None, pattern_corr=None,
                        message="diverged mid-run; rel L2/corr/clim ratio omitted (they'd measure the explosion, not the port)")
    return summary


def resolve_npz(name, run_id):
    """Path to variant `name`'s .npz within run `run_id`, or None.

    Exact match, deliberately: there is no fallback to another run. A variant
    absent from this run renders as a missing row, which is a visible gap --
    the older "newest file wins" rule filled that gap silently from a previous
    run, and a short older run reads as a pass.
    """
    exact = result_path(RESULTS_DIR, run_id, name)
    return exact if exact.exists() else None


def missing_row(variant):
    """A variant with no .npz in this run. Rendered as an explicit row rather
    than omitted: an absent row looks like a variant nobody asked for, while a
    row saying "no result" is a question."""
    return dict(
        name=variant["name"],
        family=variant["family"],
        group=FAMILIES[variant["family"]]["group"],
        overrides=variant["overrides"],
        mini_ms=None, real_ms=None,
        summary={"message": "no result in this run -- the variant never wrote an .npz"},
        has_data=False,
        err_png=None, gifs=[],
        generated_at=None, run_config=None, solver_atol=None, device=None,
    )


def render_variant(variant, run_id):
    name = variant["name"]
    npz_path = resolve_npz(name, run_id)
    if npz_path is None:
        return None
    data = np.load(npz_path)

    summary = variant_summary(name, data)
    has_data = len(data["timesteps"]) > 0

    err_png = plot_error_evolution(name, data) if has_data else None
    gifs = [g for f in SNAPSHOT_FIELDS if has_data and (g := make_gif(name, data, f)) is not None]
    gifs += [g for f in SNAPSHOT_FIELDS if has_data and (g := make_diff_gif(name, data, f)) is not None]

    generated_at = str(data["generated_at"]) if "generated_at" in data.files else None
    run_config = json.loads(str(data["run_config_json"])) if "run_config_json" in data.files else None

    def ms(key):
        value = data[key]
        return float(value) * 1000 if value.shape == () and np.isfinite(value) else None

    return dict(
        name=name,
        family=str(data["family"]),
        group=str(data["group"]),
        overrides=json.loads(str(data["overrides_json"])),
        mini_ms=ms("mini_sec_per_step"),
        real_ms=ms("real_sec_per_step"),
        summary=summary,
        has_data=has_data,
        err_png=err_png,
        gifs=gifs,
        generated_at=generated_at,
        run_config=run_config,
        # absent in every file written before --solver-atol existed; those were
        # all produced at the tolerance both codes ship with
        solver_atol=float(data["solver_atol"]) if "solver_atol" in data.files else None,
        device=str(data["device"]) if "device" in data.files else None,
    )


def plot_timing_summary(rows):
    rows = [r for r in rows if r["mini_ms"] is not None and r["real_ms"] is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: (r["group"], r["name"]))

    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(rows)), 4.5))
    x = np.arange(len(rows))
    width = 0.38
    ax.bar(x - width / 2, [r["mini_ms"] for r in rows], width, label="mini_veros")
    ax.bar(x + width / 2, [r["real_ms"] for r in rows], width, label="veros")
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("ms / step")
    ax.set_yscale("log")
    ax.set_title("average wall time per step")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIG_DIR / "timing_summary.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def format_ts(ts):
    """Run timestamp ("20260828T082645Z") as a natural date/hour string, for
    humans reading the report's provenance line alongside the raw stamp."""
    try:
        return datetime.strptime(ts, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return ts


def device_summary(rows):
    """
    Which jax platform produced these rows. Files predating --device were all
    CPU. Worth stating for the same reason as the tolerance: ms/step and the
    speedup column are hardware-specific, and float64 is throttled 1:32-1:64
    on the workstation-class GPUs this runs on.
    """
    seen = {r["device"] for r in rows}
    labelled = {(d if d is not None else "cpu") for d in seen}
    assumed = " (assumed; not recorded in these files)" if None in seen else ""
    if len(labelled) == 1:
        return f"Run on {next(iter(labelled))}{assumed}."
    return (
        "**Warning: mixed devices** -- "
        + ", ".join(f"`{d}`" for d in sorted(labelled))
        + ". Timings are not comparable across them."
    )


def solver_atol_summary(rows):
    """
    Which elliptic-solver stopping rule produced these rows.

    Worth stating on every report: the `horizon` column is not comparable
    across tolerances, since a looser solver injects a larger seed and so
    crosses the agreement gate sooner. Files written before --solver-atol
    existed carry no value and were all produced at the shipped 1e-8.
    """
    seen = {r["solver_atol"] for r in rows}
    labelled = {(a if a is not None else DEFAULT_SOLVER_ATOL) for a in seen}
    assumed = " (assumed; not recorded in these files)" if None in seen else ""
    if len(labelled) == 1:
        atol = next(iter(labelled))
        shipped = " -- the tolerance both codes ship with" if atol == DEFAULT_SOLVER_ATOL else ""
        return f"Elliptic solver forced to atol={atol:g} on both sides{assumed}{shipped}."
    return (
        "**Warning: mixed solver tolerances** -- "
        + ", ".join(f"`{a:g}`" for a in sorted(labelled))
        + ". The horizon column is not comparable across them."
    )


def run_summary(rows, run_id):
    """
    Provenance banner.

    Every row here comes from `run_id` by construction -- rendering resolves
    each variant to that id exactly, with no fallback -- so there is no
    mixed-snapshot case left to warn about. What can still happen is a variant
    with no result in the run, which shows as a missing row, so the banner
    counts those instead.

    `generated_at` is reported per run as provenance (when the work actually
    ran). It no longer selects anything.
    """
    missing = sorted(r["name"] for r in rows if r["summary"].get("message", "").startswith("no result"))
    # Rows with no file carry solver_atol=None and device=None, which both
    # summaries would read as "the old default" and report as a mix. Describe
    # only the results that exist.
    present = [r for r in rows if r["name"] not in set(missing)]
    known = [r for r in present if r["generated_at"]]
    atol_note = f"{solver_atol_summary(present)} {device_summary(present)}" if present else ""

    if not known:
        head = f"run: `{run_id}` (no generation timestamp recorded for these results)."
    else:
        stamps = sorted(r["generated_at"] for r in known)
        span = (f"`{stamps[0]}` ({format_ts(stamps[0])})" if stamps[0] == stamps[-1]
                else f"`{stamps[0]}`..`{stamps[-1]}` ({format_ts(stamps[0])} to {format_ts(stamps[-1])})")
        # A spread here is normal, not a warning: agents finish at different
        # times. It is one run because it is one id.
        head = f"run: `{run_id}`, generated {span}. {atol_note}"

    if missing:
        return (
            f"> **Warning: {len(missing)} variant(s) have no result in this run:** "
            f"{', '.join(f'`{n}`' for n in missing)}. Their rows are empty rather than filled "
            f"from an earlier run. Rerun them with "
            f"`--run-id {run_id} --variant <name>` to complete the set.\n\n"
            f"{head}"
        )
    return head


def _tex_pow10(value):
    """1e-06 as LaTeX 10^{-6}, so a threshold reads as maths inside $...$."""
    exponent = int(round(np.log10(value)))
    return f"10^{{{exponent}}}"


def metrics_section():
    """
    Definitions of every column, in LaTeX.

    Kept in the generated report rather than only in test/metrics.py, so a
    number in the table can be checked without reading the code that made it.
    """
    return [
        "## Metrics",
        "",
        "For one field at one recorded step, let $m$ and $v$ be the mini_veros and veros "
        "fields flattened over the grid, $N$ their length, and $\\overline{x}$ a spatial mean. "
        "Cells where either side is not finite are dropped before any of this.",
        "",
        "**Scale-normalized max error.** The largest point-wise disagreement, divided by the "
        "reference field's own magnitude so that the same threshold means the same thing for "
        "$\\psi$ in $\\mathrm{m^3\\,s^{-1}}$ and for temperature in K:",
        "",
        "$$\\mathrm{max\\_norm} \\;=\\; \\frac{\\max_i \\left| m_i - v_i \\right|}"
        "{\\mathrm{rms}(v)}, \\qquad "
        "\\mathrm{rms}(x) = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N} x_i^2}$$",
        "",
        "**Relative $L_2$ error.** The whole-field distance. Unlike a point-wise relative "
        "error it does not blow up where the field passes through zero, which is what made "
        "the old `max_rel` column saturate at 2 for every long run:",
        "",
        "$$\\mathrm{rel\\_L2} \\;=\\; \\frac{\\| m - v \\|_2}"
        "{\\| v \\|_2}$$",
        "",
        "**Pattern correlation.** The Pearson correlation of the two fields' anomalies. It "
        "separates *same solution, shifted or rescaled* from *different weather*: a value of "
        "1 with a large `rel_L2` means the structure survived and only its amplitude moved.",
        "",
        "$$\\mathrm{corr} \\;=\\; \\frac{\\sum_i (m_i - \\overline{m})(v_i - \\overline{v})}"
        "{\\| m - \\overline{m} \\|_2 \\; \\| v - \\overline{v} \\|_2}$$",
        "",
        f"**Agreement horizon.** The first recorded step at which any field's "
        f"$\\mathrm{{max\\_norm}}$ crosses $\\varepsilon = {_tex_pow10(MAX_NORM_OK)}$. The table "
        "prints "
        "`>T` when the run ended without ever crossing it, so `T` is a lower bound:",
        "",
        "$$T_{\\mathrm{agree}} \\;=\\; \\min\\left\\{\\, t \\;:\\; "
        "\\mathrm{max\\_norm}(t) > \\varepsilon \\,\\right\\}$$",
        "",
        "**Climatology ratio.** The one statement that survives chaotic separation. Write "
        "$\\langle x \\rangle_{A}$ for the time mean over a window $A$ of records, and split a "
        "run of $R$ records into its second half $H$ and its third and fourth quarters "
        "$Q_3, Q_4$. Then $D$ is how far apart the two models' climatologies are, and $S$ is "
        "how far veros lands from *itself* when the same statistic is measured over two "
        "successive windows of the same length:",
        "",
        "$$D \\;=\\; \\langle m \\rangle_{H} - \\langle v \\rangle_{H}, \\qquad "
        "S \\;=\\; \\langle v \\rangle_{Q_3} - \\langle v \\rangle_{Q_4}, \\qquad "
        "\\mathrm{clim\\ ratio} \\;=\\; \\frac{\\mathrm{rms}(D)}{\\mathrm{rms}(S)}$$",
        "",
        "A ratio below 1 means the two models agree with each other better than veros agrees "
        "with itself over an equally long window -- the strongest claim a 30-year comparison "
        "of a chaotic flow can support. The ratio is only reported where veros actually "
        f"varies, $\\mathrm{{rms}}(S) > {_tex_pow10(metrics.MIN_SELF_SPREAD_FRACTION)} \\cdot "
        "\\mathrm{rms}(v)$; below that the field is effectively constant and the ratio "
        "is roundoff divided by roundoff (acc's `salt` scored 41 that way).",
        "",
    ]


def _fmt(value, spec=".2e", missing="-"):
    return missing if value is None else format(value, spec)




# --- report assembly ---------------------------------------------------
#
# The report is part generated, part hand-written: the prose (intro, metric
# definitions, any commentary added by hand) lives in the .md and must
# survive a re-run, while the table, the figures and the per-variant
# sections are rebuilt from the .npz files every time. Each generated
# region is fenced by "<!-- AUTO:key -->" markers; a re-run rewrites only
# what is inside them and copies everything else through byte for byte.
# A region whose markers were deleted from the .md stays deleted -- the
# script never adds a section back to a report someone has pruned.

# "timestamp" is now the run banner. The key is the literal marker text in
# report/matrix_report.md, and a block whose markers don't match is silently
# not regenerated -- so it keeps its old name rather than orphaning the
# banner in every report already committed.
AUTO_BLOCKS = ("timestamp", "timing", "table", "detail")


def _marker(key, closing=False):
    return f"<!-- {'/' if closing else ''}AUTO:{key} -->"


def _horizon_str(r):
    horizon = r["summary"].get("horizon")
    if horizon is None:
        return "-"
    return f"{horizon}" if r["summary"].get("horizon_exceeded") else f">{horizon}"


def _speedup(r):
    return f"{r['real_ms'] / r['mini_ms']:.1f}x" if r["mini_ms"] and r["real_ms"] else "-"


# Column header as it appears in the .md -> how to fill that cell. A report
# whose header lists a subset, or a different order, keeps its own header:
# only the columns it asks for are regenerated.
TABLE_COLUMNS = {
    "variant": lambda r: r["name"],
    "group": lambda r: r["group"],
    "horizon": _horizon_str,
    "rel L2": lambda r: _fmt(r["summary"].get("rel_l2")),
    "corr": lambda r: _fmt(r["summary"].get("pattern_corr"), ".4f"),
    "clim ratio": lambda r: _fmt(r["summary"].get("clim_ratio"), ".2f"),
    "mini ms/step": lambda r: _fmt(r["mini_ms"], ".2f"),
    "veros ms/step": lambda r: _fmt(r["real_ms"], ".2f"),
    "speedup": _speedup,
}
# the columns that still say something for a variant that never ran
ALWAYS_FILLED = ("variant", "group")


def table_block(rows, columns=None):
    columns = list(columns) if columns else list(TABLE_COLUMNS)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        cells = []
        for c in columns:
            fill = TABLE_COLUMNS.get(c)
            if fill is None or (not r["has_data"] and c not in ALWAYS_FILLED):
                cells.append("-")
            else:
                cells.append(fill(r))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def detail_block(rows):
    lines = []
    for r in rows:
        lines.append(f"### {r['name']}")
        if r["overrides"]:
            lines.append(f"overrides: `{r['overrides']}`")
        if r["generated_at"]:
            cfg = r["run_config"]
            cfg_note = f", {cfg['n_steps']} steps @ interval {cfg['record_interval']}" if cfg else ""
            lines.append(f"generated: `{r['generated_at']}`{cfg_note}")
        if not r["has_data"]:
            lines += ["", f"no data: `{r['summary'].get('message', 'unknown')}`", ""]
            continue
        lines.append("")
        if r["summary"].get("message"):
            lines.append(f"note: {r['summary']['message']}")
        if r["err_png"]:
            lines.append(f"![errors]({r['err_png'].relative_to(REPORT_DIR)})")
        for g in r["gifs"]:
            lines.append(f"![{g.stem}]({g.relative_to(REPORT_DIR)})")
        lines.append("")
    return lines


def _existing_columns(inner):
    """The table header the .md already carries, so a hand-trimmed column
    set is regenerated as-is instead of being reset to all ten columns."""
    for line in inner.splitlines():
        s = line.strip()
        if s.startswith("|") and set(s) - set("|-: "):
            return [c.strip() for c in s.strip("|").split("|")]
    return None


def _splice(text, key, build):
    """Replace what sits between key's markers, leaving the rest of the
    document alone. Returns (text, found)."""
    open_m, close_m = _marker(key), _marker(key, True)
    match = re.search(re.escape(open_m) + r"\n?(.*?)\n?" + re.escape(close_m), text, re.S)
    if match is None:
        return text, False
    body = "\n".join(build(match.group(1)))
    return text[:match.start()] + f"{open_m}\n{body}\n{close_m}" + text[match.end():], True


def fresh_report(rows, timing_png, run_id):
    """The full report, markers included -- used only when there is no .md
    to update (or --rewrite says to throw the current one away)."""

    def fenced(key, body):
        return [_marker(key), *body, _marker(key, True), ""]

    lines = [
        "# mini_veros vs veros: comparison matrix report",
        "",
        *fenced("timestamp", [run_summary(rows, run_id)]),
        f"{len(rows)} variants.",
        "",
        "**How to read this.** A 30-year run cannot agree point-wise -- roundoff-level "
        "differences grow to the size of the flow's own variability, and real veros does the "
        "same against itself (see `divergence_report.md`). The numbers below quantify how far "
        "apart the two models are without collapsing that into a pass/fail call; **clim ratio** "
        "below 1.0 means the models are closer to each other than veros is to itself over an "
        "equally long window, which is the strongest claim a 30-year comparison can support.",
        "",
        f"Columns: **horizon** is the last step at which every field still agreed to a "
        f"scale-normalized max error below {MAX_NORM_OK:.0e} (`>` means it never stopped "
        "agreeing); **rel L2** is the relative L2 distance at the final step; **corr** is the "
        "worst field's pattern correlation there; **clim ratio** is the climatology difference "
        "divided by veros's own. Fields the reference run holds essentially constant (acc's "
        "`salt`) are left out of the ratio: there the comparison would be roundoff over "
        "roundoff. They are judged on rel L2 like everything else. The horizon depends on the "
        "solver tolerance named above -- a looser solver injects a larger seed and crosses the "
        "gate sooner -- so only compare it across rows produced at the same one. Definitions below.",
        "",
    ]
    lines += metrics_section()
    lines += fenced("timing", [f"![timing]({timing_png.relative_to(REPORT_DIR)})"] if timing_png else [])
    lines += fenced("table", table_block(rows))
    lines += ["", "## per-variant detail", "", *fenced("detail", detail_block(rows))]
    return lines


def write_report(rows, timing_png, run_id, rewrite=False):
    rows = sorted(rows, key=lambda r: (r["group"], r["name"]))
    out = REPORT_DIR / "matrix_report.md"
    existing = out.read_text() if out.exists() else None

    def builder(key):
        def build(inner):
            if key == "timestamp":
                return [run_summary(rows, run_id)]
            if key == "timing":
                return [f"![timing]({timing_png.relative_to(REPORT_DIR)})"] if timing_png else []
            if key == "table":
                columns = _existing_columns(inner)
                unknown = [c for c in (columns or []) if c not in TABLE_COLUMNS]
                if unknown:
                    print(f"warning: table header asks for unknown column(s) {unknown}; filled with '-'")
                return table_block(rows, columns)
            return detail_block(rows)
        return build

    if existing is None or rewrite:
        out.write_text("\n".join(fresh_report(rows, timing_png, run_id)))
        return out

    if _marker("table") not in existing:
        raise SystemExit(
            f"{out} carries no <!-- AUTO:... --> markers, so there is no way to tell its "
            f"hand-written text from the generated blocks; refusing to overwrite it. Fence "
            f"the generated regions with {_marker('table')} / {_marker('table', True)} (same "
            f"for {', '.join(k for k in AUTO_BLOCKS if k != 'table')}), or pass --rewrite to "
            f"regenerate the whole report from scratch."
        )

    text, updated = existing, []
    for key in AUTO_BLOCKS:
        text, found = _splice(text, key, builder(key))
        if found:
            updated.append(key)
    out.write_text(text)
    print(f"updated blocks: {', '.join(updated)} (everything outside the markers left as-is)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True,
                         help="the run to render, e.g. a wandb sweep id or a 'local-...' id from "
                              "generate_matrix_data.py --run-id. Required and exact: the report is "
                              "one run, never a mix of whatever files happen to be newest.")
    parser.add_argument("--rewrite", action="store_true",
                         help="regenerate report/matrix_report.md from scratch, discarding any "
                              "hand-written text in it; the default updates only the AUTO blocks")
    args = parser.parse_args()

    require_run_id(parser, RESULTS_DIR, args.run_id)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        row = render_variant(variant, args.run_id)
        if row is None:
            # An absent variant is part of the run's result, not an absence of
            # one: render it as an empty row so the report shows the hole.
            print(f"MISSING {variant['name']}: no {args.run_id}/{variant['name']}.npz")
            rows.append(missing_row(variant))
            continue
        rows.append(row)
        summary = row["summary"]
        detail = summary.get("message") or (
            f"rel_l2={_fmt(summary.get('rel_l2'))} clim_ratio={_fmt(summary.get('clim_ratio'), '.2f')}"
        )
        print(f"{row['name']}: {detail}")

    timing_png = plot_timing_summary(rows)
    report = write_report(rows, timing_png, args.run_id, rewrite=args.rewrite)
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
