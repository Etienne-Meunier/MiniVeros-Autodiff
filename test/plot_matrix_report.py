#!/usr/bin/env python3
"""
Renders the trust report from $STORE/MiniVeros-Autodiff/results/*.npz
(produced by generate_matrix_data.py): per-variant error-evolution curves,
a mini_veros/veros side-by-side field-evolution gif, a timing summary
across the whole matrix, and a short matrix_report.md tying it together.
Data stays in $STORE (results/*.npz can be regenerated); the report and
its figures are committed to the repo, under report/matrix_figures/ and
report/matrix_report.md.

Each result file is named "{variant}__{timestamp}.npz" -- re-running
generate_matrix_data.py adds a new one rather than overwriting, so
--timestamp picks which snapshot to render. Default "latest" takes the
newest file per variant (variants can end up on different snapshots after
a partial rerun -- the report's first line says which timestamp(s) it used).

Kept separate from data generation so re-plotting doesn't require
re-running the (slow) simulations.

Usage:
    python test/plot_matrix_report.py                     # latest snapshot per variant
    python test/plot_matrix_report.py --timestamp 20260828T143000Z
"""

import argparse
import json
import os
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

from setups_matrix import VARIANTS

STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
BASE_DIR = STORE / "MiniVeros-Autodiff"
RESULTS_DIR = BASE_DIR / "results"
REPORT_DIR = REPO_ROOT / "report"
FIG_DIR = REPORT_DIR / "matrix_figures"

# same tolerance policy as test_matrix.py -- point-wise rel error for most
# fields, scale-normalized for psi (arbitrary per-island gauge)
RTOL_OK = 1e-6
PSI_SCALE_TOL = 1e-6
SNAPSHOT_FIELDS = ("temp", "psi")


def plot_error_evolution(name, data):
    timesteps = data["timesteps"]
    fields = [k.removeprefix("err_").removesuffix("_max_rel_errors") for k in data.files if k.endswith("_max_rel_errors")]
    fields = sorted(f for f in fields if f != "psi")
    has_psi = "err_psi_max_abs_errors" in data.files

    fig, axes = plt.subplots(1, 2 if not has_psi else 2, figsize=(11, 4.2))
    for field in fields:
        axes[0].plot(timesteps, data[f"err_{field}_max_abs_errors"] + 1e-300, marker="o", ms=3, label=field)
        axes[1].plot(timesteps, data[f"err_{field}_max_rel_errors"] + 1e-300, marker="o", ms=3, label=field)
    if has_psi:
        scale = np.max(np.abs(data["psi_real_frames"][-1])) if "psi_real_frames" in data else 1.0
        scale = scale if scale > 0 else 1.0
        normalized = data["err_psi_max_abs_errors"] / scale
        axes[0].plot(timesteps, normalized + 1e-300, marker="o", ms=3, label="psi (scale-norm.)", color="tab:purple")

    for ax, title in zip(axes, ["max |mini - real|  (psi: scale-normalized)", "max relative error"]):
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e1)
        ax.set_xlabel("step")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    fig.suptitle(f"{name}: mini_veros vs veros error evolution")
    fig.tight_layout()
    out = FIG_DIR / f"{name}_error_evolution.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def make_gif(name, data, field):
    key_mini, key_real = f"{field}_mini_frames", f"{field}_real_frames"
    if key_mini not in data:
        return None

    mini_frames = data[key_mini]
    real_frames = data[key_real]
    timesteps = data["timesteps"]

    # 3D field (x, y, z): uppermost level as a 2D "surface" slice
    if mini_frames.ndim == 4:
        mini_frames = mini_frames[:, :, :, -1]
        real_frames = real_frames[:, :, :, -1]
        level_note = " (uppermost level)"
    else:
        level_note = ""

    vmax = np.nanmax(np.abs(real_frames))
    vmax = vmax if vmax > 0 else 1.0

    t_width = len(str(int(timesteps[-1])))

    gif_frames = []
    for i, t in enumerate(timesteps):
        # Fixed subplots_adjust instead of per-frame tight_layout(): tight_layout()
        # recomputes margins from the rendered title/tick text extents each call, and
        # the step number's growing digit count (step 0 vs step 300) shifted those
        # margins slightly frame to frame -- reads as a flickering background when
        # played as a gif. A constant rect makes every frame pixel-aligned except
        # the actual data. Fixed-width step number (zero-padded) for the same reason.
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
        axes[0].imshow(mini_frames[i].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0].set_title("mini_veros")
        axes[1].imshow(real_frames[i].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title("veros")
        fig.suptitle(f"{name}: {field}{level_note}  (step {int(t):0{t_width}d})")
        fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.08, wspace=0.15)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        gif_frames.append(frame)
        plt.close(fig)

    out = FIG_DIR / f"{name}_{field}_evolution.gif"
    imageio.mimsave(out, gif_frames, duration=0.4, loop=0)
    return out


def variant_status(name, data):
    """
    Pass/fail at the final recorded step, mirroring test_matrix.py exactly:
    psi uses a scale-normalized check (arbitrary per-island gauge); every
    other field uses the stored per-step "passes" from
    compute_error_evolution, which is compare_field's np.allclose(atol,
    rtol) -- NOT a bare max_rel threshold (max_rel alone false-flags u/v,
    whose relative error blows up on near-zero values even when the
    absolute difference is solver noise).
    """
    fields = [f.removeprefix("err_").removesuffix("_max_rel_errors") for f in data.files if f.endswith("_max_rel_errors")]
    ok = True
    worst = 0.0
    for field in fields:
        if field == "psi":
            scale = np.max(np.abs(data["psi_real_frames"][-1])) if "psi_real_frames" in data else 1.0
            scale = scale if scale > 0 else 1.0
            val = float(data["err_psi_max_abs_errors"][-1] / scale)
            ok = ok and val < PSI_SCALE_TOL
        else:
            val = float(data[f"err_{field}_max_rel_errors"][-1])
            passes_key = f"err_{field}_passes"
            if passes_key in data.files:
                ok = ok and bool(data[passes_key][-1])
            else:
                ok = ok and val < RTOL_OK
        worst = max(worst, val)
    return ok, worst


def resolve_npz(name, timestamp):
    """Path to variant `name`'s .npz for the requested snapshot.
    "latest" picks the newest timestamped file for that variant (sorts
    chronologically since the timestamp format is zero-padded/lexical);
    otherwise the file must match "{name}__{timestamp}.npz" exactly."""
    candidates = sorted(RESULTS_DIR.glob(f"{name}__*.npz"))
    if not candidates:
        return None
    if timestamp == "latest":
        return candidates[-1]
    exact = RESULTS_DIR / f"{name}__{timestamp}.npz"
    return exact if exact.exists() else None


def render_variant(variant, timestamp):
    name = variant["name"]
    npz_path = resolve_npz(name, timestamp)
    if npz_path is None:
        return None
    data = np.load(npz_path)

    err_png = plot_error_evolution(name, data)
    gifs = [g for f in SNAPSHOT_FIELDS if (g := make_gif(name, data, f)) is not None]
    ok, worst_err = variant_status(name, data)

    generated_at = str(data["generated_at"]) if "generated_at" in data.files else None
    run_config = json.loads(str(data["run_config_json"])) if "run_config_json" in data.files else None

    return dict(
        name=name,
        family=str(data["family"]),
        group=str(data["group"]),
        overrides=json.loads(str(data["overrides_json"])),
        mini_ms=float(data["mini_sec_per_step"]) * 1000 if data["mini_sec_per_step"].shape == () else None,
        real_ms=float(data["real_sec_per_step"]) * 1000 if data["real_sec_per_step"].shape == () else None,
        ok=ok,
        worst_err=worst_err,
        err_png=err_png,
        gifs=gifs,
        generated_at=generated_at,
        run_config=run_config,
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


def timestamp_summary(rows, requested):
    """First-line banner naming which snapshot(s) the report reflects: the
    single shared timestamp if every variant resolved to the same one, or
    a per-timestamp breakdown when "latest" pulled variants from different
    runs (e.g. after a partial `generate_matrix_data.py --variant X`)."""
    known = [r for r in rows if r["generated_at"]]
    if not known:
        return f"timestamp: {requested} (no generation timestamp recorded for these results)."

    timestamps = {r["generated_at"] for r in known}
    if len(timestamps) == 1:
        ts = next(iter(timestamps))
        return f"timestamp: {requested} -> `{ts}` ({format_ts(ts)})."

    by_ts = {}
    for r in known:
        by_ts.setdefault(r["generated_at"], []).append(r["name"])
    breakdown = "; ".join(f"`{ts}` ({format_ts(ts)}): {', '.join(sorted(vs))}" for ts, vs in sorted(by_ts.items()))
    return f"timestamp: {requested} -> mixed snapshots -- {breakdown}."


def write_report(rows, timing_png, requested_timestamp):
    rows = sorted(rows, key=lambda r: (r["group"], r["name"]))
    lines = [
        "# mini_veros vs veros: comparison matrix report",
        "",
        timestamp_summary(rows, requested_timestamp),
        "",
        f"{sum(r['ok'] for r in rows)}/{len(rows)} variants within tolerance "
        f"(rel error < {RTOL_OK:.0e}, psi scale-normalized < {PSI_SCALE_TOL:.0e}).",
        "",
    ]
    if timing_png:
        lines += [f"![timing]({timing_png.relative_to(REPORT_DIR)})", ""]

    lines += ["| variant | group | status | worst error | mini ms/step | veros ms/step | speedup |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        status = "ok" if r["ok"] else "FAIL"
        mini_ms = f"{r['mini_ms']:.2f}" if r["mini_ms"] is not None else "-"
        real_ms = f"{r['real_ms']:.2f}" if r["real_ms"] is not None else "-"
        speedup = f"{r['real_ms'] / r['mini_ms']:.1f}x" if r["mini_ms"] and r["real_ms"] else "-"
        lines.append(f"| {r['name']} | {r['group']} | {status} | {r['worst_err']:.2e} | {mini_ms} | {real_ms} | {speedup} |")

    lines += ["", "## per-variant detail", ""]
    for r in rows:
        lines.append(f"### {r['name']} ({'ok' if r['ok'] else 'FAIL'})")
        if r["overrides"]:
            lines.append(f"overrides: `{r['overrides']}`")
        if r["generated_at"]:
            cfg = r["run_config"]
            cfg_note = f", {cfg['n_steps']} steps @ interval {cfg['record_interval']}" if cfg else ""
            lines.append(f"generated: `{r['generated_at']}`{cfg_note}")
        lines.append("")
        lines.append(f"![errors]({r['err_png'].relative_to(REPORT_DIR)})")
        for g in r["gifs"]:
            lines.append(f"![{g.stem}]({g.relative_to(REPORT_DIR)})")
        lines.append("")

    out = REPORT_DIR / "matrix_report.md"
    out.write_text("\n".join(lines))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timestamp", default="latest",
                         help="results snapshot to render: 'latest' (default) takes the newest .npz per "
                              "variant, or an exact run timestamp like 20260828T143000Z")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        row = render_variant(variant, args.timestamp)
        if row is None:
            print(f"skip {variant['name']}: no {RESULTS_DIR / (variant['name'] + '__*.npz')} matching timestamp {args.timestamp!r}")
            continue
        rows.append(row)
        print(f"{row['name']}: {'ok' if row['ok'] else 'FAIL'}  worst_err={row['worst_err']:.2e}")

    timing_png = plot_timing_summary(rows)
    report = write_report(rows, timing_png, args.timestamp)
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
