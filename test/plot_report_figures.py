#!/usr/bin/env python3
"""
Renders PNG figures from $STORE/MiniVeros-Autodiff/results/*.npz (produced
by generate_report_data.py) for the trust report: error-evolution curves
and mini/real/diff snapshot heatmaps. Writes into
$STORE/MiniVeros-Autodiff/results/figures/.

Usage:
    python test/plot_report_figures.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
RESULTS_DIR = STORE / "MiniVeros-Autodiff" / "results"
FIG_DIR = RESULTS_DIR / "figures"

SETUPS = ["acc_basic", "acc", "global_4deg"]
FIELDS = ["u", "v", "temp", "salt", "psi", "tke", "eke"]


def plot_error_evolution(setup_name, data):
    # psi has its own physical scale (O(1e6-1e8)) and an arbitrary per-island
    # gauge -- plotting its raw absolute error on the same axis as u/v/temp
    # (O(1e-8)) is meaningless (it dwarfs everything else on the chart even
    # though it's negligible relative to psi's own scale). Give it a
    # separate scale-normalized panel instead.
    timesteps = data["timesteps"]
    fields = [f for f in FIELDS if f"err_{f}_max_rel_errors" in data and f != "psi"]
    has_psi = "err_psi_max_abs_errors" in data

    fig, axes = plt.subplots(1, 3 if has_psi else 2, figsize=(16 if has_psi else 11, 4.2))
    for field in fields:
        max_abs = data[f"err_{field}_max_abs_errors"]
        mean_abs = data[f"err_{field}_mean_abs_errors"]
        axes[0].plot(timesteps, max_abs + 1e-300, marker="o", ms=3, label=field)
        axes[1].plot(timesteps, mean_abs + 1e-300, marker="o", ms=3, label=field)

    for ax, title in zip(axes[:2], ["max |mini - real|", "mean |mini - real|"]):
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e-2)
        ax.set_xlabel("step")
        ax.set_ylabel("absolute error")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    if has_psi:
        # normalize per-timestep by that timestep's own field magnitude isn't
        # recorded, so use the final-step field's scale as a fixed reference
        # (psi's magnitude doesn't change fast enough over these horizons for
        # this to matter -- it's O(1e6-1e8) throughout).
        scale = np.max(np.abs(data["psi_real"])) if "psi_real" in data else 1.0
        scale = scale if scale > 0 else 1.0
        normalized = data["err_psi_max_abs_errors"] / scale
        axes[2].plot(timesteps, normalized + 1e-300, marker="o", ms=3, color="tab:purple")
        axes[2].set_yscale("log")
        axes[2].set_ylim(1e-12, 1e-2)
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("max|mini-real| / max|real|")
        axes[2].set_title("psi (scale-normalized)")
        axes[2].grid(alpha=0.3)

    fig.suptitle(f"{setup_name}: mini_veros vs veros error evolution")
    fig.tight_layout()
    out = FIG_DIR / f"{setup_name}_error_evolution.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_snapshots(setup_name, data):
    outs = []
    for field in ("psi", "temp", "u", "salt"):
        key_mini, key_real = f"{field}_mini", f"{field}_real"
        if key_mini not in data:
            continue
        mini_arr = data[key_mini]
        real_arr = data[key_real]

        if mini_arr.ndim == 3:
            # 3D field: take the uppermost model level as a 2D "surface" slice
            mini_2d = mini_arr[:, :, -1]
            real_2d = real_arr[:, :, -1]
            level_note = " (uppermost level)"
        else:
            mini_2d = mini_arr
            real_2d = real_arr
            level_note = ""

        diff_2d = mini_2d - real_2d

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        vmax = np.nanmax(np.abs(real_2d))
        vmax = vmax if vmax > 0 else 1.0
        im0 = axes[0].imshow(mini_2d.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0].set_title(f"mini_veros: {field}{level_note}")
        im1 = axes[1].imshow(real_2d.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"veros: {field}{level_note}")
        dmax = np.nanmax(np.abs(diff_2d))
        dmax = dmax if dmax > 0 else 1e-30
        im2 = axes[2].imshow(diff_2d.T, origin="lower", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[2].set_title(f"diff (max|diff|={dmax:.2e})")
        for ax, im in zip(axes, (im0, im1, im2)):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{setup_name}: {field} at final step")
        fig.tight_layout()
        out = FIG_DIR / f"{setup_name}_{field}_snapshot.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        outs.append(out)
    return outs


def plot_pressure_evolution():
    path = RESULTS_DIR / "pressure_acc_basic.npz"
    if not path.exists():
        return None
    data = np.load(path)
    timesteps = data["timesteps"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for field in ("u", "v", "temp", "salt", "tke"):
        key = f"err_{field}_max_abs_errors"
        if key in data:
            ax.plot(timesteps, data[key] + 1e-300, marker="o", ms=3, label=field)

    psi_key = "err_psi_max_abs_errors"
    if psi_key in data:
        scale = float(data["psi_real_scale"]) or 1.0
        ax.plot(timesteps, data[psi_key] / scale + 1e-300, marker="o", ms=3, label="psi (scale-norm.)", color="tab:purple")

    ax.set_yscale("log")
    ax.set_ylim(1e-13, 1e0)
    ax.set_xlabel("step")
    ax.set_ylabel("max absolute error")
    ax.set_title("acc_basic, enable_streamfunction=False: mini_veros vs veros")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "pressure_acc_basic_error_evolution.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = plot_pressure_evolution()
    if p:
        print("pressure solver ->", p)
    for setup_name in SETUPS:
        npz_path = RESULTS_DIR / f"{setup_name}.npz"
        if not npz_path.exists():
            print(f"skip {setup_name}: no {npz_path}")
            continue
        data = np.load(npz_path)
        p1 = plot_error_evolution(setup_name, data)
        p2 = plot_snapshots(setup_name, data)
        print(setup_name, "->", p1, *p2)


if __name__ == "__main__":
    main()
