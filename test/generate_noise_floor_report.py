#!/usr/bin/env python3
"""
Renders a standalone noise-floor report from
$STORE/MiniVeros-Autodiff/results/noise_floor_*.npz (produced by
measure_noise_floor.py): how much real veros differs from itself across
two independent runs of the same variant. Answers TODO.md's "establish the
actual noise floor first" -- the baseline test_matrix.py's atol/rtol should
be judged against, instead of an assumed number.

Kept separate from report.md (the mini_veros-vs-veros matrix report) since
this is about real veros alone, not the mini/real comparison.

Usage:
    python test/generate_noise_floor_report.py
"""

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
BASE_DIR = STORE / "MiniVeros-Autodiff"
RESULTS_DIR = BASE_DIR / "results"

# same tolerance policy test_matrix.py gates mini vs real on -- quoted here
# only for contrast against the measured noise floor, not applied to it.
RTOL_OK = 1e-6


def noise_floor_rows():
    rows = []
    for npz_path in sorted(RESULTS_DIR.glob("noise_floor_*.npz")):
        data = np.load(npz_path)
        fields = sorted(f.removeprefix("err_").removesuffix("_max_abs_errors") for f in data.files if f.endswith("_max_abs_errors"))
        per_field_worst = {f: float(np.max(data[f"err_{f}_max_abs_errors"])) for f in fields}
        worst = max(per_field_worst.values(), default=0.0)
        rows.append(dict(
            variant=str(data["variant"]) if "variant" in data.files else npz_path.stem.removeprefix("noise_floor_"),
            steps=int(data["timesteps"][-1]),
            worst_abs=worst,
            per_field_worst=per_field_worst,
        ))
    return rows


def write_report(rows):
    lines = [
        "# real veros: run-to-run noise floor",
        "",
        "Same variant (setups_matrix.py) run twice as two independent "
        "processes -- fresh interpreter each time, rules out shared "
        "JIT-cache/state artifacts -- real veros only, no mini_veros. Diffed "
        "with the same error-evolution machinery test_matrix.py uses for "
        "mini vs real.",
        "",
    ]

    if not rows:
        lines.append(f"No noise_floor_*.npz found in {RESULTS_DIR}. Run test/measure_noise_floor.py first.")
        out = BASE_DIR / "noise_floor_report.md"
        out.write_text("\n".join(lines) + "\n")
        return out

    worst_overall = max(r["worst_abs"] for r in rows)
    lines += [
        f"Measured across {len(rows)} variant(s): max abs diff (run A vs run B, "
        f"any field/step) = **{worst_overall:.2e}**.",
        "",
        f"For contrast, test_matrix.py gates mini vs real at rel error < {RTOL_OK:.0e}. "
        "If the noise floor above is ~0 (float64 eps), that gate is not run-to-run "
        "solver noise headroom -- it's an assumed margin. Any nonzero mini/real diff "
        "above the noise floor is real divergence, not noise.",
        "",
        "| variant | steps | max abs diff (run A vs run B, any field/step) |",
        "|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["variant"]):
        lines.append(f"| {r['variant']} | {r['steps']} | {r['worst_abs']:.2e} |")

    lines += ["", "## per-field detail", ""]
    for r in sorted(rows, key=lambda r: r["variant"]):
        lines.append(f"### {r['variant']} ({r['steps']} steps)")
        lines.append("")
        lines.append("| field | max abs diff |")
        lines.append("|---|---|")
        for field, val in sorted(r["per_field_worst"].items()):
            lines.append(f"| {field} | {val:.2e} |")
        lines.append("")

    out = BASE_DIR / "noise_floor_report.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main():
    rows = noise_floor_rows()
    for r in rows:
        print(f"{r['variant']}: {r['steps']} steps, worst_abs={r['worst_abs']:.2e}")
    report = write_report(rows)
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
