"""
Shared pieces for the NN-parametrized TKE closure experiments.

Reuses test/paper_figures's spin-up cache directly (same model, same 30-year settled state --
no reason to recompute it) but keeps its own results/figures directories, since this is a
separate research thread from report/paper/gradients_and_limits.md.
"""

import importlib.util
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Loaded by explicit file path (not `import common`) since this module is *also* named
# common.py -- a plain `import common` from here would just return this half-initialized
# module back out of sys.modules instead of paper_figures' one.
_spec = importlib.util.spec_from_file_location("paper_figures_common", REPO / "test" / "paper_figures" / "common.py")
paper_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper_common)

STORE = Path(os.environ.get("STORE", Path.home() / "STORE"))
DATA_DIR = STORE / "MiniVeros-Autodiff" / "results" / "nn_tke"
FIG_DIR = REPO / "report" / "nn_tke" / "figures"

spinup = paper_common.spinup
rollout = paper_common.rollout
with_params = paper_common.with_params
interior = paper_common.interior
sst = paper_common.sst
mean_square_sst = paper_common.mean_square_sst
upper_ts = paper_common.upper_ts
upper_ts_scale = paper_common.upper_ts_scale
upper_ts_misfit = paper_common.upper_ts_misfit
grid_arrays = paper_common.grid_arrays
HORIZON_DAYS = paper_common.HORIZON_DAYS


def save(name: str, **arrays):
    import numpy as np
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.npz"
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print(f"wrote {path}")


def load(name: str):
    import numpy as np
    path = DATA_DIR / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the matching expN*.py first")
    return np.load(path)
