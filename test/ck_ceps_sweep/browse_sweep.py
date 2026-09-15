import sys
from pathlib import Path

import numpy as np

from __init__ import PRP
from test import tutorials
sys.path.append(PRP + 'test/tutorials')

PRP
from utils_viz import browse

import common

# Load the sweep results
d = np.load(common.SWEEP_NPZ)
zt, k_grid, eps_grid = d["zt"], d["k_grid"], d["eps_grid"]
profiles = d["profiles"]            # (n_k, n_eps, n_logs, nz) -- full 30-day-spaced history
final_profile = d["final_profile"]  # (n_k, n_eps, nz) -- last-year average
diverged = d["diverged"]            # (n_k, n_eps) bool

mean_profile = np.nanmean(np.where(~diverged[:, :, None], final_profile, np.nan), axis=(0, 1))
anomaly = final_profile - mean_profile  # (n_k, n_eps, nz)

final_profile.shape

# Browse: sliders let you scrub over (c_k, c_eps) and/or time
browse(final_profile)  # sliders over c_k, c_eps -- last axis is depth
browse(profiles)       # sliders over c_k, c_eps, time -- last axis is depth
browse(anomaly)        # same as final_profile, minus the grid-mean at each depth

# k_grid[i], eps_grid[j] give the parameter values behind slider index (i, j)

ls {PRP}/test/tutorials
