# Paper figures 



List of clean figures for the paper. 



### The configuration 



### The figures 

|      | experiment               | figure                   | what it shows                                                |
| ---- | ------------------------ | ------------------------ | ------------------------------------------------------------ |
| 0    | `exp0_spinup_snapshot.py` | `fig0_spinup_snapshot.py` | Snapshot of the spun-up global_4deg state (cartopy Robinson maps): mixed-layer depth (de Boyer Montegut criterion) and vertically integrated kinetic energy, sum_z (u^2+v^2) dz, with u/v interpolated to the T grid first |
| 1    | `exp1_gradient_check.py` | `fig1_gradient_check.py` | reverse-mode, forward-mode and finite-difference gradients over growing rollouts, with a round-off ensemble standing in for the finite difference's own error bar |
| 2.   | `exp1_gradient_check.py` (reused, no new sim) | `fig1c_variance_stripes.py` | Round-off ensemble std, by method and parameter, over rollout length: 3 parameter groups (c_k, c_eps, A_h) each stacked as 3 colour stripes (reverse mode, forward mode, finite difference), columns are the measured rollout lengths in days, one colourbar per parameter at the right |