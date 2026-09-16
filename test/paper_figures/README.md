# Paper figures

The experiments behind `report/paper/gradients_and_limits.md`. Every one of
them runs `global_4deg` (90x40x15, real bathymetry and climatological
forcing, `dt_tracer` = 86400 s, so one step is one day and 365 steps is a
model year), after a 30-model-year spin-up. That spin-up is computed once and
cached (`common.spinup`); every experiment starts from it.

Each experiment is two scripts: `expN_*.py` runs the simulation(s) and writes
`DATA_DIR/expN_*.npz` (`DATA_DIR` = `$STORE/MiniVeros-Autodiff/results/paper`,
`STORE` defaulting to `~/STORE`), `figN_*.py` reads that file and writes
`report/paper/figures/figN_*.{pdf,png}`. Re-plotting never re-runs a
simulation.

| | experiment | figure | what it shows |
|---|---|---|---|
| 1 | `exp1_gradient_check.py` | `fig1_gradient_check.py` | reverse-mode, forward-mode and finite-difference gradients over growing rollouts, with a round-off ensemble standing in for the finite difference's own error bar |
| 1b | `exp1b_field_jvp_vjp.py` | `fig1b_field_jvp_vjp.py` | (a) the same reverse-vs-finite-difference check with the loss built from a whole field (temperature, salinity) instead of just SST; (b) forward mode vs reverse mode directly, no finite difference, over random directions in parameter space |
| 2 | `exp2_sensitivity_map.py` | `fig2_sensitivity_map.py` | `dT/dp` for four namelist quantities, propagated in forward mode, as snapshots |
| 2b | `exp2b_local_perturbation.py` | `fig2b_local_perturbation.py` | how a unit perturbation to one grid point of temperature spreads outward, snapshot by snapshot, over a short rollout |
| 3 | `exp3_calibration.py` | `fig3_calibration.py` | fitting `(c_k, c_eps)` to T,S on the top 3 layers, with the loss landscape and the descent path drawn on it |
| 3b | `exp3b_calibration_distance.py` | `fig3b_calibration_distance.py` | the same fit (c_k alone) swept over both rollout length and how wrong the starting guess is, matrix+colour, with repeats for error bars |
| 3c | `exp3c_calibration_joint.py` | `fig3c_calibration_joint.py` | fitting c_k AND c_eps jointly, starting guess placed on a circle (log-space L2 distance from truth) at 3 evenly-spread angles -- whether the direction of the initial error matters, not just its size |
| 4 | `exp4_calibration_limit.py` | `fig4_calibration_limit.py` | the same fit over rollouts of growing length, until it stops recovering the truth |
| 5 | `exp5_assimilation.py` | `fig5_assimilation.py` | recovering the initial temperature field from T,S on the top 3 layers -- once where that's identifiable, once where it isn't |
| 5b | `exp5b_assimilation_horizon.py` | `fig5b_assimilation_horizon.py` | same recovery question as figure 5, swept over rollout length (10-320 days) and observation metric (SST alone to the whole column): where it actually succeeds vs where it doesn't |
| 6 | `exp6_gradient_spikes.py` | `fig6_gradient_spikes.py` | how often the gradient spikes inside the trustworthy window, where the spike lives in the adjoint, and which model switch it traces to |
| 7 | `exp7_checkpoint_cost.py` | `fig7_checkpoint_cost.py` | time and memory of one gradient against the checkpoint block size, inside and past the horizon |
| 8 | `exp8_ensemble_gradient.py` | `fig8_ensemble_gradient.py` | whether averaging over an ensemble recovers a usable gradient past the horizon |
| 9 | `exp9_loss_choice.py` | `fig9_loss_choice.py` | whether a different choice of loss buys a longer horizon |
| 9b | `exp9b_loss_chaos.py` | `fig9b_loss_chaos.py` | whether time/space averaging quiets the finite difference's own round-off spread without moving the reverse-vs-finite-difference horizon |

Run them in order -- exp1 has to go first (everything else sizes its rollout
off the horizon `H` it measures, via `common.HORIZON_DAYS`), the rest don't
depend on each other:

```
python test/paper_figures/exp1_gradient_check.py     &&  python test/paper_figures/fig1_gradient_check.py
# read exp1's printed horizon, copy it into common.HORIZON_DAYS by hand, then:
python test/paper_figures/exp1b_field_jvp_vjp.py     &&  python test/paper_figures/fig1b_field_jvp_vjp.py
python test/paper_figures/exp2_sensitivity_map.py    &&  python test/paper_figures/fig2_sensitivity_map.py
python test/paper_figures/exp2b_local_perturbation.py    &&  python test/paper_figures/fig2b_local_perturbation.py
python test/paper_figures/exp3_calibration.py        &&  python test/paper_figures/fig3_calibration.py
python test/paper_figures/exp3b_calibration_distance.py  &&  python test/paper_figures/fig3b_calibration_distance.py
python test/paper_figures/exp3c_calibration_joint.py     &&  python test/paper_figures/fig3c_calibration_joint.py
python test/paper_figures/exp4_calibration_limit.py  &&  python test/paper_figures/fig4_calibration_limit.py
python test/paper_figures/exp5_assimilation.py       &&  python test/paper_figures/fig5_assimilation.py
python test/paper_figures/exp5b_assimilation_horizon.py  &&  python test/paper_figures/fig5b_assimilation_horizon.py
python test/paper_figures/exp6_gradient_spikes.py    &&  python test/paper_figures/fig6_gradient_spikes.py
python test/paper_figures/exp7_checkpoint_cost.py    &&  python test/paper_figures/fig7_checkpoint_cost.py
python test/paper_figures/exp8_ensemble_gradient.py  &&  python test/paper_figures/fig8_ensemble_gradient.py
python test/paper_figures/exp9_loss_choice.py        &&  python test/paper_figures/fig9_loss_choice.py
python test/paper_figures/exp9b_loss_chaos.py        &&  python test/paper_figures/fig9b_loss_chaos.py
```

Every experiment takes `--` flags to override its lengths/iterations (see each
script's own `--help`); the defaults are what produced the figures and
runtimes below. Physical choices -- which parameters, the true and starting
values, the perturbation -- are module-level constants at the top of each
`expN_*.py`, meant to be edited.

`exp1_gradient_check.py`'s full `LENGTHS` sweep, run in one process on a
memory-constrained laptop (shared with other work), has been observed to get
OOM-killed partway through the largest lengths -- XLA does not release
compiled executables between the very different rollout lengths, and RSS
climbs across the whole 16-length loop. If that happens, running one length
per subprocess and merging the resulting `.npz` files (each subprocess exits
and the OS reclaims everything before the next one starts) works around it;
`--lengths <n>` on the CLI is what makes that possible.

## Running on the server

`test/sweep/run_paper_oar.sh` (spin-up and exp1, which have to run first and
alone) and `test/sweep/paper.yaml` + `test/paper_figures/sweep_paper.py`
(exp2-exp9 and their b-variants, as a wandb sweep so `g5k agent <sweep>
--n-agents 8` runs several in parallel) mirror the matrix runs' own launch
path (`guide_to_run.md`). exp1b doesn't need `common.HORIZON_DAYS` (it sizes
its own rollout the way exp1 does), so it runs as part of the sweep alongside
exp2-9 rather than needing to go before them. In order:

```
g5k sync code
# spin-up alone first -- parallel agents would all race to build the cache
oarsub -l gpu=1,walltime=1:00:00 -p "gpu_count > 0" \
  "bash ~/code/MiniVeros-Autodiff/test/sweep/run_paper_oar.sh spinup_only"
# exp1 alone next, since H has to be known before exp2-9 can size themselves
oarsub -l gpu=1,walltime=6:00:00 -p "gpu_count > 0" \
  "bash ~/code/MiniVeros-Autodiff/test/sweep/run_paper_oar.sh exp1_gradient_check"
# pull exp1's result back, set common.HORIZON_DAYS, g5k sync code again, then:
wandb sweep test/sweep/paper.yaml
g5k agent <entity>/<project>/<sweep_id> --site grenoble --n-agents 8 --walltime 3:00:00 --no-besteffort
g5k sync model   # or a targeted scp of results/paper/*.npz
```

`sweep_paper.py` resets `sys.argv` before handing off to each `expN_*.py` --
`wandb agent` invokes it as `sweep_paper.py --device=gpu --experiment=...`,
and without that reset those flags leak into the target script's own
argparse and it refuses to start.

## Shared code

- `common.py` -- the cached spin-up (`spinup`, with optional
  `model.apply_overrides`-style overrides for ablations, each cached
  separately), the checkpointed rollout everything differentiates through
  (`rollout`, with `safe_checkpoint_every` picking a working block size for
  any `n_steps`), the T,S-on-top-layers observation operator the calibration
  and assimilation experiments share (`upper_ts`, `upper_ts_scale`,
  `upper_ts_misfit`), and the `.npz` read/write paths.
- `style.py` -- the palette, the colormaps, and the two drawing helpers the
  figures share.
- `gpu_smoke_test.py` -- not one of the nine experiments; a short check that
  a GPU node has jax's CUDA backend wired up and that its gradient matches a
  finite difference, meant to be run once before trusting a longer job on new
  hardware.

Nothing here modifies `mini-veros`: the experiments only ever replace entries
of `model.parameters` / `model.config`, fields of the initial state, or wrap
`forcing_fn`.

## Provenance

exp1 (and the spin-up it starts from) ran on a laptop CPU. exp2-exp9 ran on
Grid5000 (an NVIDIA L4), starting from that same spin-up file copied over
verbatim -- not recomputed on the GPU, so every later experiment shares
exp1's exact climate state and its horizon `H = 80` days applies to all of
them unchanged. Two experiments (exp3, exp4) happened to run once on each
machine before that decision was made; the two runs agree to 3-4 significant
digits (exp3's fitted `(c_k, c_eps)`: `(0.1039, 0.7341)` CPU vs
`(0.104, 0.739)` GPU) -- a useful small data point that the qualitative
conclusions don't depend on which machine computed them, even though the
exact digits would (a GPU-recomputed spin-up would very likely shift `H`
itself by a step or two, from float64 summation order alone).

## Notes on the results

- **The gradient is trustworthy out to 80 days and worthless past 160.**
  Figure 1: autodiff matches central finite differences to better than 0.6%
  relative error through 80 days, is 5.6x too large at 160, and reaches
  5e14x by 2560 days -- `H = 80` days is what every later experiment sizes
  its rollout against. The round-off ensemble (panel c) tells a distinct,
  later story: finite differences don't themselves become round-off-noisy
  until past ~1000 days, so the disagreement that starts at 160 days is a
  real autodiff pathology at a length where the reference is still solid, not
  two noisy numbers being compared.
- **Even inside the window the gradient occasionally spikes.** Figure 6, 50
  members 7 days apart: 0/50 spike at 40 days, 5/50 (10%) at 80 days, one of
  them 43.6x too large with the wrong sign. The adjoint norm carried backward
  through a manual per-step reverse sweep (panel b) is dominated by TKE --
  several orders of magnitude above every other prognostic field, for both a
  spiking and a normal member -- pointing at the TKE closure as where the
  amplification concentrates, and the peak-location map (panel d) shows it is
  spatially localised, not basin-wide. The obvious next hypothesis --
  switching off the flux-limiter advection schemes -- is only a mild effect
  (spike fraction 0.5 -> 0.4 with either superbee limiter off, on a
  10-member sample) and switching off the EKE closure entirely makes it
  *worse* (0.5 -> 0.8): the closure looks like it's damping the spikes,
  not causing them. n=10 per ablation is small; take the direction, not the
  digits.
- **`(c_k, c_eps)` is identifiable from T,S on the top 3 layers alone, at the
  final step.** No SST time series needed: figure 3 recovers `c_k` to 4% and
  `c_eps` to 6% (`0.104, 0.734` vs true `0.10, 0.70`) from a single
  40-day-rollout snapshot, and the loss landscape shows no sign of the old
  `c_k^4 / c_eps = const` valley -- adding salinity and two more layers is
  enough to break that degeneracy without needing to sample along the
  rollout at all.
- **The calibration limit is a cliff, sitting exactly at the horizon.**
  Figure 4: fitting `c_k` alone recovers it to <0.1% through 40 days, crosses
  10% error at 80 days (`H` itself), and is *worse than guessing*
  (>100% relative error) by 320 days -- while the loss the optimiser reaches
  keeps *decreasing* the whole time. The fit succeeding and the parameter
  being wrong is the failure mode the figure exists to show.
- **Assimilation from top-layer T,S has a real, physical null space below
  those layers.** Figure 5(a): a perturbation confined to the observed
  layers drives the observed loss down four orders of magnitude, but the
  whole-volume state error only falls to about 60% of its start rather than
  vanishing -- the observation operator has weak but nonzero sensitivity to
  deep temperature even over a 40-day rollout, so even the "identifiable"
  case isn't perfectly clean. Figure 5(b) makes the null space itself
  visible: four different perturbations confined *below* the observed
  layers all drive the same loss down to the same floor, while the four
  recovered deep states diverge from the truth and from each other,
  growing from ~0 error in the observed layers to 0.15-0.25 K rms by
  3000-4000 m -- a real null space, not a slow fit.
- **No loss among six tested buys a longer horizon.** Figure 9: final-step
  MSE, whole-rollout and last-30-day time averages, global-mean SST squared,
  the calibration/assimilation T,S misfit, and top-layer heat content all
  share `H = 80` days. (An earlier version of this check built the T,S
  misfit's target from the same parameter value being differentiated, making
  the loss and its true gradient exactly zero at that point -- comparing AD
  to FD there was comparing two round-off floors, not a real check. Fixed by
  building the target from a twin rollout at `c_k * 1.2` instead; see
  `exp9_loss_choice.py`'s docstring.)
- **An ensemble does not rescue it.** Figure 8: 24 members, same climate,
  different trajectory. At 160 days the ensemble-mean autodiff gradient is
  about 470x the ensemble-mean finite difference (~160 vs ~0.34); the
  finite-difference mean itself stays within a factor of a few across all
  four lengths tested (40-320 days). Averaging removes member-to-member
  variance, not this.
- **Checkpointing's own exactness held at every schedule tested, including
  past the horizon.** Figure 7, on the GPU: the gradient from `checkpoint_every`
  1 through 32 agrees to 3-4 significant digits at both 40 and 320 days, and
  wall time is close to flat across block sizes at a given length. This
  updates rather than confirms the original hypothesis (that checkpointing
  schedules would themselves disagree once past the horizon) -- here they
  don't; what changes past the horizon is only the gradient's relationship to
  the physics (figure 1), not its reproducibility across schedules.
  **Caveat:** this run's memory panel uses host-process RSS
  (`resource.getrusage`), which is the right measurement on the CPU backend
  but mostly measures CPU-side bookkeeping on GPU, where the tape actually
  lives in device memory -- the memory-vs-block-size numbers in this GPU run
  are not meaningful and shouldn't be read as a real measurement of the
  trade-off `exp7_checkpoint_cost.py`'s own docstring describes; a CPU rerun
  (or a device-memory-aware measurement) would be needed for that half of
  the story.


## Interaction 

- Figure 1 is very good now, no need to change it that much, although it would be interestin to add forward jvp to it, for the right figure it's easy we just need to add a new line, for the left one we could add as dotted line
- Also I would like to have a Fig 1b that does the same kind of test for gradients over field. Basically we could compare directional derivative over several field (temperature, salinity ...) of the prognositic with different rollout. Similarly we could compare JVP and VJP with different random perturbation

- Fig2 on sensitivity is a right start, I would also like to have a Fig2b that show sentivity to local perturbation of a part of the field (how a perturbation spread around) - maybe or shorter rollout 


- Fig3 on calibration is good too. A Fig3b insteresting on limit on calibration could take more rollout points (now we have very little points in x ) and also have a figure that include the distance to the true parameters of the initial param of the optim. This would be great as it would show that calibration results depends on the distance to initialisation. We could then show the distance and loss with an error bar depending to the rollout and distance of the initial time. For now show them as a matrix + color like in assimilation but save the results so that we can try different visualization. 

- We have some creative work to do on the loss figure. The idea is that normally having average over time (or space actually) should reduce the cahoticity of the loss per say no ? Like it should give finite difference more smooth ? But there is good chance it doesn't change the stability of AD as anyway gradients goes through the fields, what do you think ? How can we show that best ? 