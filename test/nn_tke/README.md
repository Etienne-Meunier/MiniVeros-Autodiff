# NN-parametrized TKE closure

A separate research thread from `test/paper_figures/` (`report/paper/gradients_and_limits.md`):
replacing the TKE closure's two scalar constants, `c_k` and `c_eps` (`mini_veros/core/tke.py`),
with neural networks of local state -- either as a state-dependent coefficient (keeping the
known physical scaling) or as a full replacement of the closure's functional form -- and asking
whether the gradient/calibration findings from the paper-figures work (the H=80-day horizon,
gradient spikes, the calibration cliff) still hold once the thing being trained is a NN with
hundreds to ~1500 parameters instead of two scalars.

**Write-up: `report/nn_tke/nn_closure_report.md`** -- full findings, including a real
methodological bug (raw-MSE pretraining of a heavy-tailed positive target silently fails) found
and fixed along the way.

## The seam (mini-veros core changes)

`Parameters` (`mini_veros/model.py`) gained four optional fields, all `None` by default:
`tke_ck_net`/`tke_ceps_net` (coefficient mode -- a net multiplies the existing physical form)
and `tke_kappaM_net`/`tke_diss_net` (full-function mode -- a net predicts `log(kappaM)` /
`log(dissipation rate)` directly, exponentiated in `mini_veros/core/tke.py`'s
`tke_closure_fields`, which is the one place that knows about all three modes; the rest of the
closure just consumes its two outputs). All four `None` -- every existing `test/paper_figures`
script -- is bit-identical to the original scalar closure (verified: `exp1_gradient_check.py
--lengths 10` reproduces the exact same loss and AD gradient as before this change).

## Experiments

| | experiment | what it checks |
|---|---|---|
| 10a | `exp10a_identity_check.py` | wiring sanity: a coefficient-mode net initialized to reproduce the constant closure exactly (zero weights, bias = constant) does, to float64 noise |
| 10b | `exp10b_gradient_check.py` | the gate: AD-vs-FD gradient check for a ~700-weight coefficient-mode closure, random directions in weight space, exp1's own method generalized -- **H_nn = 80 days, matches the scalar closure exactly** |
| 10c | `exp10c_offline_pretrain.py` | supervised pretraining of the full-function nets against the analytic closure's own output -- fit in log space, not raw (see report: a raw-MSE fit silently collapses to a near-constant closure) |
| 10d | `exp10d_online_finetune.py` | the centerpiece: BPTT fine-tuning of the pretrained full-function nets against exp3's own twin-experiment loss -- 3.8x loss reduction, clean training curve |
| 10d2 | `exp10d2_regularized_finetune.py` | exp10d with an L2 anchor to the pretrained weights -- turns out to buy nothing once pretraining itself is fixed |
| 10e | `exp10e_calibration_limit.py` | exp4's own question (does the fit still "succeed" past the horizon), swept over rollout length 10-160 days |

Figures: `fig10b_gradient_check.py`, `fig10d_online_finetune.py`, `fig10e_calibration_limit.py`,
`fig10f_closure_profile.py` (the analytic-vs-pretrained-vs-trained depth profile comparison that
caught the log-space bug).

## Shared code

`common.py` reuses `test/paper_figures/common.py`'s spin-up cache and rollout/loss helpers
directly (same model, same settled ocean state -- no reason to recompute it), but keeps its
own `results/nn_tke` and `report/nn_tke/figures` output directories, since this is a distinct
line of experiments from the paper figures. `nn_utils.py` holds `NormalizedMLP` (a fixed,
non-trained feature rescaling wrapped around an `eqx.nn.MLP` -- `tke_closure_fields` calls
whatever net is installed with raw `[Nsqr, sqrttke, mxl, depth]` features, so normalization has
to travel with the net itself) and `build_full_function_net`, used by 10c/10d/10d2/10e/10f.
