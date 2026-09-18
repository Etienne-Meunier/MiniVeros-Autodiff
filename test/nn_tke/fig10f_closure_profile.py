"""
Figure 10f -- what did the trained closure actually learn? kappaM/dissipation depth profiles.

Evaluates the analytic closure, exp10d's plain BPTT fine-tune, and exp10d2's
pretrained-anchored fine-tune, all on the same state (the settled spin-up), and compares the
resulting fields' horizontal-mean profile by depth -- the loss curves (fig10d) show training
worked; this shows *what* it converged to, and whether anchoring to the offline pretrain
(exp10d2) keeps the closure physically sensible where exp10d's plain fine-tune drifted.
exp10d2 is optional (only plotted if it has been run).

    python test/nn_tke/fig10f_closure_profile.py
"""

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import common  # nn_tke's own -- see fig10b_gradient_check.py for why this import order matters

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_figures"))

import style
from nn_utils import build_full_function_net
from mini_veros.core.tke import set_tke_diffusivities
from mini_veros.state import DiagnosticState, PrognosticState, Tendencies
from mini_veros.core.thermodynamics import calc_eq_of_state


def closure_profile(model, state0):
    """(kappaM, diss_rate) horizontal-ocean-mean profile by depth, for whatever closure is
    installed in model.parameters."""
    eos = calc_eq_of_state(model, state0.state.temp, state0.state.salt)
    zero_dSp = Tendencies(K_diss_v=jnp.zeros_like(state0.state.tke))
    diag = set_tke_diffusivities(model, PrognosticState(tke=state0.state.tke), DiagnosticState(Nsqr=eos.Nsqr), zero_dSp)

    ocean = common.interior(model.boundary_conditions.maskW) != 0
    kappaM, diss = common.interior(diag.kappaM), common.interior(diag.tke_diss_rate)
    n_ocean = np.maximum(ocean.sum(axis=(0, 1)), 1)
    kappaM_profile = np.where(ocean, kappaM, 0).sum(axis=(0, 1)) / n_ocean
    diss_profile = np.where(ocean, diss, 0).sum(axis=(0, 1)) / n_ocean
    return np.asarray(kappaM_profile), np.asarray(diss_profile)


def main():
    style.use()
    model, state0, forcing_fn = common.spinup()
    z = np.asarray(model.grid.zt)

    kappaM_analytic, diss_analytic = closure_profile(model, state0)

    def load_and_profile(tag, key0, key1):
        kM = common.DATA_DIR / f"{tag}_kappaM_net.eqx"
        kD = common.DATA_DIR / f"{tag}_diss_net.eqx"
        if not (kM.exists() and kD.exists()):
            return None
        kappaM_net = eqx.tree_deserialise_leaves(kM, build_full_function_net(jax.random.PRNGKey(key0)))
        diss_net = eqx.tree_deserialise_leaves(kD, build_full_function_net(jax.random.PRNGKey(key1)))
        model_trained = eqx.tree_at(
            lambda m: (m.parameters.tke_kappaM_net, m.parameters.tke_diss_net), model, (kappaM_net, diss_net),
            is_leaf=lambda x: x is None,
        )
        return closure_profile(model_trained, state0)

    pretrained = load_and_profile("exp10c", 0, 1)
    plain = load_and_profile("exp10d", 0, 1)
    regularized = load_and_profile("exp10d2", 0, 1)

    series = [("analytic (c_k * mxl * sqrttke)", "analytic (c_eps * sqrttke / mxl)", kappaM_analytic, diss_analytic, style.SERIES[0])]
    if pretrained:
        series.append(("offline-pretrained (exp10c, before any BPTT)", "offline-pretrained (exp10c, before any BPTT)",
                        pretrained[0], pretrained[1], style.SERIES[3]))
    if plain:
        series.append(("BPTT fine-tune (exp10d, unregularized)", "BPTT fine-tune (exp10d, unregularized)",
                        plain[0], plain[1], style.SERIES[1]))
    if regularized:
        series.append(("BPTT fine-tune (exp10d2, anchored to pretrain)", "BPTT fine-tune (exp10d2, anchored to pretrain)",
                        regularized[0], regularized[1], style.SERIES[2]))

    fig, (ax_k, ax_d) = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
    for label_k, _, kM, _, color in series:
        ax_k.plot(kM, z, color=color, marker="o", ms=3, label=label_k)
    ax_k.set_xscale("log")
    ax_k.set_xlabel(r"$\kappa_M$ (ocean-mean, m$^2$/s)")
    ax_k.set_ylabel("depth (m)")
    ax_k.set_title("(a) vertical viscosity")
    ax_k.legend(fontsize=7)

    for _, label_d, _, diss, color in series:
        ax_d.plot(diss, z, color=color, marker="o", ms=3, label=label_d)
    ax_d.set_xscale("log")
    ax_d.set_xlabel("dissipation rate (ocean-mean, 1/s)")
    ax_d.set_title("(b) TKE dissipation rate")
    ax_d.legend(fontsize=7)

    fig.suptitle("What the trained closure converged to, vs the analytic one (spin-up state)",
                  fontsize=10.5, color=style.INK)
    style.save(fig, common.FIG_DIR / "fig10f_closure_profile")


if __name__ == "__main__":
    main()
