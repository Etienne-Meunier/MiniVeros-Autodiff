"""
Experiment 10c -- offline-pretrain the full-function closure to match the analytic one.

Full-function mode (tke_kappaM_net / tke_diss_net) has no "start exactly at the known physics"
trick the way coefficient mode does (exp10a): the net's output doesn't factor into a constant
times a fixed physical form, so there is no zero-weight init that reproduces `c_k * mxl *
sqrttke` for every input -- that product isn't representable by a bias alone. Instead: collect
(feature, target) pairs from a short rollout under the *analytic* closure -- target = what the
analytic closure would have output at that state -- and fit the nets to it by ordinary
supervised regression. Cheap (no rollout differentiation, no PDE at all after data collection)
and gives training/exp10d a stable starting point instead of a randomly-initialized closure
dropped straight into a live ocean model.

Data: N_SNAPSHOTS states along a short (N_STEPS-day) rollout from the settled spin-up, so the
dataset spans some of the state variation an online fine-tune would see, not just one instant.
Nsqr/sqrttke/mxl are recomputed directly from each snapshot's (temp, salt, tke) via the same
functions the model itself uses (calc_eq_of_state, set_tke_diffusivities), with the closure
nets still None at this point -- so the *targets* are exactly the current physics.

    python test/nn_tke/exp10c_offline_pretrain.py

A few minutes on a laptop CPU (or much less on GPU): this is a plain regression, not BPTT.

Fit in log space, not raw. The first version of this fit raw kappaM/diss_rate with plain MSE
and reported good-looking loss (kappaM MSE ~1e-3 against a target range up to 2.4), but the
resulting net was nearly *constant* with depth -- nothing like the analytic closure's own
profile (see fig10f_closure_profile.py). The target distribution is the reason: kappaM and the
dissipation rate are strictly positive and span several orders of magnitude (the ocean-mean
profile itself is ~1e-3 to 6e-3, but individual points reach into the 1-2 range at rare intense-
mixing locations), so a handful of large-magnitude outliers dominate a raw squared-error loss,
and the net converges toward predicting something close to the tail's own scale everywhere
rather than tracking the bulk's fine depth structure. Regressing log(target) with plain MSE
(equivalently, log(net_output) = the model's own tke_kappaM_net/tke_diss_net convention --
see mini_veros.core.tke.tke_closure_fields, which exponentiates) turns that into a *relative*
error metric, which is what actually matters for a quantity like this.

Dissipation additionally has ~half its points at *exactly* zero (quiescent, TKE=0 regions) --
forcing those to an arbitrary log-floor target would bias the fit the same way the original
raw-MSE tail did, just in the opposite direction. Excluded from diss_net's fit entirely; the
net's own extrapolation (very negative log-output -> ~0 via exp) covers the quiescent regions.
"""

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

import common
from mini_veros import loop
from mini_veros.core.thermodynamics import calc_eq_of_state
from mini_veros.core.tke import set_tke_diffusivities
from mini_veros.state import DiagnosticState, PrognosticState, Tendencies
from nn_utils import build_full_function_net

N_STEPS = 40  # days -- well inside H, just needs to cover some real state variation
N_SNAPSHOTS = 8
N_EPOCHS = 500
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
SEED = 0


def collect_dataset(model, state0, forcing_fn, n_steps, n_snapshots):
    """(features, target_kappaM, target_diss), each (n_snapshots * n_ocean_points, ...) --
    ocean points only (land masked out), analytic-closure targets."""

    def block(integrator_state, _):
        def one_step(integrator_state, _):
            force = forcing_fn(model, integrator_state.state)
            return loop.step(model, integrator_state, force), None

        integrator_state, _ = lax.scan(one_step, integrator_state, length=n_steps // n_snapshots)
        return integrator_state, (integrator_state.state.temp, integrator_state.state.salt, integrator_state.state.tke)

    _, (temps, salts, tkes) = lax.scan(eqx.filter_jit(block), state0, length=n_snapshots)

    zero_dSp = Tendencies(K_diss_v=jnp.zeros_like(state0.state.tke))
    ocean = common.interior(model.boundary_conditions.maskW) != 0

    all_features, all_kappaM, all_diss = [], [], []
    for k in range(n_snapshots):
        eos = calc_eq_of_state(model, temps[k], salts[k])
        diag_in = DiagnosticState(Nsqr=eos.Nsqr)
        # analytic-closure targets: model.parameters.tke_{kappaM,diss}_net are still None here,
        # so tke_closure_fields (called internally) returns the constant-closure values, and
        # kappaM/tke_diss_rate below are exactly what the model would actually use.
        diag_out = set_tke_diffusivities(model, PrognosticState(tke=tkes[k]), diag_in, zero_dSp)
        depth = jnp.broadcast_to(model.grid.zt[None, None, :], eos.Nsqr.shape)
        features = jnp.stack([eos.Nsqr, diag_out.sqrttke, diag_out.mxl, depth], axis=-1)

        all_features.append(common.interior(features)[ocean])
        all_kappaM.append(common.interior(diag_out.kappaM)[ocean])
        all_diss.append(common.interior(diag_out.tke_diss_rate)[ocean])

    return (jnp.concatenate(all_features), jnp.concatenate(all_kappaM), jnp.concatenate(all_diss))


def fit(net, features, target, n_epochs, batch_size, lr, key):
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(net, eqx.is_array))
    n = features.shape[0]

    @eqx.filter_jit
    def loss_fn(net, batch_f, batch_t):
        pred = jax.vmap(net)(batch_f)[:, 0]
        return jnp.mean((pred - batch_t) ** 2)

    @eqx.filter_jit
    def step(net, opt_state, batch_f, batch_t):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(net, batch_f, batch_t)
        updates, opt_state = opt.update(grads, opt_state)
        net = eqx.apply_updates(net, updates)
        return net, opt_state, loss

    losses = []
    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n)
        batch_idx = perm[:batch_size]
        net, opt_state, loss = step(net, opt_state, features[batch_idx], target[batch_idx])
        losses.append(float(loss))
        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:4d}  mse={losses[-1]:.4e}")
    return net, np.array(losses)


LOG_FLOOR = 1e-10  # kappaM touching exactly 0 (never observed, but guard anyway) needs a floor before log
DISS_ZERO_THRESHOLD = 1e-9  # ~half of all points have *exactly* zero dissipation (quiescent,
                             # TKE=0 regions) -- forcing those to an arbitrary log-floor target
                             # would bias the fit the same way the original raw-MSE tail did,
                             # just in the opposite direction; exclude them instead and let the
                             # net's own extrapolation (very negative log-output -> ~0 via exp)
                             # cover the quiescent regions


def main(n_steps, n_snapshots, n_epochs, batch_size, lr):
    model, state0, forcing_fn = common.spinup()
    t0 = time.time()
    features, target_kappaM, target_diss = collect_dataset(model, state0, forcing_fn, n_steps, n_snapshots)
    print(f"dataset: {features.shape[0]} points ({time.time() - t0:.0f}s)")
    print(f"target kappaM range: [{float(target_kappaM.min()):.3e}, {float(target_kappaM.max()):.3e}]")
    print(f"target diss  range: [{float(target_diss.min()):.3e}, {float(target_diss.max()):.3e}]")

    # log space, not raw -- see this file's own docstring for why (a raw-MSE fit converges to
    # a near-constant net, dominated by rare large-magnitude points)
    log_target_kappaM = jnp.log(jnp.maximum(target_kappaM, LOG_FLOOR))

    diss_active = target_diss > DISS_ZERO_THRESHOLD
    frac_zero = float((~diss_active).mean())
    print(f"dissipation: {frac_zero:.1%} of points are exactly zero (quiescent) -- excluded from diss_net's fit")
    features_diss = features[diss_active]
    log_target_diss = jnp.log(target_diss[diss_active])

    print(f"log target kappaM range: [{float(log_target_kappaM.min()):.2f}, {float(log_target_kappaM.max()):.2f}]")
    print(f"log target diss  range: [{float(log_target_diss.min()):.2f}, {float(log_target_diss.max()):.2f}]")

    key = jax.random.PRNGKey(SEED)
    key_km, key_diss, key_fit1, key_fit2 = jax.random.split(key, 4)

    print("fitting kappaM_net (log-space)...")
    kappaM_net, loss_km = fit(build_full_function_net(key_km), features, log_target_kappaM, n_epochs, batch_size, lr, key_fit1)
    print("fitting diss_net (log-space, active points only)...")
    diss_net, loss_diss = fit(build_full_function_net(key_diss), features_diss, log_target_diss, n_epochs, batch_size, lr, key_fit2)

    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(common.DATA_DIR / "exp10c_kappaM_net.eqx", kappaM_net)
    eqx.tree_serialise_leaves(common.DATA_DIR / "exp10c_diss_net.eqx", diss_net)
    common.save("exp10c_offline_pretrain", loss_kappaM=loss_km, loss_diss=loss_diss,
                n_steps=n_steps, n_snapshots=n_snapshots, n_epochs=n_epochs)
    print("wrote exp10c_kappaM_net.eqx, exp10c_diss_net.eqx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--n-snapshots", type=int, default=N_SNAPSHOTS)
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    main(args.n_steps, args.n_snapshots, args.n_epochs, args.batch_size, args.lr)
