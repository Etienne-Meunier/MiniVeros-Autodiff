"""
Shared NN pieces for the full-function closure experiments (10c onward).

Features are [Nsqr, sqrttke, mxl, depth], the same four mini_veros.core.tke.tke_closure_fields
always builds -- but raw, they span wildly different scales (Nsqr ~1e-4, mxl ~1-100, depth
~-5000..0), which is fine for the coefficient-mode nets (their zero-weight init makes the input
scale irrelevant) but not for a genuinely-trained full-function net. NormalizedMLP bakes a fixed
(non-trained) rescaling into the net itself, since tke_closure_fields calls whatever net is
installed with raw features -- normalization has to travel with the net, not live externally.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

FEATURE_SCALE = (1e-4, 0.01, 10.0, 1000.0)  # Nsqr, sqrttke, mxl, depth -- typical magnitudes


class NormalizedMLP(eqx.Module):
    """An eqx.nn.MLP preceded by a fixed elementwise rescaling (not trained)."""

    mlp: eqx.nn.MLP
    scale: tuple[float, ...] = eqx.field(static=True)

    def __call__(self, x):
        return self.mlp(x / jnp.array(self.scale))


def build_full_function_net(key, width=32, depth=2, activation=jax.nn.tanh, scale=FEATURE_SCALE):
    """A fresh (untrained) NormalizedMLP: in_size=len(scale), out_size=1. tanh by default --
    smooth (no kinks), unlike relu's default -- avoiding a needlessly non-smooth closure feeds
    into the same story exp6_gradient_spikes.py tells about where gradient amplification comes
    from (discontinuities like min/max and flux limiters), without deliberately adding a new one.
    """
    mlp = eqx.nn.MLP(in_size=len(scale), out_size=1, width_size=width, depth=depth,
                      activation=activation, key=key)
    return NormalizedMLP(mlp=mlp, scale=scale)
