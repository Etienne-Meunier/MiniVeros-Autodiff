from __init__ import PRP
import jax

jax.config.update("jax_enable_x64", True)
from mini_veros import loop
from mini_veros.setups.acc.basic import build
import equinox as eqx
from utils_viz import browse
import dataclasses
from jax import lax

from mini_veros.model import Model
from mini_veros.state import DiagnosticState, Forcing, IntegratorState, PrognosticState, StatefulDiag, Tendencies, _add

# Function with checkpoint 
def run_checkpoint(
    model: Model,
    initial_integrator_state: IntegratorState,
    forcing_fn,
    n_steps: int,
    checkpoint_block: int ) :
    assert n_steps % checkpoint_block == 0, "n_steps must be a multiple of checkpoint_block"

    def inner(integrator_state, _):
        force = forcing_fn(model, integrator_state.state)
        return loop.step(model, integrator_state, force), None

    @jax.checkpoint
    def block(carry, _):
        integrator_state, _ = lax.scan(inner, carry, length=checkpoint_block)
        return integrator_state, None

    final_integrator_state, _ = lax.scan(
        block, initial_integrator_state, length=n_steps // checkpoint_block
    )

    return final_integrator_state

@eqx.filter_jit
@eqx.filter_value_and_grad
def grad_run_fn_alt(params, model, integrator_state, forcing_fn, n_steps=10, checkpoint_block=2) : 
    model = dataclasses.replace(model, parameters=params)
    ns = run_checkpoint(model, integrator_state, forcing_fn, n_steps=n_steps, checkpoint_block=checkpoint_block)
    return (ns.state.temp**2).mean()


if __name__ == '__main__' :
    
    # Spin up model 
    model, integrator_state, forcing_fn = build()
    integrator_state, keep = eqx.filter_jit(loop.run)(model, integrator_state, forcing_fn, log_fn, 1000, 10)
    
    
    (loss, keep), grad_params = grad_run_fn_alt(model.parameters, model, integrator_state, forcing_fn, 20, 10)
    
    grad_params.r_bot
    
    loss, grad_params = grad_run_fn_alt(model.parameters, model, integrator_state, forcing_fn, 3000, 10)
    
    grad_params.r_bot
    keep
    browse(grad_params)
