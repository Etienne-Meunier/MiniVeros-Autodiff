from longterm_gradients import grad_run_fn_alt

from __init__ import PRP
import jax

jax.config.update("jax_enable_x64", True)
from mini_veros import loop
from mini_veros.setups.acc.basic import build
import equinox as eqx
from utils_viz import browse
from jax import lax

# Spin up model 
model, integrator_state, forcing_fn = build()
integrator_state, keep = eqx.filter_jit(loop.run)(model, integrator_state, forcing_fn, lambda x : None, 1000, 10)

loss, grad_params = grad_run_fn_alt(model.parameters, model, integrator_state, forcing_fn, 20, 10)


loss
import optimistix as optx

def fn(params, args):
    loss, _ = grad_run_fn_alt(params, model, integrator_state, forcing_fn, 20, 10)
    return loss

solver = optx.BFGS(rtol=1e-6, atol=1e-8)
sol = optx.minimise(fn, solver, model.parameters, max_steps=100)
