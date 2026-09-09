from __init__ import PRP
import jax
jax.config.update("jax_enable_x64", True)
from mini_veros import loop
from mini_veros.setups.acc.basic import build
import equinox as eqx
from utils_viz import browse
import dataclasses

# Jit useful function 

run = eqx.filter_jit(loop.run)
step = eqx.filter_jit(loop.step)

log_fn = lambda integrator_state : integrator_state.state.temp[:,:, -1] # We only log sst

# Spin up model 
model, integrator_state, forcing_fn = build()
integrator_state, keep = run(model, integrator_state, forcing_fn, log_fn, 1000, 10)

browse(keep.transpose(1,2,0)) # Vizualize evolution

# Compute gradient on a step
force = forcing_fn(model, integrator_state.state)

def grad_fn(model, integrator_state, force) :
    ns = step(model, integrator_state, force)
    return (ns.state.temp**2).mean()

# Grad over a state
grad_state = jax.grad(grad_fn, argnums=1)(model, integrator_state, force)
browse(grad_state)

# Grad over params
grad_params = eqx.filter_grad(grad_fn)(model, integrator_state, force)
browse(grad_params)


# Grad over a rollout 
def grad_run_fn(model, integrator_state, forcing_fn, log_fn, n_steps=10, log_every=2) :
    ns, keep = run(model, integrator_state, forcing_fn, log_fn, n_steps=n_steps, log_every=log_every)
    return (ns.state.temp**2).mean(), keep

(loss, keep), grad = jax.value_and_grad(grad_run_fn, argnums=1, has_aux=True)(model, integrator_state, forcing_fn, log_fn, 100, 2)

(loss, keep), grad_params = eqx.filter_value_and_grad(grad_run_fn, has_aux=True)(model, integrator_state, forcing_fn, log_fn, 100, 100)

browse(grad_params)
# You can also target a more precise part of the input this way : 


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def grad_run_fn_alt(params, model, integrator_state, forcing_fn, log_fn, n_steps=10, log_every=2) : 
    model = dataclasses.replace(model, parameters=params)
    ns, keep = run(model, integrator_state, forcing_fn, log_fn, n_steps=n_steps, log_every=log_every)
    return (ns.state.temp**2).mean(), keep


(loss, keep), grad_params = grad_run_fn_alt(model.parameters, model, integrator_state, forcing_fn, log_fn, 300, 100)

browse(grad_params)
