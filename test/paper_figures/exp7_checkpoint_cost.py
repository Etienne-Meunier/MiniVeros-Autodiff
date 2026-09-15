"""
Experiment 7 -- what a long gradient costs, and how checkpointing trades it.

Reverse mode over an n-step rollout has to keep something from the forward pass
for every step it will differentiate. `common.rollout` stores one state per block
of `checkpoint_every` steps and recomputes the steps inside a block during the
backward pass, so the two costs move in opposite directions:

    memory  ~  (n / checkpoint_every) * (one stored state)
             +  checkpoint_every * (one step's tape)
    time    ~  n extra forward steps, one recompute per step

The textbook optimum for that is checkpoint_every = sqrt(n), but only when a
stored state and a taped step cost the same. They do not here: measured on this
model a stored state is a few MB and a taped step is several times that, so the
inner term dominates and the optimum sits at a much smaller block than sqrt(n).
This measures both ends of the trade: wall time for value-and-gradient, and the
peak resident memory the tape adds on top of the process baseline.

Memory is the awkward one to measure honestly -- this jaxlib exposes no
memory_analysis on the compiled executable, and peak RSS within a process is a
high-water mark that never comes back down. So the parent process runs one child
per setting and reads its peak RSS, which is a real measurement rather than an
increment. That is what the `--every` flag is for; you do not normally pass it.

    python test/paper_figures/exp7_checkpoint_cost.py

Two rollouts are measured: H/2 days, inside the window where figure 1 says the
gradient is trustworthy, and 4H days, past it (H = common.HORIZON_DAYS). The
cost trade looks the same either way, but the gradient does not -- inside the
window every schedule returns the same number to every digit, and past it they
disagree by orders of magnitude and a sign. Checkpointing is exact; the
rollout is what stopped being.
"""

import argparse
import json
import resource
import subprocess
import sys
import time

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import common


def default_rollouts():
    if common.HORIZON_DAYS is None:
        raise SystemExit("common.HORIZON_DAYS is not set yet -- run exp1 first, or pass --rollouts explicitly")
    return [common.HORIZON_DAYS // 2, common.HORIZON_DAYS * 4]


CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64, 80, 160, 320]  # those that divide the rollout are used


def peak_rss_mb():
    """Peak resident memory so far. ru_maxrss is bytes on macOS, kilobytes on Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def measure(n_steps, checkpoint_every):
    """Child process: time one value-and-gradient at this blocking, and report peak RSS."""
    model, state0, forcing_fn = common.spinup()
    baseline_mb = peak_rss_mb()  # JAX, the model and the spun-up state, before any tape

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def value_and_grad(params):
        model_p = common.with_params(model, params)
        return common.mean_square_sst(common.rollout(model_p, state0, forcing_fn, n_steps, checkpoint_every).state)

    params = {"c_k": jnp.array(0.1)}
    t0 = time.time()
    loss, grads = value_and_grad(params)
    jnp.asarray(grads["c_k"]).block_until_ready()
    compile_and_run = time.time() - t0

    t0 = time.time()
    loss, grads = value_and_grad(params)
    jnp.asarray(grads["c_k"]).block_until_ready()
    run = time.time() - t0

    peak_mb = peak_rss_mb()
    print("RESULT " + json.dumps({
        "checkpoint_every": checkpoint_every, "stored_states": n_steps // checkpoint_every,
        "compile_and_run_s": compile_and_run, "run_s": run,
        "peak_mb": peak_mb, "baseline_mb": baseline_mb, "tape_mb": peak_mb - baseline_mb,
        "grad": float(grads["c_k"]),
    }))


def main(rollouts, checkpoints):
    rows = []
    for n_steps in rollouts:
        print(f"--- rollout of {n_steps} steps")
        for every in checkpoints:
            if n_steps % every:
                continue
            out = subprocess.run(
                [sys.executable, __file__, "--n-steps", str(n_steps), "--every", str(every)],
                capture_output=True, text=True,
            )
            line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line is None:
                print(f"checkpoint_every={every} failed:\n{out.stdout[-500:]}{out.stderr[-1500:]}")
                continue
            rows.append({"n_steps": n_steps, **json.loads(line[len("RESULT "):])})
            print(f"checkpoint_every={every:4d}  stored={rows[-1]['stored_states']:5d}  "
                  f"run={rows[-1]['run_s']:6.2f}s  tape={rows[-1]['tape_mb']:7.0f} MB  "
                  f"grad={rows[-1]['grad']:+.8e}")

    common.save(
        "exp7_checkpoint_cost",
        rollout=[r["n_steps"] for r in rows],
        checkpoint_every=[r["checkpoint_every"] for r in rows],
        stored_states=[r["stored_states"] for r in rows],
        run_s=[r["run_s"] for r in rows],
        compile_and_run_s=[r["compile_and_run_s"] for r in rows],
        peak_mb=[r["peak_mb"] for r in rows],
        baseline_mb=[r["baseline_mb"] for r in rows],
        tape_mb=[r["tape_mb"] for r in rows],
        grad=[r["grad"] for r in rows],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-steps", type=int, default=320, help="internal: the child's rollout")
    parser.add_argument("--rollouts", type=int, nargs="+", default=None)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=CHECKPOINTS)
    parser.add_argument("--every", type=int, default=None, help="internal: measure this one setting")
    args = parser.parse_args()
    if args.every is not None:
        measure(args.n_steps, args.every)
    else:
        main(args.rollouts or default_rollouts(), args.checkpoints)
