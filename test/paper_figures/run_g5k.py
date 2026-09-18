"""
Generic `g5k launch` entry point for paper-figures scripts that aren't wired into the wandb
sweep (test/sweep/paper.yaml covers exp1b-exp9b; those need common.HORIZON_DAYS already set
by a leader job -- exp10/exp11 don't, but still want the GPU node `g5k launch` provides).

.g5k_config's single `train_script` slot points here; which script actually runs, and with
what flags, is passed through model_cfg at launch time, e.g.:

    g5k launch "exp10_density_calibration" ""
    g5k launch "exp11_density_calibration_limit --lengths 320 --tag len320" ""

`g5k launch` always runs `python <train_script>` -- there's no shell wrapper option -- so this
re-dispatches by editing sys.argv and running the named script under __main__, rather than
being a bash script itself, so that script's own argparse sees exactly what it would running
it directly.
"""
import runpy
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_g5k.py <expN_script_name_without_.py> [script args...]")
    name, *rest = sys.argv[1:]
    sys.argv = [f"{name}.py", *rest]
    runpy.run_path(f"{name}.py", run_name="__main__")
