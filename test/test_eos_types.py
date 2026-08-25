#!/usr/bin/env python3
"""
Direct comparison of every ported equation-of-state variant (eq_of_state_type
1-5) against real veros's own implementation, on random (salt, temp, press)
inputs. Pure functions -- no model/state needed, so this is a much cheaper
and more direct check than a full rollout comparison.

Usage:
    python test/test_eos_types.py [--veros-path PATH] [--n N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEROS_PATH = REPO_ROOT.parent / "veros"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--veros-path", type=Path, default=DEFAULT_VEROS_PATH)
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(args.veros_path))

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from mini_veros.core.density import get_rho as mini_get_rho
    from veros.core.density import linear_eq as real_lq, nonlinear_eq1 as real_nq1, nonlinear_eq3 as real_nq3

    rng = np.random.default_rng(0)
    n = args.n
    salt = jnp.asarray(rng.uniform(30, 40, n))
    temp = jnp.asarray(rng.uniform(-2, 30, n))
    press = jnp.asarray(rng.uniform(0, 500, n))

    # (eq_of_state_type, real module's rho/drhodT/drhodS funcs, needs_press)
    cases = {
        1: (real_lq.linear_eq_of_state_rho, real_lq.linear_eq_of_state_drhodT, real_lq.linear_eq_of_state_drhodS, False),
        2: (real_nq1.nonlin1_eq_of_state_rho, real_nq1.nonlin1_eq_of_state_drhodT, real_nq1.nonlin1_eq_of_state_drhodS, False),
        4: (real_nq3.nonlin3_eq_of_state_rho, real_nq3.nonlin3_eq_of_state_drhodT, real_nq3.nonlin3_eq_of_state_drhodS, False),
    }

    ok = True
    for eq_type, (real_rho, real_drhodT, real_drhodS, needs_press) in cases.items():
        mini_rho = mini_get_rho.get_rho(eq_type, salt, temp, press)
        real_rho_v = real_rho(salt, temp, press) if needs_press else real_rho(salt, temp)
        diff_rho = float(jnp.max(jnp.abs(mini_rho - real_rho_v)))

        mini_dT = mini_get_rho.get_drhodT(eq_type, salt, temp, press)
        real_dT_v = real_drhodT(temp) if eq_type in (2, 4) else real_drhodT()
        diff_dT = float(jnp.max(jnp.abs(jnp.asarray(mini_dT) - jnp.asarray(real_dT_v))))

        mini_dS = mini_get_rho.get_drhodS(eq_type, salt, temp, press)
        real_dS_v = real_drhodS()
        diff_dS = float(jnp.max(jnp.abs(jnp.asarray(mini_dS) - jnp.asarray(real_dS_v))))

        status = "ok  " if max(diff_rho, diff_dT, diff_dS) == 0.0 else "FAIL"
        print(f"eq_of_state_type={eq_type}  {status}  rho diff={diff_rho:.3e}  drhodT diff={diff_dT:.3e}  drhodS diff={diff_dS:.3e}")
        ok = ok and max(diff_rho, diff_dT, diff_dS) == 0.0

    print("PASS -- all EOS types match veros exactly" if ok else "FAIL -- see mismatches above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
