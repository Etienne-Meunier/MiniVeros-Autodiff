"""
Writes DATA_DIR/full_state/acc_full_config.json: the acc/full StaticConfig and
Parameters (with c_k=0.10, c_eps=0.70 defaults -- the 5 TOP5 runs each only
override c_k/c_eps, everything else here is shared by all of them), for the
"configuration in detail" section of the top5 runs report.

    python test/ck_ceps_sweep/dump_config.py
"""

import dataclasses
import json

import equinox as eqx
import jax.numpy as jnp

import common
from mini_veros.setups.acc import full


def to_json(v):
    if isinstance(v, eqx.Module):
        v = dump(v)
    elif isinstance(v, jnp.ndarray):
        v = v.item() if v.ndim == 0 else v.tolist()
    return v


def dump(dc):
    return {f.name: to_json(getattr(dc, f.name)) for f in dataclasses.fields(dc)}


def main():
    model, _, _ = full.build({})
    out = {"config": dump(model.config), "parameters": dump(model.parameters),
           "TOP5_overrides": [{"tke_closure.c_k": c_k, "tke_closure.c_eps": c_eps} for c_k, c_eps in common.TOP5]}
    path = common.DATA_DIR / "full_state" / "acc_full_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
