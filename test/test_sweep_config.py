#!/usr/bin/env python3
"""
The wandb sweep's variant list is hand-maintained, so it drifts.

A variant missing from matrix.yaml is not skipped loudly -- the sweep just
never queues it, every agent finishes cleanly, and plot_matrix_report.py then
renders that variant from whatever older snapshot it can find. That is the
same silent-stale-row failure that once made three crashed variants the
report's only three "passing" rows, arriving by a different route.

`global_1deg` is excluded on purpose (see matrix.yaml), so the test asserts
against an explicit exclusion set rather than plain equality -- otherwise
"deliberately excluded" and "forgotten" look identical.
"""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test"))

from setups_matrix import VARIANTS_BY_NAME

# Variants the sweep deliberately does not queue. Each needs a reason, and
# a way to run it that the guide documents.
EXCLUDED = {
    # ~56x global_4deg per step and its own horizon; run it on its own with
    # --variant global_1deg rather than letting an agent block a node on it
    "global_1deg",
}


MATRIX_YAML = REPO_ROOT / "test" / "sweep" / "matrix.yaml"


def _sweep_params():
    return yaml.safe_load(MATRIX_YAML.read_text())["parameters"]


def _sweep_variants():
    return set(_sweep_params()["variant"]["values"])


def test_sweep_yaml_covers_every_variant_except_the_excluded():
    listed = _sweep_variants()
    known = set(VARIANTS_BY_NAME)
    assert listed == known - EXCLUDED, (
        "test/sweep/matrix.yaml is out of sync with setups_matrix.VARIANTS: "
        f"missing {sorted(known - EXCLUDED - listed)}, unexpected {sorted(listed - known)}"
    )


def test_excluded_variants_still_exist():
    """A stale exclusion would silently start dropping nothing, or hide a rename."""
    unknown = EXCLUDED - set(VARIANTS_BY_NAME)
    assert not unknown, f"EXCLUDED names no longer in setups_matrix.VARIANTS: {sorted(unknown)}"
