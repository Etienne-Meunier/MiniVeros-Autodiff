"""
Merge exp11_density_calibration_limit.py's per-length chunks (run with --lengths N --tag lenN)
into the one exp11_density_calibration_limit.npz that fig11*.py read.

    python test/paper_figures/exp11_merge.py --lengths 5 40 80 160 320 --tags len5 len40 len80 len160 len320

Every chunk must share distances, angles_deg, n_iterations, param_names, true, n_layers and
the (c_k, c_eps) grid -- checked, not just assumed -- since fig11*.py treat them as one sweep.
"""

import argparse

import numpy as np

import common

SHARED_KEYS = ["distances", "angles_deg", "n_iterations", "param_names", "true", "n_layers", "c_k", "c_eps"]
STACKED_KEYS = ["landscape", "trajectories", "fitted", "loss_start", "loss_end"]


def main(lengths, tags):
    chunks = [common.load(f"exp11_density_calibration_limit__{tag}") for tag in tags]

    for c, tag, length in zip(chunks, tags, lengths):
        chunk_lengths = [int(v) for v in c["lengths"]]
        if chunk_lengths != [length]:
            raise SystemExit(f"{tag}: expected lengths=[{length}], found {chunk_lengths}")

    reference = chunks[0]
    for c, tag in zip(chunks[1:], tags[1:]):
        for key in SHARED_KEYS:
            if not np.array_equal(c[key], reference[key]):
                raise SystemExit(f"{tag}: {key} does not match {tags[0]}'s -- chunks are not one sweep")

    merged = {key: reference[key] for key in SHARED_KEYS}
    merged["lengths"] = lengths
    merged["dt_tracer"] = reference["dt_tracer"]
    for key in STACKED_KEYS:
        merged[key] = np.concatenate([c[key] for c in chunks], axis=0)

    common.save("exp11_density_calibration_limit", **merged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", required=True,
                         help="the lengths in LENGTHS order, matching --tags position by position")
    parser.add_argument("--tags", nargs="+", required=True,
                         help="each chunk's --tag, same order as --lengths")
    args = parser.parse_args()
    if len(args.lengths) != len(args.tags):
        raise SystemExit("--lengths and --tags must have the same number of entries")
    main(args.lengths, args.tags)
