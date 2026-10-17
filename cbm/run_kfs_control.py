#!/usr/bin/env python3
# Known-class feature-selection control, CBM side (supplement Table kfs).
# Trains MLPBaseline, JointCBM (gamma 0/0.1/0.5/1.0), SequentialCBM and HybridCBM for
# seeds 0-4 on the two CTU datasets (same rows):
#   ctuk  - features selected on known-class training rows only (leak-free; Table I CTU)
#   ctuko - the submitted 37-feature list on the same rows
# using the cbm.run_multiseed protocol, then aggregates to
# results/cbm/multiseed_{ctuk,ctuko}_v2.json. Resumable: finished (variant, seed) rows are
# skipped.
#
# usage: python -m cbm.run_kfs_control --device cuda [--datasets ctuk ctuko] [--seeds 5]
import argparse

import torch

from cbm.run_multiseed import run, aggregate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ctuk", "ctuko"], choices=["ctuk", "ctuko"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    for ds in args.datasets:
        run(ds, list(range(args.seeds)), device)
    aggregate(args.seeds)


if __name__ == "__main__":
    main()
