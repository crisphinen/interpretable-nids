#!/usr/bin/env python3
# Aggregate the lambda_alpha sweep (results/nesy/{ds}_nesy_s{seed}[_a{la}]_eval.json)
# into results/nesy/r2_alpha_sweep.json, the input of cbm/make_alpha_figure.py.
# lambda_alpha=0 evals carry no _a suffix (they are the main-run models).
#
# usage: python -m nesy.aggregate_alpha_sweep [--require N]   (N = min seeds per cell)

import argparse, json
from pathlib import Path
import numpy as np

RES = Path(__file__).parent.parent / "results" / "nesy"
LAMBDAS = ["0.0", "0.05", "0.1", "0.2", "0.5", "1.0"]
SEEDS = range(5)


def cell(ds, la):
    rows = []
    for s in SEEDS:
        suf = "" if float(la) == 0 else f"_a{la}"
        f = RES / f"{ds}_nesy_s{s}{suf}_eval.json"
        if f.exists():
            rows.append(json.load(open(f)))
    if not rows:
        return None
    a = np.array([r["gate_alpha"] for r in rows])
    au = np.array([r["ood_auroc_mahal"] for r in rows])
    f1 = np.array([r["known_f1"] for r in rows])
    return {"alpha": a.mean(), "alpha_std": a.std(), "f1": f1.mean(), "f1_std": f1.std(),
            "auroc": au.mean(), "auroc_std": au.std(), "n_seeds": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require", type=int, default=5)
    ap.add_argument("--out", default=str(RES / "r2_alpha_sweep.json"))
    args = ap.parse_args()
    out, incomplete = {}, []
    for ds in ["ctu", "cic"]:
        out[ds] = {}
        for la in LAMBDAS:
            c = cell(ds, la)
            n = c["n_seeds"] if c else 0
            if n < args.require:
                incomplete.append(f"{ds} la={la} ({n}/{args.require})")
            if c:
                out[ds][la] = c
                print(f"{ds:3s} la={la:<4s} n={n}  alpha={c['alpha']:.3f}±{c['alpha_std']:.3f}  "
                      f"F1={c['f1']:.4f}±{c['f1_std']:.4f}  AUROC={c['auroc']:.3f}±{c['auroc_std']:.3f}")
    if incomplete:
        print("INCOMPLETE:", "; ".join(incomplete))
        raise SystemExit(1)
    json.dump(out, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
