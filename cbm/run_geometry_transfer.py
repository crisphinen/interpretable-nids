#!/usr/bin/env python3
# Do known-class statistics predict OOD detectability on an unseen corpus? Leave-one-corpus-out
# over the 90 CBM concept-space models of results/cbm/geometry.json (30 per corpus: ctuk, cicr,
# ton). For each known-class statistic and held-out corpus H, fit AUROC = a + b * statistic by
# least squares on the 60 models of the other two corpora and evaluate on the 30 models of H:
#   mae        mean absolute AUROC error on H (baseline: predict the training-corpora mean AUROC)
#   spearman   rank correlation of predicted and true AUROC within H
#   top        whether the configuration with the highest mean predicted AUROC is the one with
#              the highest mean true AUROC in H
# The unknown/known distance ratio uses unknown flows and is excluded.
# Writes results/cbm/geometry_transfer.json.
#
# usage: python -m cbm.run_geometry_transfer

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
STATS = ["within_var", "fisher", "ridge_lda", "silhouette", "nc_margin", "entropy", "f1"]
CORPORA = ["ctuk", "cicr", "ton"]


def main():
    g = json.loads((ROOT / "results" / "cbm" / "geometry.json").read_text())["cbm"]
    rows = {c: g[c]["rows"] for c in CORPORA}
    out = []
    print(f"{'held-out':8s} {'statistic':11s} {'slope':>7s} {'MAE':>6s} {'MAE base':>8s} {'spearman':>8s} {'top':>4s}")
    for h in CORPORA:
        tr = [r for c in CORPORA if c != h for r in rows[c]]
        te = rows[h]
        ytr = np.array([r["auroc"] for r in tr]); yte = np.array([r["auroc"] for r in te])
        configs = sorted({r["config"] for r in te})
        true_best = max(configs, key=lambda k: np.mean([r["auroc"] for r in te if r["config"] == k]))
        for s in STATS:
            xtr = np.array([r[s] for r in tr]); xte = np.array([r[s] for r in te])
            b, a = np.polyfit(xtr, ytr, 1)
            pred = a + b * xte
            mae = float(np.abs(pred - yte).mean()); mae0 = float(np.abs(ytr.mean() - yte).mean())
            rho = float(spearmanr(pred, yte).correlation) if b != 0 else 0.0
            pbest = max(configs, key=lambda k: np.mean([p for p, r in zip(pred, te) if r["config"] == k]))
            out.append(dict(held_out=h, statistic=s, slope=float(b), mae=mae, mae_baseline=mae0,
                            spearman=rho, predicted_best=pbest, true_best=true_best, top_correct=pbest == true_best))
            print(f"{h:8s} {s:11s} {b:+7.3f} {mae:6.3f} {mae0:8.3f} {rho:+8.2f} {'yes' if pbest == true_best else 'no':>4s}")
    (ROOT / "results" / "cbm" / "geometry_transfer.json").write_text(json.dumps(out, indent=2))
    n = len(out)
    print(f"\nbeats mean baseline on MAE: {sum(o['mae'] < o['mae_baseline'] for o in out)}/{n}; "
          f"positive spearman: {sum(o['spearman'] > 0 for o in out)}/{n}; best configuration found: "
          f"{sum(o['top_correct'] for o in out)}/{n}")
    for h in CORPORA:
        oo = [o for o in out if o["held_out"] == h]
        print(f"  {h}: MAE beats baseline {sum(o['mae'] < o['mae_baseline'] for o in oo)}/7, "
              f"spearman>0 {sum(o['spearman'] > 0 for o in oo)}/7, best config {sum(o['top_correct'] for o in oo)}/7 "
              f"(true best {oo[0]['true_best']})")


if __name__ == "__main__":
    main()
