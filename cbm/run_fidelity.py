#!/usr/bin/env python3
# JointCBM concept fidelity (tab:fidelity CBM rows, supplement tab:cintervene), five seeds per gamma,
# on the known test split:
#   f1_pred   weighted F1 with the model's predicted concepts (normal inference)
#   f1_gt     declared predicates (the concept labels) fed through the model's own classifier
#   f1_lr     logistic regression trained on declared predicates alone (train -> test; no seeds)
#   probe gap linear probe from soft vs thresholded predicted concepts to class (train -> test);
#             taken from the per-seed rows of cbm.run_multiseed ("leakage")
# Writes results/cbm/fidelity_{ds}.json.
#
# usage: python -m cbm.run_fidelity --datasets ctuk cicr --device cuda
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from cbm.model import JointCBM
from cbm.train import load_dataset, EMBED_DIM
from cbm.evaluate import load_test_data

RES = Path(__file__).parent.parent / "results" / "cbm"
GAMMAS = ["JointCBM_g0", "JointCBM_g0.1", "JointCBM_g0.5", "JointCBM_g1.0"]


def f1(y, p): return float(f1_score(y, p, average="weighted", zero_division=0))


def run(ds, device):
    (Xtr, ytr, Ctr), *_ = load_dataset(ds)
    (Xte, yte, Cte), *_ = load_test_data(ds)
    seeds_rows = json.loads((RES / f"multiseed_{ds}_v2_seeds.json").read_text())
    f1_lr = f1(yte, LogisticRegression(max_iter=2000).fit(Ctr, ytr).predict(Cte))
    out = {"f1_lr": f1_lr, "gamma": {}}
    Xt = torch.tensor(Xte, dtype=torch.float32, device=device)
    Ct = torch.tensor(Cte, dtype=torch.float32, device=device)
    for v in GAMMAS:
        rows = []
        for s in range(5):
            ck = torch.load(RES / "ms" / f"{ds}_{v}_s{s}.pt", map_location=device, weights_only=False)
            m = JointCBM(ck["n_features"], ck["n_concepts"], ck["n_classes"], EMBED_DIM).to(device)
            m.load_state_dict(ck["model_state_dict"]); m.eval()
            with torch.no_grad():
                pred = m(Xt)[0].argmax(1).cpu().numpy()
                gt = m.classifier(Ct).argmax(1).cpu().numpy()
            leak = next(r["leakage"] for r in seeds_rows if r["variant"] == v and r["seed"] == s)
            rows.append({"seed": s, "f1_pred": f1(yte, pred), "f1_gt": f1(yte, gt), "probe_gap": leak})
        agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
               for k in ("f1_pred", "f1_gt", "probe_gap")}
        gap = [r["f1_pred"] - r["f1_gt"] for r in rows]
        agg["gap"] = (float(np.mean(gap)), float(np.std(gap)))
        out["gamma"][v] = {"rows": rows, **agg}
        print(f"[{ds}] {v:14s} F1_pred={agg['f1_pred'][0]:.3f}({agg['f1_pred'][1]:.3f}) "
              f"F1_gt={agg['f1_gt'][0]:.3f}({agg['f1_gt'][1]:.3f}) gap={agg['gap'][0]:.3f} "
              f"probe soft-thr={agg['probe_gap'][0]:.3f}")
    print(f"[{ds}] logistic regression on declared predicates: F1={f1_lr:.3f}")
    (RES / f"fidelity_{ds}.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ctuk", "cicr"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    for ds in args.datasets:
        run(ds, device)


if __name__ == "__main__":
    main()
