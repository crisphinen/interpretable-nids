#!/usr/bin/env python3
# Joint (lambda, gamma) sweep (supplement tab:lambdagamma): JointCBM for lambda in {0.1, 0.5, 1.0}
# x gamma in {0, 0.5, 1.0} x seeds 0-2, with the training and scoring of cbm/run_lambda_sweep.py
# (whose gamma=0 column is tab:lambda). Per-seed rows go to results/cbm/lambdagamma_{ds}_seeds.json
# (resumable); --aggregate prints the grid of mean (std) OOD AUROC.
#
# usage: python -m cbm.run_lambdagamma --dataset ctu --device cuda
#        python -m cbm.run_lambdagamma --aggregate --dataset ctu
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from cbm.model import JointCBM
from cbm.train import load_dataset, train_joint, make_tensors, make_loader, compute_val_f1, EMBED_DIM
from cbm.evaluate import load_test_data, ood_auroc, concept_accuracies
from dsets import DATASETS

RES = Path(__file__).parent.parent / "results" / "cbm"
LAMBDAS, GAMMAS, SEEDS = [0.1, 0.5, 1.0], [0.0, 0.5, 1.0], [0, 1, 2]


def run(ds, device):
    (Xtr, ytr, Ctr), (Xva, yva, Cva), nf, nk, nc, _ = load_dataset(ds)
    (Xte, yte, Cte), _, Xun, _, _ = load_test_data(ds)
    ltr = make_loader(*make_tensors(Xtr, ytr, Ctr, device), shuffle=True)
    lva = make_loader(*make_tensors(Xva, yva, Cva, device), shuffle=False)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    dest = RES / f"lambdagamma_{ds}_seeds.json"
    rows = json.loads(dest.read_text()) if dest.exists() else []
    done = {(r["lambda"], r["gamma"], r["seed"]) for r in rows}
    for la in LAMBDAS:
        for g in GAMMAS:
            for s in SEEDS:
                if (la, g, s) in done:
                    print(f"  [skip] la={la} g={g} seed={s}"); continue
                torch.manual_seed(s); np.random.seed(s)
                model = JointCBM(nf, nc, nk, EMBED_DIM).to(device)
                model = train_joint(model, ltr, lva, device, label=f"JointCBM-l{la}-g{g}-s{s}",
                                    lambda_concept=la, gamma_leakage=g)
                model.eval()
                with torch.no_grad():
                    preds = model(Xte_t)[0].argmax(1).cpu().numpy()
                r = {"lambda": la, "gamma": g, "seed": s,
                     "val_f1": float(compute_val_f1(model, lva, device)),
                     "test_f1": float(f1_score(yte, preds, average="weighted", zero_division=0)),
                     "concept_acc": float(concept_accuracies(model, Xte, Cte, device).mean()),
                     "auroc": float(ood_auroc(model, Xtr, ytr, Xte, Xun, device))}
                rows.append(r); dest.write_text(json.dumps(rows, indent=2))
                print(f"  la={la} g={g} seed={s}: testF1={r['test_f1']:.4f} concept={r['concept_acc']:.3f} "
                      f"AUROC={r['auroc']:.4f}", flush=True)


def aggregate(ds):
    rows = json.loads((RES / f"lambdagamma_{ds}_seeds.json").read_text())
    print(f"[{ds}] OOD AUROC mean (std), lambda down, gamma across")
    for la in LAMBDAS:
        cells = []
        for g in GAMMAS:
            a = [r["auroc"] for r in rows if r["lambda"] == la and r["gamma"] == g]
            cells.append(f"{np.mean(a):.3f} ({np.std(a):.3f}) n={len(a)}" if a else "-")
        print(f"   lambda={la}: " + "   ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.dataset); return
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run(args.dataset, device)


if __name__ == "__main__":
    main()
