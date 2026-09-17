#!/usr/bin/env python3
# Concept-supervision-weight (lambda) sensitivity driver.
# Trains JointCBM (gamma=0) on the
# standard split for each lambda x seed and reports known-class F1, mean concept
# accuracy and per-class-Mahalanobis OOD AUROC, using the same loaders and
# scoring as cbm.train / cbm.evaluate. Writes per-seed rows to
# results/cbm/lambda_sensitivity_{ds}_seeds.json; --aggregate rebuilds
# results/cbm/lambda_sensitivity.json (input of Table tab:lambda) from them.
#
# usage: python -m cbm.run_lambda_sweep --dataset ctu --seeds 5 --device cuda
#        python -m cbm.run_lambda_sweep --aggregate
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from cbm.model import JointCBM
from cbm.train import (load_dataset, train_joint, make_tensors, make_loader,
                       compute_val_f1, EMBED_DIM)
from cbm.evaluate import load_test_data, ood_auroc, concept_accuracies

RES = Path(__file__).parent.parent / "results" / "cbm"
LAMBDAS = [0.1, 0.5, 1.0]


def run(ds, seeds, device):
    (Xtr, ytr, Ctr), (Xva, yva, Cva), nf, nk, nc, _ = load_dataset(ds)
    (Xte, yte, Cte), _, Xun, _, _ = load_test_data(ds)
    Xtr_t, ytr_t, Ctr_t = make_tensors(Xtr, ytr, Ctr, device)
    Xva_t, yva_t, Cva_t = make_tensors(Xva, yva, Cva, device)
    ltr = make_loader(Xtr_t, ytr_t, Ctr_t, shuffle=True)
    lva = make_loader(Xva_t, yva_t, Cva_t, shuffle=False)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    dest = RES / f"lambda_sensitivity_{ds}_seeds.json"
    rows = json.loads(dest.read_text()) if dest.exists() else []
    done = {(r["lambda"], r["seed"]) for r in rows}
    for la in LAMBDAS:
        for s in seeds:
            if (la, s) in done:
                print(f"  [skip] la={la} seed={s}"); continue
            torch.manual_seed(s); np.random.seed(s)
            model = JointCBM(nf, nc, nk, EMBED_DIM).to(device)
            model = train_joint(model, ltr, lva, device, label=f"JointCBM-l{la}-s{s}",
                                lambda_concept=la, gamma_leakage=0.0)
            model.eval()
            with torch.no_grad():
                preds = model(Xte_t)[0].argmax(1).cpu().numpy()
            r = {"lambda": la, "seed": s,
                 "val_f1": float(compute_val_f1(model, lva, device)),
                 "test_f1": float(f1_score(yte, preds, average="weighted", zero_division=0)),
                 "concept_acc": float(concept_accuracies(model, Xte, Cte, device).mean()),
                 "auroc": float(ood_auroc(model, Xtr, ytr, Xte, Xun, device))}
            rows.append(r)
            dest.write_text(json.dumps(rows, indent=2))
            print(f"  la={la} seed={s}: valF1={r['val_f1']:.4f} testF1={r['test_f1']:.4f} "
                  f"concept={r['concept_acc']:.3f} AUROC={r['auroc']:.4f}")
    print(f"[{ds}] wrote {dest}")


def aggregate(require):
    out, incomplete = {}, []
    for ds in ["ctu", "cic"]:
        f = RES / f"lambda_sensitivity_{ds}_seeds.json"
        rows = json.loads(f.read_text()) if f.exists() else []
        out[ds] = {}
        for la in LAMBDAS:
            rs = [r for r in rows if r["lambda"] == la]
            if len(rs) < require:
                incomplete.append(f"{ds} la={la} ({len(rs)}/{require})")
            if not rs:
                continue
            g = lambda k: np.array([r[k] for r in rs])
            c = {"f1": g("val_f1").mean(), "f1_std": g("val_f1").std(),
                 "test_f1": g("test_f1").mean(), "test_f1_std": g("test_f1").std(),
                 "concept_acc": g("concept_acc").mean(), "concept_acc_std": g("concept_acc").std(),
                 "auroc": g("auroc").mean(), "auroc_std": g("auroc").std(), "n_seeds": len(rs)}
            out[ds][str(la)] = {k: float(v) for k, v in c.items()}
            print(f"{ds:3s} la={la:<4} n={len(rs)}  F1={c['f1']:.4f}±{c['f1_std']:.4f}  "
                  f"concept={c['concept_acc']:.3f}±{c['concept_acc_std']:.3f}  "
                  f"AUROC={c['auroc']:.3f}±{c['auroc_std']:.3f}")
    if incomplete:
        print("INCOMPLETE:", "; ".join(incomplete)); raise SystemExit(1)
    (RES / "lambda_sensitivity.json").write_text(json.dumps(out, indent=2))
    print("wrote", RES / "lambda_sensitivity.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ctu", "cic"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--require", type=int, default=5)
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.require); return
    if not args.dataset:
        ap.error("--dataset required unless --aggregate")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run(args.dataset, list(range(args.seeds)), device)


if __name__ == "__main__":
    main()
