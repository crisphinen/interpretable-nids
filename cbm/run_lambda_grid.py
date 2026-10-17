#!/usr/bin/env python3
# Concept-supervision weight (lambda) x concept-fidelity weight (gamma) for JointCBM on the main
# splits (ctuk, cicr), seeds 0-4, with the training and scoring protocol of cbm/run_multiseed.py.
# The lambda=0.5 cells reuse the results/cbm/ms/{ds}_JointCBM_g{gamma}_s{seed}.pt checkpoints of
# Table III, so that column reproduces Table III exactly; the other cells are trained here and
# saved as results/cbm/ms/{ds}_JointCBM_l{lambda}_g{gamma}_s{seed}.pt.
# The gamma=0 column is the lambda-sensitivity table; the full grid is the (lambda, gamma) table.
# Per-seed rows: results/cbm/lambdagrid_{ds}_seeds.json (resumable).
#
# usage: python -m cbm.run_lambda_grid --dataset ctuk --device cuda
#        python -m cbm.run_lambda_grid --dataset ctuk --aggregate

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from cbm.model import JointCBM
from cbm.train import load_dataset, train_joint, make_tensors, make_loader, compute_val_f1, EMBED_DIM
from cbm.evaluate import load_test_data, ood_auroc, concept_accuracies

RES, CK = ROOT / "results" / "cbm", ROOT / "results" / "cbm" / "ms"
LAMBDAS, GAMMAS, SEEDS = [0.1, 0.5, 1.0], [0.0, 0.5, 1.0], [0, 1, 2, 3, 4]
gname = lambda g: "0" if g == 0 else str(g)


def run(ds, device):
    (Xtr, ytr, Ctr), (Xva, yva, Cva), nf, nk, nc, _ = load_dataset(ds)
    (Xte, yte, Cte), _, Xun, _, _ = load_test_data(ds)
    a, b, c = make_tensors(Xtr, ytr, Ctr, device); d, e, f = make_tensors(Xva, yva, Cva, device)
    ltr = make_loader(a, b, c, shuffle=True); lva = make_loader(d, e, f, shuffle=False)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    dest = RES / f"lambdagrid_{ds}_seeds.json"
    rows = json.loads(dest.read_text()) if dest.exists() else []
    done = {(r["lambda"], r["gamma"], r["seed"]) for r in rows}
    for la in LAMBDAS:
        for g in GAMMAS:
            for s in SEEDS:
                if (la, g, s) in done:
                    continue
                if la == 0.5:
                    ck = torch.load(CK / f"{ds}_JointCBM_g{gname(g)}_s{s}.pt", map_location=device, weights_only=False)
                    model = JointCBM(nf, nc, nk, EMBED_DIM).to(device)
                    model.load_state_dict(ck["model_state_dict"]); source = "table_iii"
                else:
                    torch.manual_seed(s); np.random.seed(s)
                    model = JointCBM(nf, nc, nk, EMBED_DIM).to(device)
                    model = train_joint(model, ltr, lva, device, label=f"JointCBM-l{la}-g{g}-s{s}",
                                        lambda_concept=la, gamma_leakage=g)
                    torch.save({"model_state_dict": model.state_dict(), "variant": "JointCBM", "seed": s,
                                "lambda": la, "gamma": g, "n_features": nf, "n_classes": nk, "n_concepts": nc},
                               CK / f"{ds}_JointCBM_l{la}_g{gname(g)}_s{s}.pt")
                    source = "trained"
                model.eval()
                with torch.no_grad():
                    preds = model(Xte_t)[0].argmax(1).cpu().numpy()
                np.random.seed(42)                       # same evaluation subset as run_multiseed
                r = {"lambda": la, "gamma": g, "seed": s, "source": source,
                     "val_f1": float(compute_val_f1(model, lva, device)),
                     "test_f1": float(f1_score(yte, preds, average="weighted", zero_division=0)),
                     "auroc": float(ood_auroc(model, Xtr, ytr, Xte, Xun, device)),
                     "concept_acc": float(concept_accuracies(model, Xte, Cte, device).mean())}
                rows.append(r); dest.write_text(json.dumps(rows, indent=2))
                print(f"  [{ds}] la={la} g={g} s={s} ({source}): F1={r['test_f1']:.4f} "
                      f"concept={r['concept_acc']:.3f} AUROC={r['auroc']:.4f}", flush=True)


def aggregate(ds):
    rows = json.loads((RES / f"lambdagrid_{ds}_seeds.json").read_text())
    print(f"[{ds}] AUROC mean (std), lambda down, gamma across; then test F1 and concept acc at gamma=0")
    for la in LAMBDAS:
        cells = []
        for g in GAMMAS:
            a = [r["auroc"] for r in rows if r["lambda"] == la and r["gamma"] == g]
            cells.append(f"{np.mean(a):.3f} ({np.std(a):.3f}) n={len(a)}")
        z = [r for r in rows if r["lambda"] == la and r["gamma"] == 0.0]
        print(f"  lambda={la}: " + "  ".join(cells) + f" | F1 {np.mean([r['test_f1'] for r in z]):.4f} "
              f"valF1 {np.mean([r['val_f1'] for r in z]):.4f} concept {np.mean([r['concept_acc'] for r in z]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ctuk", "cicr"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.dataset); return
    run(args.dataset, torch.device(args.device if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
