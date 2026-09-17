#!/usr/bin/env python3
# Reconstructed multi-seed CBM driver (the original was ad hoc and lost).
# Trains MLPBaseline, JointCBM (gamma in {0,0.1,0.5,1.0}), SequentialCBM and
# HybridCBM for seeds 0..N-1 on the standard split with the current code and
# concept labels, and records per seed: val F1 (model-selection split), test F1,
# per-concept accuracy on test_known, per-class-Mahalanobis OOD AUROC and
# TPR@5%FPR (fit on train, test_known vs unknown; cbm.evaluate.ood_auroc), and
# a concept-leakage probe (linear probe from soft vs binarised concepts to
# class, trained on train / scored on test_known; leakage = acc gap).
# Checkpoints go to results/cbm/ms/. Per-seed rows are written incrementally
# (resumable); --aggregate summarises to results/cbm/multiseed_{ds}_v2.json.
#
# usage: python -m cbm.run_multiseed --dataset ctu --seeds 5 --device cuda
#        python -m cbm.run_multiseed --dataset cic --seed_list 5 6 7 8 9 --variants MLPBaseline JointCBM_g0 ...
#        python -m cbm.run_multiseed --aggregate
# Seeds 0-4 aggregate to multiseed_{ds}_v2.json; seeds 5-9 (the independent
# replication set) to multiseed_{ds}_v2_affirm.json.
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from cbm.model import MLPBaseline, JointCBM, SequentialCBM, HybridCBM
from cbm.train import (load_dataset, train_joint, train_classifier_stage2,
                       make_tensors, make_loader, compute_val_f1, EMBED_DIM, LAMBDA_CONCEPT)
from cbm.evaluate import load_test_data, ood_auroc, tpr_at_fpr, concept_accuracies

RES = Path(__file__).parent.parent / "results" / "cbm"
CK = RES / "ms"
VARIANTS = [("MLPBaseline", 0.0), ("JointCBM_g0", 0.0), ("JointCBM_g0.1", 0.1),
            ("JointCBM_g0.5", 0.5), ("JointCBM_g1.0", 1.0),
            ("SequentialCBM", 0.0), ("HybridCBM", 0.0)]


def build(name, nf, nc, nk):
    return {"MLPBaseline": lambda: MLPBaseline(nf, nk, EMBED_DIM),
            "JointCBM": lambda: JointCBM(nf, nc, nk, EMBED_DIM),
            "SequentialCBM": lambda: SequentialCBM(nf, nc, nk, EMBED_DIM),
            "HybridCBM": lambda: HybridCBM(nf, nc, nk, EMBED_DIM)}[name]()


def leakage_probe(model, Xtr, ytr, Xte, yte, device):
    with torch.no_grad():
        ctr = model.get_embedding(torch.tensor(Xtr, dtype=torch.float32, device=device)).cpu().numpy()
        cte = model.get_embedding(torch.tensor(Xte, dtype=torch.float32, device=device)).cpu().numpy()
    def acc(a, b):
        clf = LogisticRegression(max_iter=2000).fit(a, ytr)
        return float((clf.predict(b) == yte).mean())
    a_soft = acc(ctr, cte); a_bin = acc((ctr > 0.5).astype(np.float32), (cte > 0.5).astype(np.float32))
    return a_soft, a_bin


def run(ds, seeds, device, variants=None):
    (Xtr, ytr, Ctr), (Xva, yva, Cva), nf, nk, nc, names = load_dataset(ds)
    (Xte, yte, Cte), _, Xun, _, _ = load_test_data(ds)
    a, b, c = make_tensors(Xtr, ytr, Ctr, device); d, e, f = make_tensors(Xva, yva, Cva, device)
    ltr = make_loader(a, b, c, shuffle=True); lva = make_loader(d, e, f, shuffle=False)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    CK.mkdir(exist_ok=True)
    dest = RES / f"multiseed_{ds}_v2_seeds.json"
    rows = json.loads(dest.read_text()) if dest.exists() else []
    done = {(r["variant"], r["seed"]) for r in rows}
    for vname, gamma in VARIANTS:
        if variants and vname not in variants:
            continue
        base = "JointCBM" if vname.startswith("JointCBM") else vname
        for s in seeds:
            if (vname, s) in done:
                print(f"  [skip] {vname} s{s}"); continue
            torch.manual_seed(s); np.random.seed(s)
            model = build(base, nf, nc, nk).to(device)
            if base == "SequentialCBM":
                model = train_joint(model, ltr, lva, device, label=f"{vname}-s{s}", concept_only=True)
                model = train_classifier_stage2(model, ltr, lva, device, label=f"{vname}-s{s}")
            else:
                model = train_joint(model, ltr, lva, device, label=f"{vname}-s{s}",
                                    lambda_concept=LAMBDA_CONCEPT if base != "MLPBaseline" else 0.0,
                                    gamma_leakage=gamma)
            model.eval()
            torch.save({"model_state_dict": model.state_dict(), "variant": vname, "seed": s,
                        "gamma": gamma, "n_features": nf, "n_classes": nk, "n_concepts": nc},
                       CK / f"{ds}_{vname}_s{s}.pt")
            with torch.no_grad():
                preds = model(Xte_t)[0].argmax(1).cpu().numpy()
            np.random.seed(42)
            r = {"variant": vname, "seed": s, "gamma": gamma,
                 "val_f1": float(compute_val_f1(model, lva, device)),
                 "test_f1": float(f1_score(yte, preds, average="weighted", zero_division=0)),
                 "auroc": float(ood_auroc(model, Xtr, ytr, Xte, Xun, device)),
                 "tpr5": float(tpr_at_fpr(model, Xtr, ytr, Xte, Xun, device))}
            if base != "MLPBaseline":
                ca = concept_accuracies(model, Xte, Cte, device)
                r["concept_acc"] = float(ca.mean()); r["concept_acc_per"] = dict(zip(names, map(float, ca)))
                a_s, a_b = leakage_probe(model, Xtr, ytr, Xte, yte, device)
                r["probe_soft"], r["probe_bin"], r["leakage"] = a_s, a_b, a_s - a_b
            rows.append(r); dest.write_text(json.dumps(rows, indent=2))
            print(f"  {vname:14} s{s}: valF1={r['val_f1']:.4f} testF1={r['test_f1']:.4f} "
                  f"AUROC={r['auroc']:.3f} TPR={r['tpr5']:.3f}"
                  + (f" concept={r['concept_acc']:.3f} leak={r['leakage']:.3f}" if 'leakage' in r else ""))
    print(f"[{ds}] wrote {dest}")


def aggregate(require):
  for ds in ["ctu", "cic"]:
    f = RES / f"multiseed_{ds}_v2_seeds.json"
    if not f.exists(): continue
    all_rows = json.loads(f.read_text())
    for tag, seed_set in [("", range(5)), ("_affirm", range(5, 10))]:
        rows = [r for r in all_rows if r["seed"] in seed_set]
        if not rows: continue
        out = {"dataset": ds, "seeds": sorted({r["seed"] for r in rows}), "results": {}}
        for vname, _ in VARIANTS:
            rs = [r for r in rows if r["variant"] == vname]
            if 0 < len(rs) < require: print(f"INCOMPLETE {ds}{tag} {vname} {len(rs)}/{require}")
            if not rs: continue
            g = lambda k: np.array([r[k] for r in rs])
            o = {"seeds": [r["seed"] for r in rs]}
            for k in ["val_f1", "test_f1", "auroc", "tpr5"] + (["concept_acc", "leakage"] if "leakage" in rs[0] else []):
                o[f"{k}_mean"], o[f"{k}_std"], o[f"{k}_all"] = float(g(k).mean()), float(g(k).std()), [float(x) for x in g(k)]
            if "concept_acc_per" in rs[0]:
                o["concept_acc_per_mean"] = {n: float(np.mean([r["concept_acc_per"][n] for r in rs])) for n in rs[0]["concept_acc_per"]}
            out["results"][vname] = o
            print(f"{ds}{tag} {vname:14} n={len(rs)} valF1={o['val_f1_mean']:.4f}±{o['val_f1_std']:.4f} "
                  f"AUROC={o['auroc_mean']:.3f}±{o['auroc_std']:.3f} TPR={o['tpr5_mean']:.3f}±{o['tpr5_std']:.3f}"
                  + (f" concept={o['concept_acc_mean']:.3f} leak={o['leakage_mean']:.3f}±{o['leakage_std']:.3f}" if 'leakage_mean' in o else ""))
        (RES / f"multiseed_{ds}_v2{tag}.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ctu", "cic"]); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed_list", type=int, nargs="+", help="explicit seeds (overrides --seeds)")
    ap.add_argument("--variants", nargs="+", help="subset of variant names to run")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--require", type=int, default=5)
    args = ap.parse_args()
    if args.aggregate: aggregate(args.require); return
    if not args.dataset: ap.error("--dataset required unless --aggregate")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run(args.dataset, args.seed_list or list(range(args.seeds)), device, args.variants)


if __name__ == "__main__":
    main()
