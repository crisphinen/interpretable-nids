#!/usr/bin/env python3
# Scorer selection without novel traffic. For each CBM/MLP checkpoint (results/cbm/ms/), choose
# between per-class Mahalanobis (concept/embedding space) and the logit-energy score using only
# known traffic: synthetic outliers are built from the known VALIDATION flows, and the scorer that
# better separates validation flows from them is selected. The choice is then compared with the
# scorer that is actually better on the held-out unknown families (same model, same subsets).
#
# Synthetic outliers (declared before running; "perm" is the primary rule):
#   perm  every feature column shuffled independently across validation flows (keeps each
#         marginal, breaks the joint structure of real flows)
#   mix   midpoints of random pairs of validation flows
#
# Writes results/cbm/scorer_selection_{datasets}.json.
#
# usage: python -m cbm.run_scorer_selection --datasets ctuk cicr ton --device cuda

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from cbm.evaluate import load_test_data, rebuild_model, mahalanobis_scores, energy_score

SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["MLPBaseline", "JointCBM_g0", "JointCBM_g0.1", "JointCBM_g0.5",
            "JointCBM_g1.0", "SequentialCBM", "HybridCBM"]
N = 5000
base = lambda v: "JointCBM" if v.startswith("JointCBM") else v


def auroc(sk, su):
    return float(roc_auc_score(np.r_[np.zeros(len(sk)), np.ones(len(su))], np.r_[sk, su]))


def synthetic(Xv, kind, rng):
    if kind == "perm":
        return np.stack([rng.permutation(Xv[:, j]) for j in range(Xv.shape[1])], 1)
    i, j = rng.integers(0, len(Xv), len(Xv)), rng.integers(0, len(Xv), len(Xv))
    return 0.5 * (Xv[i] + Xv[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ctuk", "cicr", "ton"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for ds in args.datasets:
        rng = np.random.default_rng(0)
        (Xk, _, _), (Xtr, ytr, _), Xun, Xv, _ = load_test_data(ds)
        pick = lambda X, n: X if len(X) <= n else X[rng.choice(len(X), n, replace=False)]
        if len(Xtr) > 20000:
            i = rng.choice(len(Xtr), 20000, replace=False); Xtr, ytr = Xtr[i], ytr[i]
        Xk, Xun, Xv = pick(Xk, N), pick(Xun, N), pick(Xv, 2 * N)
        Xv_ref, Xv_src = Xv[:N], Xv[N:]                     # disjoint halves: inliers vs outlier source
        syn = {k: synthetic(Xv_src, k, rng) for k in ("perm", "mix")}
        for v in VARIANTS:
            for s in SEEDS:
                ck = torch.load(ROOT / "results" / "cbm" / "ms" / f"{ds}_{v}_s{s}.pt", map_location=dev, weights_only=False)
                m = rebuild_model(base(v), ck["n_features"], ck["n_classes"], ck["n_concepts"]).to(dev)
                m.load_state_dict(ck["model_state_dict"]); m.eval()
                T = lambda X: torch.tensor(X, dtype=torch.float32, device=dev)
                with torch.no_grad():
                    emb = lambda X: m.get_embedding(T(X)).cpu().numpy()
                    lg = lambda X: m(T(X))[0].cpu().numpy()
                    etr = emb(Xtr)
                    score = {"mahal": lambda X: mahalanobis_scores(etr, emb(X), train_labels=ytr),
                             "energy": lambda X: energy_score(lg(X))}
                    r = dict(dataset=ds, variant=v, seed=s)
                    for sc, f in score.items():
                        r[f"true_{sc}"] = auroc(f(Xk), f(Xun))
                        ref = f(Xv_ref)
                        for k, Xs in syn.items():
                            r[f"{k}_{sc}"] = auroc(ref, f(Xs))
                rows.append(r)
            print(f"  {ds} {v} done", flush=True)
    out = ROOT / "results" / "cbm" / f"scorer_selection_{'_'.join(args.datasets)}.json"
    out.write_text(json.dumps(rows, indent=2)); print("wrote", out)

    print("\n=== per (dataset, variant): true AUROC (mahal / energy), synthetic AUROC, choice vs truth (5 seeds) ===")
    for k in ("perm", "mix"):
        hit = tot = 0; regret = []
        print(f"\n[{k}]")
        for ds in args.datasets:
            for v in VARIANTS:
                rs = [r for r in rows if r["dataset"] == ds and r["variant"] == v]
                ok = 0
                for r in rs:
                    truth = "mahal" if r["true_mahal"] >= r["true_energy"] else "energy"
                    choice = "mahal" if r[f"{k}_mahal"] >= r[f"{k}_energy"] else "energy"
                    ok += truth == choice; regret.append(max(r["true_mahal"], r["true_energy"]) - r[f"true_{choice}"])
                hit += ok; tot += len(rs)
                tm = np.mean([r["true_mahal"] for r in rs]); te = np.mean([r["true_energy"] for r in rs])
                sm = np.mean([r[f"{k}_mahal"] for r in rs]); se = np.mean([r[f"{k}_energy"] for r in rs])
                print(f"  {ds:5s} {v:14s} true {tm:.3f}/{te:.3f}  synth {sm:.3f}/{se:.3f}  correct {ok}/{len(rs)}")
        print(f"  overall correct {hit}/{tot}; mean AUROC lost to wrong choices {np.mean(regret):.3f}")
    for name, pickfn in (("always mahal", lambda r: "mahal"), ("always energy", lambda r: "energy")):
        hit = sum((("mahal" if r["true_mahal"] >= r["true_energy"] else "energy") == pickfn(r)) for r in rows)
        print(f"  baseline {name}: correct {hit}/{len(rows)}")


if __name__ == "__main__":
    main()
