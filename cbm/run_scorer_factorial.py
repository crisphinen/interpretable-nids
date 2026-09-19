# B: scorer x architecture factorial. On the same per-seed checkpoints behind the
# validated table (results/cbm/ms/), score OOD with every scorer used in Table V
# (Mahalanobis, Hamming, Jaccard, energy, z-normalised Mahalanobis+energy) on each
# CBM/MLP architecture x dataset x seed. The point: which scorer wins is not
# consistent across architectures or datasets, a second independent leg of the
# "no reliable rule" result. Hamming/Jaccard are only defined on the binary-ish
# concept space, not the MLP penultimate. Writes results/cbm/scorer_factorial.json.

import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from statistics import mean, pstdev

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from cbm.evaluate import load_test_data, rebuild_model, mahalanobis_scores, energy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["MLPBaseline", "JointCBM_g0", "JointCBM_g0.1", "JointCBM_g0.5",
            "JointCBM_g1.0", "SequentialCBM", "HybridCBM"]
RNG = np.random.default_rng(0)
base = lambda v: "JointCBM" if v.startswith("JointCBM") else v


def centroid_scores(etr, ytr, ete, kind):
    B = (etr > 0.5).astype(np.float32)
    cents = {c: (B[ytr == c].mean(0) > 0.5).astype(np.float32) for c in np.unique(ytr)}
    T = (ete > 0.5).astype(np.float32)
    ds = []
    for v in cents.values():
        if kind == "hamming":
            ds.append((T != v).sum(1))
        else:
            inter = (T * v).sum(1); union = ((T + v) > 0).sum(1)
            ds.append(1 - inter / np.maximum(union, 1))
    return np.stack(ds, 1).min(1)


def znorm(s, ref):
    m, sd = ref.mean(), ref.std() + 1e-9
    return (s - m) / sd


def auroc(sk, su):
    return float(roc_auc_score(np.r_[np.zeros(len(sk)), np.ones(len(su))], np.r_[sk, su]))


def cap(a, n=5000):
    return a if len(a) <= n else a[RNG.choice(len(a), n, replace=False)]


def main():
    rows = []
    for ds in ["ctu", "cic"]:
        (Xk, yk, _), (Xtr, ytr, _), Xun, _, _ = load_test_data(ds)
        if len(Xtr) > 20000:
            i = RNG.choice(len(Xtr), 20000, replace=False); Xtr, ytr = Xtr[i], ytr[i]
        Xk, Xun = cap(Xk), cap(Xun)
        for v in VARIANTS:
            interp = v != "MLPBaseline"
            for s in SEEDS:
                ck = torch.load(ROOT / "results" / "cbm" / "ms" / f"{ds}_{v}_s{s}.pt",
                                map_location=DEVICE, weights_only=False)
                m = rebuild_model(base(v), ck["n_features"], ck["n_classes"], ck["n_concepts"]).to(DEVICE)
                m.load_state_dict(ck["model_state_dict"]); m.eval()
                with torch.no_grad():
                    E = lambda X: m.get_embedding(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()
                    L = lambda X: m(torch.tensor(X, dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
                    etr, ekn, eun = E(Xtr), E(Xk), E(Xun)
                    lkn, lun = L(Xk), L(Xun)
                sc = {"mahal": auroc(mahalanobis_scores(etr, ekn, train_labels=ytr),
                                     mahalanobis_scores(etr, eun, train_labels=ytr)),
                      "energy": auroc(energy_score(lkn), energy_score(lun))}
                mk = mahalanobis_scores(etr, ekn, train_labels=ytr); mu = mahalanobis_scores(etr, eun, train_labels=ytr)
                ek = energy_score(lkn); eu = energy_score(lun)
                sc["comb"] = auroc(znorm(mk, mk) + znorm(ek, ek), znorm(mu, mk) + znorm(eu, ek))
                if interp:
                    for kind in ("hamming", "jaccard"):
                        sc[kind] = auroc(centroid_scores(etr, ytr, ekn, kind), centroid_scores(etr, ytr, eun, kind))
                rows.append(dict(dataset=ds, variant=v, seed=s, interp=interp, **{f"auroc_{k}": val for k, val in sc.items()}))
        print(f"  {ds} done", flush=True)
    (ROOT / "results" / "cbm" / "scorer_factorial.json").write_text(json.dumps(rows, indent=2))

    SC = ["mahal", "hamming", "jaccard", "energy", "comb"]
    print("\n=== OOD AUROC by scorer x architecture (mean over 5 seeds); [best] per row ===")
    print(f"{'dataset':>7} {'architecture':>14} " + " ".join(f"{s:>8}" for s in SC) + "   best")
    for ds in ["ctu", "cic"]:
        for v in VARIANTS:
            rs = [r for r in rows if r["dataset"] == ds and r["variant"] == v]
            vals = {s: (mean(r[f"auroc_{s}"] for r in rs) if all(f"auroc_{s}" in r for r in rs) else None) for s in SC}
            best = max((s for s in SC if vals[s] is not None), key=lambda s: vals[s])
            cells = " ".join((f"{vals[s]:>8.3f}" if vals[s] is not None else f"{'--':>8}") for s in SC)
            print(f"{ds:>7} {v:>14} {cells}   {best}")
    # is the best scorer consistent?
    print("\n=== best scorer per (dataset, architecture): consistency check ===")
    from collections import Counter
    best_by = {}
    for ds in ["ctu", "cic"]:
        for v in VARIANTS:
            rs = [r for r in rows if r["dataset"] == ds and r["variant"] == v]
            vals = {s: (mean(r[f"auroc_{s}"] for r in rs) if all(f"auroc_{s}" in r for r in rs) else -1) for s in SC}
            best_by[(ds, v)] = max(SC, key=lambda s: vals[s])
    print("  best-scorer counts:", dict(Counter(best_by.values())))
    print("  distinct best scorers across the grid:", len(set(best_by.values())))


if __name__ == "__main__":
    main()
