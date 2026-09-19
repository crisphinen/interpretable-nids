# A+B for the reviewer's methods critique. For every CBM/MLP checkpoint:
#  (A) a scorer-INDEPENDENT geometry metric of the scoring space: mean within-class
#      variance, and the degeneracy of the pooled within-class covariance S_W
#      (effective rank = participation ratio, and log-det) -- the quantity that
#      makes per-class Mahalanobis well- or ill-posed, measured without any scorer.
#  (B) OOD AUROC under four scorers on the SAME representation (Mahalanobis, Hamming,
#      Jaccard, energy) -- to test scorer x architecture interactions.
# Combined later with the NeSy k-sweep JSON. Writes results/cbm/geometry_factorial.json.

import sys, json, re
from pathlib import Path
import numpy as np
import torch
from numpy.linalg import slogdet, eigvalsh
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from cbm.evaluate import load_test_data, rebuild_model, mahalanobis_scores, energy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = ["MLPBaseline", "JointCBM", "JointCBM_g0.1", "JointCBM_g0.5", "JointCBM_g1.0",
          "SequentialCBM", "HybridCBM"]
RNG = np.random.default_rng(0)


def within_cov(emb, y):
    # pooled within-class covariance S_W and mean within-class variance
    res = np.concatenate([emb[y == c] - emb[y == c].mean(0) for c in np.unique(y)], 0)
    Sw = np.cov(res.T)
    if Sw.ndim == 0:
        Sw = Sw.reshape(1, 1)
    return Sw


def geometry(emb, y):
    Sw = within_cov(emb, y)
    d = Sw.shape[0]
    tr = np.trace(Sw)
    tr2 = np.trace(Sw @ Sw)
    eff_rank = float(tr * tr / tr2) if tr2 > 0 else 1.0          # participation ratio in [1, d]
    within_var = float(np.mean(np.diag(Sw)))                      # mean per-dim within-class variance
    sign, ld = slogdet(Sw + 1e-6 * np.eye(d))
    logdet = float(ld) if sign > 0 else float("-inf")
    ev = eigvalsh(Sw)
    cond = float(ev[-1] / max(ev[0], 1e-12))
    return dict(within_var=within_var, eff_rank=eff_rank, eff_rank_frac=eff_rank / d,
                logdet=logdet, cond=cond, dim=int(d))


def centroid_scores(emb_tr, y_tr, emb_te, kind):
    B = (emb_tr > 0.5).astype(np.float32)
    cents = {c: (B[y_tr == c].mean(0) > 0.5).astype(np.float32) for c in np.unique(y_tr)}
    T = (emb_te > 0.5).astype(np.float32)
    dists = []
    for v in cents.values():
        if kind == "hamming":
            dists.append((T != v).sum(1))
        else:  # jaccard
            inter = (T * v).sum(1); union = ((T + v) > 0).sum(1)
            dists.append(1 - inter / np.maximum(union, 1))
    return np.stack(dists, 1).min(1)


def cap(a, n=5000):
    return a if len(a) <= n else a[RNG.choice(len(a), n, replace=False)]


def run(ds):
    (Xk, yk, _), (Xtr, ytr, _), Xun, _, _ = load_test_data(ds)
    out = []
    for m in MODELS:
        fn = f"{ds}_{m}.pt"
        ck = torch.load(ROOT / "results" / "cbm" / fn, map_location=DEVICE, weights_only=False)
        model = rebuild_model(ck["model_name"], ck["n_features"], ck["n_classes"], ck["n_concepts"]).to(DEVICE)
        model.load_state_dict(ck["model_state_dict"]); model.eval()
        gamma = float(re.search(r"_g([0-9.]+)$", m).group(1)) if "_g" in m else (0.0 if m != "MLPBaseline" else None)
        interp = m != "MLPBaseline"
        with torch.no_grad():
            E = lambda X: model.get_embedding(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()
            L = lambda X: model(torch.tensor(X, dtype=torch.float32, device=DEVICE))[0].cpu().numpy()
            etr, ekn, eun = E(Xtr), E(Xk), E(Xun)
            lkn, lun = L(Xk), L(Xun)
        if len(etr) > 20000:
            idx = RNG.choice(len(etr), 20000, replace=False); etr, ytr_f = etr[idx], ytr[idx]
        else:
            ytr_f = ytr
        geo = geometry(etr, ytr_f)
        ekn_c, eun_c = cap(ekn), cap(eun)
        lab = np.r_[np.zeros(len(ekn_c)), np.ones(len(eun_c))]
        sc = {}
        sc["mahal"] = roc_auc_score(lab, np.r_[mahalanobis_scores(etr, ekn_c, train_labels=ytr_f),
                                                mahalanobis_scores(etr, eun_c, train_labels=ytr_f)])
        sc["energy"] = roc_auc_score(lab, np.r_[energy_score(cap(lkn)), energy_score(cap(lun))])
        if interp:  # binary-simplex scorers only meaningful in the concept space
            for kind in ("hamming", "jaccard"):
                sc[kind] = roc_auc_score(lab, np.r_[centroid_scores(etr, ytr_f, ekn_c, kind),
                                                    centroid_scores(etr, ytr_f, eun_c, kind)])
        row = dict(dataset=ds, model=m.replace("_g", " g="), arch=("MLP" if m == "MLPBaseline" else ("NeSy" if False else "CBM")),
                   gamma=gamma, interp=interp, val_f1=ck.get("val_f1"), **geo,
                   **{f"auroc_{k}": float(v) for k, v in sc.items()})
        out.append(row)
        print(f"  {ds} {m:16s} wvar={geo['within_var']:.4f} eff_rank={geo['eff_rank']:.2f}/{geo['dim']} "
              f"logdet={geo['logdet']:.1f}  maha={sc['mahal']:.3f} energy={sc['energy']:.3f}"
              + (f" hamming={sc['hamming']:.3f} jaccard={sc['jaccard']:.3f}" if interp else ""), flush=True)
    return out


if __name__ == "__main__":
    allrows = []
    for ds in ["ctu", "cic"]:
        print(f"===== {ds.upper()} =====")
        allrows += run(ds)
    (ROOT / "results" / "cbm" / "geometry_factorial.json").write_text(json.dumps(allrows, indent=2))
    print("\nwrote results/cbm/geometry_factorial.json", len(allrows), "rows")
