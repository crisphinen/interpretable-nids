#!/usr/bin/env python3
# Score every NeSy-NIDS checkpoint on exactly the evaluation flows of the CBM/MLP rows of the main
# results table, so all its rows share one set of flows:
#   rows    the training, known-test and unknown-test rows of cbm.evaluate.load_test_data
#           (verified below against that function), with NeSy-NIDS's own input columns
#   AUROC   np.random.seed(42), then the 20,000-flow fit subset and the 5,000 known / 5,000
#           unknown subsets drawn exactly as cbm.evaluate.ood_auroc draws them after
#           cbm.run_multiseed seeds the generator
#   TPR     the next 20,000-flow fit subset, threshold at the 95th percentile of all known-test
#           scores, TPR over all unknown flows (cbm.evaluate.tpr_at_fpr)
# Scorers are NeSy-NIDS's own: per-class Mahalanobis on the hard rule activations, plus energy,
# Hamming/Jaccard to the class centroid and the z-normalised Mahalanobis+energy average.
# Writes results/nesy/shared_eval_{ds}.json.
#
# usage: python -m nesy.run_shared_eval --datasets ctuk ctuko cicr --device cuda
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from dsets import is_ctu, data_dir, nesy_feature_cols
from cbm.evaluate import load_test_data
from nesy.evaluate import load_model, K_EVAL, RESULTS_DIR, _mahalanobis_scores_perclass
from nesy.scorer_ablation import _centroid_dists

TAGS = [("", "main"), ("_a0.05", "a0.05"), ("_a0.1", "a0.1"), ("_a0.2", "a0.2"), ("_a0.5", "a0.5"),
        ("_a1.0", "a1.0"), ("_k2", "k2"), ("_k5", "k5"), ("_k20", "k20")]


def rows(ds, cols):
    d = data_dir(ds); v = json.loads((d / "vocab.json").read_text())
    known = v["known_ids"] if is_ctu(ds) else [v["label_to_id"][c] for c in v["known_classes"]]
    idm = {o: i for i, o in enumerate(sorted(known))}
    def kn(split):
        df = pd.read_parquet(d / f"{split}.parquet"); df = df[df["label_id"].isin(known)].reset_index(drop=True)
        return df[cols].values.astype(np.float32), np.array([idm[l] for l in df["label_id"]])
    Xtr, ytr = kn("train"); Xk, yk = kn("test_known")
    if is_ctu(ds):
        df = pd.read_parquet(d / "test_unknown.parquet")
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42).reset_index(drop=True)
        Xu = df[cols].values.astype(np.float32)
    else:
        ids = pd.read_parquet(d / "test_unknown.parquet", columns=["label_id"])
        rng = np.random.default_rng(42); parts = []
        for lid in ids["label_id"].unique():
            idx = np.where(ids["label_id"].values == lid)[0]
            parts.append(rng.choice(idx, 2000, replace=False) if len(idx) > 2000 else idx)
        sel = np.sort(np.concatenate(parts))
        Xu = pd.read_parquet(d / "test_unknown.parquet", columns=cols).iloc[sel][cols].values.astype(np.float32)
    return Xtr, ytr, Xk, yk, Xu


def subsets(n_tr, n_k, n_u):
    np.random.seed(42)
    it = np.random.choice(n_tr, 20000, replace=False) if n_tr > 20000 else np.arange(n_tr)
    ik = np.random.choice(n_k, 5000, replace=False) if n_k > 5000 else np.arange(n_k)
    iu = np.random.choice(n_u, 5000, replace=False) if n_u > 5000 else np.arange(n_u)
    it2 = np.random.choice(n_tr, 20000, replace=False) if n_tr > 20000 else np.arange(n_tr)
    return it, ik, iu, it2


def auroc(sk, su):
    return float(roc_auc_score(np.r_[np.zeros(len(sk)), np.ones(len(su))], np.r_[sk, su]))


def energy(logits):
    m = logits.max(1, keepdims=True)
    return -(np.log(np.exp(logits - m).sum(1)) + m[:, 0])


def run(ds, dev):
    # check the row selection against the CBM pipeline on its own columns
    (Xk_c, _, _), (Xtr_c, _, _), Xu_c, _, _ = load_test_data(ds)
    v = json.loads((data_dir(ds) / "vocab.json").read_text())
    a = rows(ds, v["feature_cols"])
    assert np.array_equal(a[0], Xtr_c) and np.array_equal(a[2], Xk_c) and np.array_equal(a[4], Xu_c), "row mismatch"
    Xtr, ytr, Xk, yk, Xu = rows(ds, nesy_feature_cols(ds) if is_ctu(ds) else v["feature_cols"])
    it, ik, iu, it2 = subsets(len(Xtr), len(Xk), len(Xu))
    T = lambda X: torch.tensor(X, dtype=torch.float32, device=dev)
    out = []
    for model_name in ("nesy", "rule_only"):
        for s in range(5):
            for tag, label in TAGS:
                f = RESULTS_DIR / f"{ds}_{model_name}_s{s}{tag}.pt"
                if not f.exists() or (model_name == "rule_only" and tag):
                    continue
                m, _ = load_model(f, dev); m.eval()
                with torch.no_grad():
                    H = lambda X: m.get_embedding(T(X), k=K_EVAL).cpu().numpy()
                    htr, hk, hu = H(Xtr), H(Xk), H(Xu)
                    lk = m(T(Xk[ik]), k=K_EVAL)[0].cpu().numpy(); lu = m(T(Xu[iu]), k=K_EVAL)[0].cpu().numpy()
                mk = _mahalanobis_scores_perclass(htr[it], ytr[it], hk[ik])
                mu = _mahalanobis_scores_perclass(htr[it], ytr[it], hu[iu])
                ek, eu = energy(lk), energy(lu)
                z = lambda x, r: (x - r.mean()) / (r.std() + 1e-9)
                tk = _mahalanobis_scores_perclass(htr[it2], ytr[it2], hk)
                tu = _mahalanobis_scores_perclass(htr[it2], ytr[it2], hu)
                r = {"model": model_name, "variant": label, "seed": s,
                     "auroc_mahal": auroc(mk, mu), "tpr5": float((tu > np.percentile(tk, 95)).mean()),
                     "auroc_energy": auroc(ek, eu), "auroc_comb": auroc(z(mk, mk) + z(ek, ek), z(mu, mk) + z(eu, ek)),
                     "auroc_hamming": auroc(_centroid_dists(htr[it], ytr[it], hk[ik], "hamming"),
                                            _centroid_dists(htr[it], ytr[it], hu[iu], "hamming")),
                     "auroc_jaccard": auroc(_centroid_dists(htr[it], ytr[it], hk[ik], "jaccard"),
                                            _centroid_dists(htr[it], ytr[it], hu[iu], "jaccard"))}
                out.append(r)
                print(f"  [{ds}] {model_name} {label} s{s}: AUROC {r['auroc_mahal']:.4f} TPR {r['tpr5']:.4f} "
                      f"energy {r['auroc_energy']:.4f}", flush=True)
    (RESULTS_DIR / f"shared_eval_{ds}.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ctuk", "ctuko", "cicr"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for ds in args.datasets:
        run(ds, torch.device(args.device if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
