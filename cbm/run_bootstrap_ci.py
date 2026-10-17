#!/usr/bin/env python3
# 95% confidence intervals for the AUROC comparisons the paper states (main results, scorer,
# concept-fidelity and bottleneck paragraphs). Hierarchical bootstrap over the two sources of
# variation: seeds (resampled with replacement, independently per model) and evaluation flows
# (the 5,000 known and 5,000 unknown flows, resampled with replacement and shared by both models
# of a comparison, since every model is scored on the same flows). Each replicate takes the
# difference of the two models' seed-mean AUROCs; the interval is the 2.5-97.5 percentile.
#
# Per-flow scores are recomputed from the saved checkpoints with the scoring protocol of the
# main table (cbm.evaluate.ood_auroc; nesy/run_shared_eval.py for NeSy-NIDS), and each model's
# recomputed AUROC is checked against its stored value before any resampling.
# Writes results/cbm/bootstrap_ci.json.
#
# usage: python -m cbm.run_bootstrap_ci --device cuda
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cbm.model import JointCBM
from cbm.run_multiseed import build
from cbm.train import load_dataset, EMBED_DIM
from cbm.evaluate import load_test_data, mahalanobis_scores, energy_score
from nesy.evaluate import load_model, K_EVAL, _mahalanobis_scores_perclass
from nesy.run_shared_eval import rows as nesy_rows, subsets
from dsets import nesy_feature_cols

RES = ROOT / "results"
B = 2000

# (label, dataset, model A, model B): CI of mean AUROC(A) - mean AUROC(B)
COMPARISONS = [
    ("CTU: NeSy-NIDS - MLP", "ctuk", "NeSy", "MLPBaseline"),
    ("CTU: NeSy-NIDS (alpha) - MLP", "ctuk", "NeSy_a0.1", "MLPBaseline"),
    ("CTU: SequentialCBM - MLP", "ctuk", "SequentialCBM", "MLPBaseline"),
    ("CTU: JointCBM gamma=1.0 - gamma=0", "ctuk", "JointCBM_g1.0", "JointCBM_g0"),
    ("CTU: JointCBM gamma=0 - 8-unit bottleneck", "ctuk", "JointCBM_g0", "Bottleneck8"),
    ("CIC: MLP - JointCBM gamma=0", "cicr", "MLPBaseline", "JointCBM_g0"),
    ("CIC: JointCBM gamma=1.0 - gamma=0", "cicr", "JointCBM_g1.0", "JointCBM_g0"),
    ("CIC: 8-unit bottleneck - JointCBM gamma=0", "cicr", "Bottleneck8", "JointCBM_g0"),
    ("CIC: NeSy-NIDS energy - Mahalanobis", "cicr", "NeSy_energy", "NeSy"),
    ("TON: JointCBM gamma=0 - MLP", "ton", "JointCBM_g0", "MLPBaseline"),
    ("TON: JointCBM gamma=0 - 8-unit bottleneck", "ton", "JointCBM_g0", "Bottleneck8"),
]


def stored(ds, model, seed):
    if model == "Bottleneck8":
        return [r for r in json.loads((RES / f"cbm/bottleneck8_{ds}.json").read_text()) if r["seed"] == seed][0]["auroc"]
    if model.startswith("NeSy"):
        var = "a0.1" if model == "NeSy_a0.1" else "main"
        key = "auroc_energy" if model == "NeSy_energy" else "auroc_mahal"
        r = [r for r in json.loads((RES / f"nesy/shared_eval_{ds}.json").read_text())
             if r["model"] == "nesy" and r["variant"] == var and r["seed"] == seed][0]
        return r[key]
    return [r for r in json.loads((RES / f"cbm/multiseed_{ds}_v2_seeds.json").read_text())
            if r["variant"] == model and r["seed"] == seed][0]["auroc"]


def cbm_scores(ds, model, seed, dev, data):
    """Known and unknown scores on the main-table flows, exactly as cbm.evaluate.ood_auroc."""
    Xtr, ytr, Xk, Xu, nf, nk, nc = data
    if model == "Bottleneck8":
        m, f = JointCBM(nf, nc, nk, EMBED_DIM), RES / f"cbm/ms/{ds}_Bottleneck8_s{seed}.pt"
    else:
        m = build("JointCBM" if model.startswith("JointCBM") else model, nf, nc, nk)
        f = RES / f"cbm/ms/{ds}_{model}_s{seed}.pt"
    m.load_state_dict(torch.load(f, map_location=dev)["model_state_dict"]); m.to(dev).eval()
    T = lambda X: torch.tensor(X, dtype=torch.float32, device=dev)
    with torch.no_grad():
        etr, ek, eu = (m.get_embedding(T(X)).cpu().numpy() for X in (Xtr, Xk, Xu))
    np.random.seed(42)
    it = np.random.choice(len(etr), 20000, replace=False) if len(etr) > 20000 else np.arange(len(etr))
    sk = mahalanobis_scores(etr[it], ek, train_labels=ytr[it])
    su = mahalanobis_scores(etr[it], eu, train_labels=ytr[it])
    if len(sk) > 5000: sk = sk[np.random.choice(len(sk), 5000, replace=False)]
    if len(su) > 5000: su = su[np.random.choice(len(su), 5000, replace=False)]
    return sk, su


def nesy_scores(ds, model, seed, dev, nd):
    Xtr, ytr, Xk, yk, Xu = nd
    tag = "_a0.1" if model == "NeSy_a0.1" else ""
    m, _ = load_model(RES / f"nesy/{ds}_nesy_s{seed}{tag}.pt", dev); m.eval()
    it, ik, iu, _ = subsets(len(Xtr), len(Xk), len(Xu))
    T = lambda X: torch.tensor(X, dtype=torch.float32, device=dev)
    with torch.no_grad():
        if model == "NeSy_energy":
            return (energy_score(m(T(Xk[ik]), k=K_EVAL)[0].cpu().numpy()),
                    energy_score(m(T(Xu[iu]), k=K_EVAL)[0].cpu().numpy()))
        htr, hk, hu = (m.get_embedding(T(X), k=K_EVAL).cpu().numpy() for X in (Xtr, Xk[ik], Xu[iu]))
    return (_mahalanobis_scores_perclass(htr[it], ytr[it], hk), _mahalanobis_scores_perclass(htr[it], ytr[it], hu))


def auc(sk, su):
    return roc_auc_score(np.r_[np.zeros(len(sk)), np.ones(len(su))], np.r_[sk, su])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda")
    dev = torch.device(ap.parse_args().device if torch.cuda.is_available() else "cpu")
    cache, data, ndata, out = {}, {}, {}, []
    for label, ds, a, b in COMPARISONS:
        if ds not in data:
            (Xtr, ytr, _), _, nf, nk, nc, _ = load_dataset(ds)
            (Xk, _, _), _, Xu, _, _ = load_test_data(ds)
            data[ds] = (Xtr, ytr, Xk, Xu, nf, nk, nc)
        S = {}
        for model in (a, b):
            if (ds, model) not in cache:
                per_seed = []
                for s in range(5):
                    if model.startswith("NeSy"):
                        if ds not in ndata:
                            ndata[ds] = nesy_rows(ds, nesy_feature_cols(ds))
                        sk, su = nesy_scores(ds, model, s, dev, ndata[ds])
                    else:
                        sk, su = cbm_scores(ds, model, s, dev, data[ds])
                    assert abs(auc(sk, su) - stored(ds, model, s)) < 1e-9, (ds, model, s, auc(sk, su))
                    per_seed.append((sk, su))
                cache[(ds, model)] = per_seed
            S[model] = cache[(ds, model)]
        nk_, nu_ = len(S[a][0][0]), len(S[a][0][1])
        assert all(len(x[0]) == nk_ and len(x[1]) == nu_ for m in (a, b) for x in S[m])
        rng = np.random.default_rng(0)
        point = np.mean([auc(*x) for x in S[a]]) - np.mean([auc(*x) for x in S[b]])
        diffs = np.empty(B)
        for i in range(B):
            ik = rng.integers(0, nk_, nk_); iu = rng.integers(0, nu_, nu_)
            sa = rng.integers(0, 5, 5); sb = rng.integers(0, 5, 5)
            ma = np.mean([auc(S[a][j][0][ik], S[a][j][1][iu]) for j in sa])
            mb = np.mean([auc(S[b][j][0][ik], S[b][j][1][iu]) for j in sb])
            diffs[i] = ma - mb
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out.append({"comparison": label, "dataset": ds, "a": a, "b": b, "diff": float(point),
                    "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0), "B": B})
        print(f"{label:44s} diff {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"{'  *' if lo > 0 or hi < 0 else ''}", flush=True)
    (RES / "cbm/bootstrap_ci.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
