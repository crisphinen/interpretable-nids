#!/usr/bin/env python3
# Post-hoc OOD baselines on the unconstrained MLP (Table tab:ood_baselines).
# Scores:
#   MSP    1 - max softmax probability                      [Hendrycks & Gimpel 2017]
#   ODIN   temperature-scaled MSP (T=1000) with input perturbation eps=0.0014
#   Energy -T*logsumexp(logits/T), T=1                       [Liu et al. 2020]
#   kNN    distance to k-th nearest normalised training embedding, k=50
#                                                            [Sun et al. 2022]
#   Mahal  per-class Mahalanobis on the 64-d embedding, fit on train
#                                                            [Lee et al. 2018]
# Protocol as elsewhere: fit on train, AUROC on test_known vs test_unknown
# (5,000 / 5,000 subsample, seed 42). Uses results/cbm/ms/{ds}_MLPBaseline_s0.pt.
#
# usage: python -m cbm.run_ood_baselines --device cuda
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from cbm.model import MLPBaseline
from cbm.train import load_dataset, EMBED_DIM
from cbm.evaluate import load_test_data, mahalanobis_scores

RES = Path(__file__).parent.parent / "results" / "cbm"


def _auroc(s_in, s_out):
    y = np.concatenate([np.zeros(len(s_in)), np.ones(len(s_out))])
    return float(roc_auc_score(y, np.concatenate([s_in, s_out])))


def run(ds, device, seed=0):
    (Xtr, ytr, _), _, nf, nk, nc, _ = load_dataset(ds)
    (Xte, _, _), _, Xun, _, _ = load_test_data(ds)
    ck = torch.load(RES / "ms" / f"{ds}_MLPBaseline_s{seed}.pt", map_location=device, weights_only=False)
    model = MLPBaseline(nf, nk, EMBED_DIM).to(device); model.load_state_dict(ck["model_state_dict"]); model.eval()
    rng = np.random.RandomState(42)
    Xte = Xte[rng.choice(len(Xte), min(5000, len(Xte)), replace=False)]
    Xun = Xun[rng.choice(len(Xun), min(5000, len(Xun)), replace=False)]
    if len(Xtr) > 20000:
        idx = rng.choice(len(Xtr), 20000, replace=False); Xtr, ytr = Xtr[idx], ytr[idx]
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=device)

    @torch.no_grad()
    def logits_emb(X):
        l, _ = model(T(X)); e = model.get_embedding(T(X))
        return l.cpu().numpy(), e.cpu().numpy()

    def odin(X, temp=1000.0, eps=0.0014):
        x = T(X).requires_grad_(True)
        l, _ = model(x)
        loss = F.cross_entropy(l / temp, l.argmax(1))
        g = torch.autograd.grad(loss, x)[0].sign()
        with torch.no_grad():
            l2, _ = model(x - eps * g)
            return (1 - F.softmax(l2 / temp, 1).max(1).values).cpu().numpy()

    l_te, e_te = logits_emb(Xte); l_un, e_un = logits_emb(Xun); _, e_tr = logits_emb(Xtr)
    sm = lambda l: 1 - F.softmax(torch.tensor(l), 1).max(1).values.numpy()
    en = lambda l: -torch.logsumexp(torch.tensor(l), 1).numpy()
    nrm = lambda e: e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)

    def knn(e, k=50):
        d = np.linalg.norm(nrm(e)[:, None, :] - nrm(e_tr)[None, :, :], axis=2)
        return np.sort(d, axis=1)[:, k - 1]

    out = {
        "MSP": _auroc(sm(l_te), sm(l_un)),
        "ODIN": _auroc(odin(Xte), odin(Xun)),
        "Energy": _auroc(en(l_te), en(l_un)),
        "kNN": _auroc(knn(e_te), knn(e_un)),
        "Mahalanobis": _auroc(mahalanobis_scores(e_tr, e_te, train_labels=ytr),
                              mahalanobis_scores(e_tr, e_un, train_labels=ytr)),
    }
    print(ds, {k: round(v, 3) for k, v in out.items()})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    res = {ds: run(ds, device) for ds in ["ctu", "cic"]}
    (RES / "r2_ood_baselines.json").write_text(json.dumps(res, indent=2)); print("wrote r2_ood_baselines.json")


if __name__ == "__main__":
    main()
