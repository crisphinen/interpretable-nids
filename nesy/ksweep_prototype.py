# PROTOTYPE (not a paper driver): eval-time k-sweep to test whether within-class
# rule-activation variance governs activation-space OOD detectability.
#
# The paper's OOD scorer uses HARD binary activations, which are k-invariant at
# eval (sigmoid(k(x-θ))>0.5 iff x>θ). To manipulate the mechanism variable
# (within-class activation variance) without retraining, we score on the SOFT
# activations at a swept steepness k, using the existing seed-0 checkpoints.
#
# Prediction: activation-space Mahalanobis AUROC tracks within-class soft-activation
# variance (inverted-U in k); CIC (which fails at k=10 because rules saturate)
# should RECOVER as k is lowered; energy-on-logits should be far flatter.

import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from nesy.evaluate import (
    load_model, load_known_split, load_unknown_test,
    _mahalanobis_scores_perclass, _energy_score,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_GRID = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
RNG = np.random.default_rng(0)


def subsample(X, y=None, per_class=4000, total=8000):
    if y is None:
        idx = RNG.choice(len(X), min(total, len(X)), replace=False)
        return X[idx]
    keep = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        keep.extend(RNG.choice(ci, min(per_class, len(ci)), replace=False))
    keep = np.array(keep)
    return X[keep], y[keep]


@torch.no_grad()
def soft_activations(model, X, k):
    # (N, R) soft rule activations at steepness k (no STE, no importance weighting)
    cols = [rule(X, float(k), use_ste=False) for rule in model.rule_bank.rules]
    return torch.stack(cols, dim=1).cpu().numpy()


@torch.no_grad()
def logits_at_k(model, X, k):
    lg, _ = model(X, k=float(k))
    return lg.cpu().numpy()


def within_class_var(A, y):
    # mean over classes of (mean over rule dims of per-class variance)
    return float(np.mean([A[y == c].var(axis=0).mean() for c in np.unique(y)]))


def crispness(A):  # fraction of (flow,rule) soft activations within 0.1 of {0,1}
    return float(((A < 0.1) | (A > 0.9)).mean())


def run(dataset):
    model, _ = load_model(ROOT / "results" / "nesy" / f"{dataset}_nesy_s0.pt", DEVICE)
    Xtr, ytr, _ = load_known_split(dataset, DEVICE, "train")
    Xkn, ykn, _ = load_known_split(dataset, DEVICE, "test_known")
    Xun = load_unknown_test(dataset, DEVICE, n_sample=10000)

    Xtr_np, ytr_np = subsample(Xtr.cpu().numpy(), ytr.cpu().numpy())
    Xkn_np, ykn_np = subsample(Xkn.cpu().numpy(), ykn.cpu().numpy(), per_class=2500, total=5000)
    Xun_np = subsample(Xun.cpu().numpy(), total=5000)
    Xtr_t = torch.tensor(Xtr_np, device=DEVICE); Xkn_t = torch.tensor(Xkn_np, device=DEVICE)
    Xun_t = torch.tensor(Xun_np, device=DEVICE)
    y_ood = np.r_[np.zeros(len(Xkn_np)), np.ones(len(Xun_np))]

    rows = []
    for k in K_GRID:
        Atr = soft_activations(model, Xtr_t, k)
        Akn = soft_activations(model, Xkn_t, k)
        Aun = soft_activations(model, Xun_t, k)
        s_kn = _mahalanobis_scores_perclass(Atr, ytr_np, Akn)
        s_un = _mahalanobis_scores_perclass(Atr, ytr_np, Aun)
        auroc_maha = roc_auc_score(y_ood, np.r_[s_kn, s_un])

        lg_kn = logits_at_k(model, Xkn_t, k); lg_un = logits_at_k(model, Xun_t, k)
        e = np.r_[_energy_score(lg_kn), _energy_score(lg_un)]
        auroc_en = roc_auc_score(y_ood, e)

        f1 = f1_score(ykn_np, logits_at_k(model, Xkn_t, k).argmax(1), average="weighted", zero_division=0)
        rows.append(dict(k=k, wvar=within_class_var(Atr, ytr_np), crisp=crispness(Akn),
                         auroc_maha=float(auroc_maha), auroc_energy=float(auroc_en), f1=float(f1)))
    return rows


if __name__ == "__main__":
    out = {}
    for ds in ["ctu", "cic"]:
        rows = run(ds); out[ds] = rows
        print(f"\n===== {ds.upper()}  (soft-activation eval-time k-sweep, seed 0) =====")
        print(f"{'k':>5} {'within_var':>11} {'crispness':>10} {'AUROC_maha':>11} {'AUROC_energy':>13} {'F1_known':>9}")
        for r in rows:
            print(f"{r['k']:>5} {r['wvar']:>11.4f} {r['crisp']:>10.3f} {r['auroc_maha']:>11.3f} {r['auroc_energy']:>13.3f} {r['f1']:>9.4f}")
        km = max(rows, key=lambda r: r['auroc_maha'])
        print(f"  peak activation-Mahalanobis AUROC = {km['auroc_maha']:.3f} at k={km['k']} "
              f"(k=10 baseline = {[r for r in rows if r['k']==10][0]['auroc_maha']:.3f})")
    Path("/tmp/claude-1002/-home-Ngari-Research/49e29f2b-be51-452f-a13c-f21bafc8287b/scratchpad/ksweep.json").write_text(json.dumps(out, indent=2))
