# Training-time K_FINAL sweep: retrain NeSy-NIDS annealed to different final gate
# steepness and measure how activation-space OOD detectability responds.
#
# For each (dataset, k_final, seed) we retrain (monkeypatching the annealing target),
# then evaluate with the paper's HARD-activation per-class Mahalanobis scorer
# (k-invariant, reflects the learned thresholds) plus continuity metrics: soft
# Mahalanobis at k=k_final, energy AUROC, crispness, within-class variance, F1.
#
# Env overrides for smoke tests: KS_DATASETS, KS_KFINALS, KS_SEEDS, KS_EPOCHS.
# Writes results/nesy/ksweep_train.json incrementally.

import os, sys, json, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import nesy.train as T
from nesy.model import NeSyNIDS
from nesy.evaluate import (
    load_known_split, load_unknown_test,
    _mahalanobis_scores_perclass, _energy_score,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = os.environ.get("KS_DATASETS", "ctu,cic").split(",")
K_FINALS = [float(x) for x in os.environ.get("KS_KFINALS", "2,5,10,20").split(",")]
SEEDS    = [int(x) for x in os.environ.get("KS_SEEDS", "0,1,2,3,4").split(",")]
EPOCHS   = int(os.environ.get("KS_EPOCHS", "60"))
OUT = ROOT / "results" / "nesy" / "ksweep_train.json"
RNG = np.random.default_rng(0)


def sub(X, y=None, per_class=4000, total=8000):
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
def soft_act(model, X, k):
    return torch.stack([r(X, float(k), use_ste=False) for r in model.rule_bank.rules], 1).cpu().numpy()


@torch.no_grad()
def hard_act(model, X):
    return model.get_embedding(X, k=10.0).cpu().numpy()  # STE hard, k-invariant


@torch.no_grad()
def logits_at_k(model, X, k):
    lg, _ = model(X, k=float(k)); return lg.cpu().numpy()


def wvar(A, y):
    return float(np.mean([A[y == c].var(axis=0).mean() for c in np.unique(y)]))


def evaluate(model, ds_cache, k_final):
    Xtr, ytr, Xkn, ykn, Xun = ds_cache
    auroc_hard = roc_auc_score(
        np.r_[np.zeros(len(Xkn)), np.ones(len(Xun))],
        np.r_[_mahalanobis_scores_perclass(hard_act(model, Xtr), ytr, hard_act(model, Xkn)),
              _mahalanobis_scores_perclass(hard_act(model, Xtr), ytr, hard_act(model, Xun))])
    Atr, Akn, Aun = soft_act(model, Xtr, k_final), soft_act(model, Xkn, k_final), soft_act(model, Xun, k_final)
    auroc_soft = roc_auc_score(
        np.r_[np.zeros(len(Xkn)), np.ones(len(Xun))],
        np.r_[_mahalanobis_scores_perclass(Atr, ytr, Akn), _mahalanobis_scores_perclass(Atr, ytr, Aun)])
    en = np.r_[_energy_score(logits_at_k(model, Xkn, k_final)), _energy_score(logits_at_k(model, Xun, k_final))]
    auroc_en = roc_auc_score(np.r_[np.zeros(len(Xkn)), np.ones(len(Xun))], en)
    crisp = float(((Akn < 0.1) | (Akn > 0.9)).mean())
    f1 = f1_score(ykn, logits_at_k(model, Xkn, k_final).argmax(1), average="weighted", zero_division=0)
    return dict(auroc_hard=float(auroc_hard), auroc_soft=float(auroc_soft), auroc_energy=float(auroc_en),
                crisp=crisp, wvar_hard=wvar(hard_act(model, Xtr), ytr), wvar_soft=wvar(Atr, ytr), f1=float(f1))


def load_cache(ds):
    Xtr, ytr, _ = load_known_split(ds, DEVICE, "train")
    Xkn, ykn, _ = load_known_split(ds, DEVICE, "test_known")
    Xun = load_unknown_test(ds, DEVICE, n_sample=10000)
    Xtr_np, ytr_np = sub(Xtr.cpu().numpy(), ytr.cpu().numpy())
    Xkn_np, ykn_np = sub(Xkn.cpu().numpy(), ykn.cpu().numpy(), per_class=2500, total=5000)
    Xun_np = sub(Xun.cpu().numpy(), total=5000)
    return (torch.tensor(Xtr_np, device=DEVICE), ytr_np,
            torch.tensor(Xkn_np, device=DEVICE), ykn_np, torch.tensor(Xun_np, device=DEVICE))


def main():
    T.N_EPOCHS = EPOCHS
    results = {}
    if OUT.exists():
        results = json.loads(OUT.read_text())
    t0 = time.time()
    for ds in DATASETS:
        print(f"\n########## dataset={ds} ##########", flush=True)
        cache = load_cache(ds)
        Xtr, ytr, Xval, yval, n_feat, n_cls, rules = T.load_dataset(ds)
        loader_tr = T.make_loader(Xtr, ytr, DEVICE, shuffle=True)
        loader_val = T.make_loader(Xval, yval, DEVICE, shuffle=False)
        for kf in K_FINALS:
            T.K_FINAL = kf  # anneal target -> get_k()
            for seed in SEEDS:
                key = f"{ds}|k{kf}|s{seed}"
                if key in results:
                    print(f"  skip {key} (done)", flush=True); continue
                torch.manual_seed(seed); np.random.seed(seed)
                model = NeSyNIDS(n_feat, n_cls, rules).to(DEVICE)
                model, valf1 = T.train_nesy(model, loader_tr, loader_val, DEVICE, key)
                m = evaluate(model, cache, kf); m["val_f1"] = float(valf1)
                results[key] = m
                OUT.write_text(json.dumps(results, indent=2))
                print(f"  DONE {key}: F1={m['f1']:.4f} AUROC_hard={m['auroc_hard']:.3f} "
                      f"AUROC_soft={m['auroc_soft']:.3f} energy={m['auroc_energy']:.3f} "
                      f"crisp={m['crisp']:.3f} wvar_hard={m['wvar_hard']:.4f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    # aggregate mean/std over seeds
    agg = {}
    for ds in DATASETS:
        for kf in K_FINALS:
            rows = [results[k] for k in results if k.startswith(f"{ds}|k{kf}|")]
            if not rows: continue
            agg[f"{ds}|k{kf}"] = {m: [float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows])), len(rows)]
                                  for m in ["f1","auroc_hard","auroc_soft","auroc_energy","crisp","wvar_hard","wvar_soft"]}
    results["_aggregate"] = agg
    OUT.write_text(json.dumps(results, indent=2))
    print("\n===== AGGREGATE (mean over seeds) =====", flush=True)
    for k, v in agg.items():
        print(f"  {k}: F1={v['f1'][0]:.4f} AUROC_hard={v['auroc_hard'][0]:.3f}±{v['auroc_hard'][1]:.3f} "
              f"AUROC_soft={v['auroc_soft'][0]:.3f} energy={v['auroc_energy'][0]:.3f} crisp={v['crisp'][0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
