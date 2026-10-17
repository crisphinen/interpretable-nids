#!/usr/bin/env python3
# Non-CBM baseline rows of Table I (tab:main) and tab:kfs, scored on the same evaluation flows
# as cbm.run_multiseed (np.random.seed(42), then cbm.evaluate.ood_auroc / tpr_at_fpr):
#   PostHocCBM  logistic concept probes on the frozen MLPBaseline_s{seed} embedding and a logistic
#               classifier on the probe outputs, both fit on the training split (main paper,
#               Training variants); per-class Mahalanobis on probe outputs
#   DT          depth chosen on val from {5, 10, 15, None}; per-class Mahalanobis on raw features
#               (does not use the tree, so no seed variance)
#   RF          random forest, 100 trees, depth 10 (main paper, Baselines), OOD score 1 - max p
# Writes results/cbm/baselines_{ds}.json (per-seed rows plus mean/std).
#
# usage: python -m cbm.run_baselines --datasets ctuk ctuko cic --device cuda
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from cbm.model import MLPBaseline
from cbm.train import load_dataset, EMBED_DIM
from cbm.evaluate import load_test_data, ood_auroc, tpr_at_fpr

RES = Path(__file__).parent.parent / "results" / "cbm"
SEEDS = range(5)


class Rep:
    """Adapter so a numpy representation goes through the shared ood_auroc / tpr_at_fpr."""
    def __init__(self, fn): self.fn = fn
    def eval(self): return self
    def get_embedding(self, x): return torch.tensor(self.fn(x.cpu().numpy()), dtype=torch.float32)


def mahal_rows(fn, Xtr, ytr, Xte, Xun, device):
    np.random.seed(42)
    m = Rep(fn)
    return {"auroc": float(ood_auroc(m, Xtr, ytr, Xte, Xun, device)),
            "tpr5": float(tpr_at_fpr(m, Xtr, ytr, Xte, Xun, device))}


def score_rows(score, Xtr, Xte, Xun):
    """AUROC / TPR@5%FPR for a direct OOD score, drawing the same subsets as ood_auroc / tpr_at_fpr."""
    np.random.seed(42)
    if len(Xtr) > 20000: np.random.choice(len(Xtr), 20000, replace=False)
    sk, su = score(Xte), score(Xun)
    ik = np.random.choice(len(sk), 5000, replace=False) if len(sk) > 5000 else slice(None)
    iu = np.random.choice(len(su), 5000, replace=False) if len(su) > 5000 else slice(None)
    a, b = sk[ik], su[iu]
    auroc = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])
    return {"auroc": float(auroc), "tpr5": float((su > np.percentile(sk, 95)).mean())}


def f1(y, p): return float(f1_score(y, p, average="weighted", zero_division=0))


def posthoc(ds, s, data, device):
    (Xtr, ytr, Ctr), (Xva, yva, Cva), (Xte, yte, Cte), Xun = data
    ck = torch.load(RES / "ms" / f"{ds}_MLPBaseline_s{s}.pt", map_location=device, weights_only=False)
    mlp = MLPBaseline(ck["n_features"], ck["n_classes"], EMBED_DIM).to(device)
    mlp.load_state_dict(ck["model_state_dict"]); mlp.eval()

    @torch.no_grad()
    def emb(X):
        return np.concatenate([mlp.get_embedding(torch.tensor(X[i:i + 8192], dtype=torch.float32, device=device)).cpu().numpy()
                               for i in range(0, len(X), 8192)])
    Etr = emb(Xtr)
    probes = []
    for j in range(Ctr.shape[1]):
        if len(np.unique(Ctr[:, j])) < 2:
            probes.append(float(Ctr[0, j])); continue
        probes.append(LogisticRegression(max_iter=200, random_state=42).fit(Etr, Ctr[:, j]))

    def concepts(X):
        E = emb(X)
        return np.stack([np.full(len(E), p, np.float32) if isinstance(p, float) else p.predict_proba(E)[:, 1]
                         for p in probes], 1).astype(np.float32)
    head = LogisticRegression(max_iter=500, random_state=42).fit(concepts(Xtr), ytr)
    Pte = concepts(Xte)
    acc = float(np.mean([((Pte[:, j] > 0.5) == Cte[:, j]).mean() for j in range(Cte.shape[1])]))
    return {"test_f1": f1(yte, head.predict(Pte)), "concept_acc": acc,
            **mahal_rows(concepts, Xtr, ytr, Xte, Xun, device)}


def run(ds, device):
    (Xtr, ytr, Ctr), (Xva, yva, Cva), *_ = load_dataset(ds)
    (Xte, yte, Cte), _, Xun, _, _ = load_test_data(ds)
    data = ((Xtr, ytr, Ctr), (Xva, yva, Cva), (Xte, yte, Cte), Xun)
    out = {"dataset": ds, "rows": {}}

    best = max((DecisionTreeClassifier(max_depth=d, random_state=42).fit(Xtr, ytr) for d in (5, 10, 15, None)),
               key=lambda t: f1(yva, t.predict(Xva)))
    dt = {"test_f1": f1(yte, best.predict(Xte)), "depth": int(best.get_depth()),
          **mahal_rows(lambda X: X, Xtr, ytr, Xte, Xun, device)}
    out["rows"]["DT"] = [dt]
    print(f"[{ds}] DT depth={dt['depth']} F1={dt['test_f1']:.4f} AUROC={dt['auroc']:.3f} TPR={dt['tpr5']:.3f}", flush=True)

    for name in ("RF", "PostHocCBM"):
        rows = []
        for s in SEEDS:
            if name == "RF":
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=s).fit(Xtr, ytr)
                r = {"test_f1": f1(yte, rf.predict(Xte)),
                     **score_rows(lambda X: 1 - rf.predict_proba(X).max(1), Xtr, Xte, Xun)}
            else:
                r = posthoc(ds, s, data, device)
            rows.append({"seed": s, **r})
            print(f"[{ds}] {name} s{s} F1={r['test_f1']:.4f} AUROC={r['auroc']:.3f} TPR={r['tpr5']:.3f}", flush=True)
        out["rows"][name] = rows

    out["summary"] = {n: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
                          for k in ("test_f1", "auroc", "tpr5")} for n, rs in out["rows"].items()}
    (RES / f"baselines_{ds}.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ctuk", "ctuko", "cic"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    for ds in args.datasets:
        o = run(ds, device)
        for n, v in o["summary"].items():
            print(f"  {ds} {n:10} F1={v['test_f1'][0]:.4f} AUROC={v['auroc'][0]:.3f}±{v['auroc'][1]:.3f} "
                  f"TPR={v['tpr5'][0]:.3f}±{v['tpr5'][1]:.3f}")


if __name__ == "__main__":
    main()
