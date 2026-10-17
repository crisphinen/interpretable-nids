#!/usr/bin/env python3
# Leave-family-out scorer selection (main tab:lofo; supplement, Leave-Family-Out Scorer Selection).
# For every known attack family of CTU (ctuk) and CIC (cicr): drop the family from the known set,
# retrain JointCBM (gamma=0) on the remaining classes (five seeds), and score how well per-class
# Mahalanobis (concept activations) and logit-space energy detect the held-out family; the fold's
# winner is the scorer with the higher AUROC.
# Folds are vocab variants over the base split's rows (data/{base}_lofo_{family}/, parquet files
# symlinked): the held-out family's flows stay in the parquet files but are no longer known, so the
# loaders drop them from training. CTU re-selects its features on each fold's reduced known training
# rows (class-weighted random forest, 300 trees, 99% cumulative importance, as 06_make_kfs_splits.py);
# CIC keeps its fixed list. Held-out flows for scoring: the family's test_known rows; known side:
# the fold's known test rows; 5,000 / 5,000 caps, seed 42, as elsewhere.
# Writes results/cbm/lofo.json.
#
# usage: python -m cbm.run_lofo --make-folds
#        python -m cbm.run_lofo --train --folds ctuk_lofo_ddos ... --device cuda
#        python -m cbm.run_lofo --score --device cuda
import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "cbm"
CAT_IDS = ["conn_state_id", "history_id", "service_id"]
FAMILIES = {"ctuk": ["C&C-HeartBeat", "DDoS", "Okiru"],
            "cicr": ["DDoS-SYN_Flood", "Mirai-greeth_flood", "Recon-PortScan", "VulnerabilityScan"]}
SLUG = {"C&C-HeartBeat": "cchb", "DDoS": "ddos", "Okiru": "okiru", "DDoS-SYN_Flood": "ddossyn",
        "Mirai-greeth_flood": "mirai", "Recon-PortScan": "recon", "VulnerabilityScan": "vuln"}


def fold_names():
    return {f"{b}_lofo_{SLUG[f]}": (b, f) for b, fams in FAMILIES.items() for f in fams}


def make_folds():
    for name, (base, fam) in fold_names().items():
        src, dst = ROOT / "data" / base, ROOT / "data" / name
        dst.mkdir(exist_ok=True)
        for s in ("train", "val", "test_known", "test_unknown"):
            if not (dst / f"{s}.parquet").is_symlink():
                os.symlink(f"../{base}/{s}.parquet", dst / f"{s}.parquet")
        v = json.loads((src / "vocab.json").read_text())
        hold = v["label_to_id"][fam]
        v["known_classes"] = [c for c in v["known_classes"] if c != fam]
        if "known_ids" in v:
            v["known_ids"] = [i for i in v["known_ids"] if i != hold]
        v["lofo"] = {"base": base, "held_out": fam, "held_out_id": hold}
        if base.startswith("ctu"):
            tr = pd.read_parquet(src / "train.parquet")
            tr = tr[tr.label_id.isin(v["known_ids"])]
            cands = [c for c in tr.columns if c not in CAT_IDS + ["label_id", "detailed_label"]]
            rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
            rf.fit(tr[cands].values, tr["label_id"].values)
            imp = pd.Series(rf.feature_importances_, index=cands).sort_values(ascending=False)
            keep = imp.index[:int(((imp.cumsum() / imp.sum()) < 0.99).sum()) + 1].tolist()
            v["feature_cols"] = keep + CAT_IDS
            v["nesy_feature_cols"] = v["feature_cols"]
            v["n_features"] = len(v["feature_cols"])
            v["selection"] = {"method": "class-weighted RF on the fold's reduced known training rows",
                              "rerun_numeric": keep, "n_trees": 300, "rf_seed": 42, "cum_importance": 0.99}
        (dst / "vocab.json").write_text(json.dumps(v, indent=2))
        print(f"{name}: held out {fam}, {len(v['known_classes'])} known, {len(v['feature_cols'])} features")


def train(folds, device):
    from cbm.run_multiseed import run
    for f in folds:
        run(f, list(range(5)), device, ["JointCBM_g0"])


@torch.no_grad()
def score(device):
    from cbm.evaluate import load_test_data, mahalanobis_scores, energy_score, rebuild_model
    from cbm.train import load_dataset
    out = {}
    for name, (base, fam) in fold_names().items():
        if not all((RES / "ms" / f"{name}_JointCBM_g0_s{s}.pt").exists() for s in range(5)):
            print(f"{name}: not fully trained yet, skipped"); continue
        v = json.loads((ROOT / "data" / name / "vocab.json").read_text())
        (Xtr, ytr, _), *_ = load_dataset(name)
        (Xk, _, _), *_ = load_test_data(name)
        tk = pd.read_parquet(ROOT / "data" / name / "test_known.parquet")
        Xh = tk[tk.label_id == v["lofo"]["held_out_id"]][v["feature_cols"]].values.astype(np.float32)
        rng = np.random.default_rng(42)
        cap = lambda X, n: X[rng.choice(len(X), n, replace=False)] if len(X) > n else X
        Xfit, yfit = (Xtr, ytr) if len(Xtr) <= 20000 else (lambda i: (Xtr[i], ytr[i]))(rng.choice(len(Xtr), 20000, replace=False))
        Xk, Xh = cap(Xk, 5000), cap(Xh, 5000)
        lab = np.r_[np.zeros(len(Xk)), np.ones(len(Xh))]
        rows = []
        for s in range(5):
            ck = torch.load(RES / "ms" / f"{name}_JointCBM_g0_s{s}.pt", map_location=device, weights_only=False)
            m = rebuild_model("JointCBM", ck["n_features"], ck["n_classes"], ck["n_concepts"]).to(device)
            m.load_state_dict(ck["model_state_dict"]); m.eval()
            T = lambda X: torch.tensor(X, device=device)
            emb = lambda X: m.get_embedding(T(X)).cpu().numpy()
            efit = emb(Xfit)
            mk, mh = (mahalanobis_scores(efit, emb(X), train_labels=yfit) for X in (Xk, Xh))
            ek, eh = (energy_score(m(T(X))[0].cpu().numpy()) for X in (Xk, Xh))
            a_m, a_e = roc_auc_score(lab, np.r_[mk, mh]), roc_auc_score(lab, np.r_[ek, eh])
            rows.append({"seed": s, "mahal": float(a_m), "energy": float(a_e), "winner": "M" if a_m >= a_e else "E"})
        nm = sum(r["winner"] == "M" for r in rows)
        out[name] = {"base": base, "held_out": fam, "n_held_out": int(len(Xh)), "rows": rows,
                     "mahal": float(np.mean([r["mahal"] for r in rows])),
                     "energy": float(np.mean([r["energy"] for r in rows])), "winner_M": nm, "winner_E": 5 - nm}
        print(f"{base:5s} {fam:20s} Mahal {out[name]['mahal']:.2f}  Energy {out[name]['energy']:.2f}  winner {nm}/{5 - nm}")
    (RES / "lofo.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-folds", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--folds", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.make_folds:
        make_folds()
    if args.train:
        train(args.folds or list(fold_names()), device)
    if args.score:
        score(device)


if __name__ == "__main__":
    main()
