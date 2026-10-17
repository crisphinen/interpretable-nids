#!/usr/bin/env python3
# Open-set (C+1) evaluation (main tab:openset_agg; supplement tab:openset_perfam, tab:confusion).
# Protocol (supplement, Open-set evaluation): per-class Mahalanobis on each model's scoring space
# (CBM concept activations, NeSy hard rule activations, MLP embedding), class means and shared
# covariance fit on <=20,000 training flows; tau = 95th percentile of known-validation scores; a
# flow is "unknown" if its score exceeds tau, else the argmax known-class logit. Evaluation: 5,000
# known test flows (same for all seeds) and up to 3,000 flows per held-out family, every family
# merged into one "unknown" class. Reported per model over seeds 0-4: known rejection rate,
# macro-F1 over C+1, equal-family-weighted AUROC and recall at tau, tau coefficient of variation;
# per-family AUROC/recall; the (C+1) confusion matrix of seed 0.
# Writes results/cbm/openset_{ds}.json.
#
# usage: python -m cbm.run_openset --datasets ctuk cicr --device cuda
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from cbm.evaluate import mahalanobis_scores, rebuild_model
from dsets import data_dir, load_vocab, nesy_feature_cols, is_ctu
from nesy.evaluate import load_model as load_nesy, K_EVAL

ROOT = Path(__file__).resolve().parent.parent
CBM_RES, NESY_RES = ROOT / "results" / "cbm", ROOT / "results" / "nesy"
MODELS = ["MLPBaseline", "JointCBM_g0", "SequentialCBM", "HybridCBM", "NeSy"]
N_KNOWN, UNK_CAP, FIT_CAP = 5000, 3000, 20000


def load_frames(ds):
    v, d = load_vocab(ds), data_dir(ds)
    known_ids = v["known_ids"] if is_ctu(ds) else [v["label_to_id"][c] for c in v["known_classes"]]
    i2l = ({int(k): l for k, l in v["id_to_label"].items()} if "id_to_label" in v
           else {i: l for l, i in v["label_to_id"].items()})
    remap = {orig: i for i, orig in enumerate(sorted(known_ids))}
    rng = np.random.default_rng(42)
    fr = {}
    for s in ("train", "val", "test_known"):
        df = pd.read_parquet(d / f"{s}.parquet")
        df = df[df.label_id.isin(known_ids)].reset_index(drop=True)
        if s == "train" and len(df) > FIT_CAP:
            df = df.iloc[np.sort(rng.choice(len(df), FIT_CAP, replace=False))].reset_index(drop=True)
        if s == "test_known" and len(df) > N_KNOWN:
            df = df.iloc[np.sort(rng.choice(len(df), N_KNOWN, replace=False))].reset_index(drop=True)
        fr[s] = df
    ids = pd.read_parquet(d / "test_unknown.parquet", columns=["label_id"])["label_id"].values
    pick = []
    for lid in np.unique(ids):
        idx = np.where(ids == lid)[0]
        pick.append(np.sort(rng.choice(idx, UNK_CAP, replace=False)) if len(idx) > UNK_CAP else idx)
    fr["unknown"] = pd.read_parquet(d / "test_unknown.parquet").iloc[np.sort(np.concatenate(pick))].reset_index(drop=True)
    for k in ("train", "val", "test_known"):
        fr[k]["y"] = fr[k].label_id.map(remap)
    return fr, [i2l[o] for o in sorted(known_ids)], i2l


def model_fns(ds, name, seed, device):
    """(embedding fn, logits fn, feature columns) for one trained model."""
    if name == "NeSy":
        m, _ = load_nesy(NESY_RES / f"{ds}_nesy_s{seed}.pt", device)
        return (lambda X: m.get_embedding(X, k=K_EVAL), lambda X: m(X, k=K_EVAL)[0], nesy_feature_cols(ds))
    ck = torch.load(CBM_RES / "ms" / f"{ds}_{name}_s{seed}.pt", map_location=device, weights_only=False)
    base = "JointCBM" if name.startswith("JointCBM") else name
    m = rebuild_model(base, ck["n_features"], ck["n_classes"], ck["n_concepts"]).to(device)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m.get_embedding, (lambda X: m(X)[0]), load_vocab(ds)["feature_cols"]


@torch.no_grad()
def evaluate(fr, name, ds, seed, device, n_known):
    emb, logit, cols = model_fns(ds, name, seed, device)
    T = lambda df: torch.tensor(df[cols].values.astype(np.float32), device=device)
    E = {k: emb(T(fr[k])).cpu().numpy() for k in ("train", "val", "test_known", "unknown")}
    sc = {k: mahalanobis_scores(E["train"], E[k], train_labels=fr["train"].y.values) for k in ("val", "test_known", "unknown")}
    tau = float(np.percentile(sc["val"], 95))
    pk = logit(T(fr["test_known"])).argmax(1).cpu().numpy()
    pu = logit(T(fr["unknown"])).argmax(1).cpu().numpy()
    pred = np.r_[np.where(sc["test_known"] > tau, n_known, pk), np.where(sc["unknown"] > tau, n_known, pu)]
    true = np.r_[fr["test_known"].y.values, np.full(len(fr["unknown"]), n_known)]
    fam = {}
    for lid in np.unique(fr["unknown"].label_id):
        su = sc["unknown"][fr["unknown"].label_id.values == lid]
        fam[int(lid)] = {"auroc": float(roc_auc_score(np.r_[np.zeros(len(sc["test_known"])), np.ones(len(su))],
                                                      np.r_[sc["test_known"], su])),
                         "recall": float((su > tau).mean()), "n": int(len(su))}
    return {"seed": seed, "tau": tau, "known_rej": float((sc["test_known"] > tau).mean()),
            "macro_f1": float(f1_score(true, pred, labels=list(range(n_known + 1)), average="macro", zero_division=0)),
            "eqfam_auroc": float(np.mean([f["auroc"] for f in fam.values()])),
            "eqfam_recall": float(np.mean([f["recall"] for f in fam.values()])),
            "families": fam,
            "confusion": confusion_matrix(true, pred, labels=list(range(n_known + 1))).tolist()}


def run(ds, device):
    fr, classes, i2l = load_frames(ds)
    print(f"[{ds}] known test {len(fr['test_known'])}, unknown {len(fr['unknown'])} "
          f"({fr['unknown'].label_id.nunique()} families), fit {len(fr['train'])}, val {len(fr['val'])}")
    out = {"classes": classes + ["unknown"], "family_names": {}, "models": {}}
    for name in MODELS:
        rows = [evaluate(fr, name, ds, s, device, len(classes)) for s in range(5)]
        taus = np.array([r["tau"] for r in rows])
        agg = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))
               for k in ("known_rej", "macro_f1", "eqfam_auroc", "eqfam_recall")}
        agg["tau_cv"] = float(taus.std() / abs(taus.mean()))
        fams = rows[0]["families"].keys()
        agg["per_family"] = {i2l[f]: {k: float(np.mean([r["families"][f][k] for r in rows])) for k in ("auroc", "recall")}
                             for f in fams}
        out["models"][name] = {"rows": rows, **agg}
        print(f"[{ds}] {name:14s} known_rej={100 * agg['known_rej'][0]:.1f}% ({100 * agg['known_rej'][1]:.1f}) "
              f"macroF1={agg['macro_f1'][0]:.3f} ({agg['macro_f1'][1]:.3f}) eqfam AUROC={agg['eqfam_auroc'][0]:.3f} "
              f"recall={agg['eqfam_recall'][0]:.2f} tau_cv={agg['tau_cv']:.2f}")
    (CBM_RES / f"openset_{ds}.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ctuk", "cicr"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    for ds in args.datasets:
        run(ds, device)


if __name__ == "__main__":
    main()
