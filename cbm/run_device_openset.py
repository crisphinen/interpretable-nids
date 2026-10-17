#!/usr/bin/env python3
# Open-set test on the source-IP holdout splits with known AND unknown test flows from the held-out
# source address. On these splits test_known.parquet holds the held-out address's known-class flows,
# and test_unknown_device.parquet that address's own unknown-family flows (the main split's unknown
# pool, test_unknown.parquet, mostly comes from addresses seen in training).
#
# Scoring follows the main table exactly (cbm.evaluate.ood_auroc / tpr_at_fpr after
# np.random.seed(42); nesy/run_shared_eval.py for NeSy-NIDS): per-class Mahalanobis fitted on 20,000
# training vectors; AUROC on up to 5,000 known test vs 5,000 unknown flows; diagnostic TPR at the 95th
# percentile of ALL known test scores (a point on the test ROC curve, not a deployable threshold).
# In addition, the deployable operating point: tau = 95th percentile of the known validation scores
# (validation flows come from the training addresses), with the resulting rejection rate of the
# held-out address's known flows and the recall of its unknown flows.
#
# Unknown families: CTU address .198 keeps all its unknown flows (C&C 3,499, C&C-FileDownload 14,
# Okiru-Attack 3); TON_IoT keeps DoS only (sampled as the TON loader samples a family: 2,000 flows,
# np.random.default_rng(42)); the TON MITM flows (2 on address .30, 510 on .31) are scored as a
# separate set (--mitm).
# --reference scores the main-split models the same way against the main split's own C&C (CTU) or
# DoS (TON_IoT) flows: the seen-address baseline.
# Writes results/cbm/device_openset.json.
#
# usage: python -m cbm.run_device_openset --device cuda
#        python -m cbm.run_device_openset --reference --device cuda
#        python -m cbm.run_device_openset --mitm --device cuda
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cbm.run_multiseed import build, VARIANTS
from cbm.train import load_dataset
from cbm.evaluate import load_test_data, mahalanobis_scores
from nesy.evaluate import load_model, K_EVAL, _mahalanobis_scores_perclass
from nesy.run_shared_eval import rows as nesy_rows
from dsets import data_dir, nesy_feature_cols, is_ctu, load_vocab

RES = ROOT / "results"
UNK = "test_unknown_device.parquet"
# name -> (dataset of the models, unknown file, unknown families kept (None = all))
SETS = {"ctud_h198": ("ctud_h198", UNK, None), "ctud_o_h198": ("ctud_o_h198", UNK, None),
        "tond_a30": ("tond_a30", UNK, ["dos"]), "tond_a31": ("tond_a31", UNK, ["dos"])}
REFS = {"ctuk_cc": ("ctuk", "test_unknown.parquet", ["C&C"]), "ton_dos": ("ton", "test_unknown.parquet", ["dos"])}
MITM = {"tond_a30_mitm": ("tond_a30", UNK, ["mitm"]), "tond_a31_mitm": ("tond_a31", UNK, ["mitm"])}


def unknown_rows(base, fname, fams, cols):
    """Unknown flows of the given families, sampled as cbm.evaluate.load_test_data samples them."""
    df = pd.read_parquet(data_dir(base) / fname)
    v = load_vocab(base)
    if fams is not None:
        if "detailed_label" in df.columns:
            df = df[df["detailed_label"].isin(fams)]
        else:
            df = df[df["label_id"].isin([v["label_to_id"][f] for f in fams])]
        df = df.reset_index(drop=True)
    if is_ctu(base):
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    else:
        rng = np.random.default_rng(42); parts = []
        for lid in df["label_id"].unique():
            idx = np.where(df["label_id"].values == lid)[0]
            parts.append(rng.choice(idx, 2000, replace=False) if len(idx) > 2000 else idx)
        df = df.iloc[np.sort(np.concatenate(parts))].reset_index(drop=True)
    return df[cols].values.astype(np.float32)


def operating_points(fit, Xtr_e, ytr, Xval_e, Xk_e, Xu_e):
    """AUROC and diagnostic TPR in the random order of ood_auroc + tpr_at_fpr after seed 42, plus
    the validation-threshold rejection rate and recall. fit(train_vecs, train_labels, vecs) -> scores."""
    np.random.seed(42)
    it = np.random.choice(len(Xtr_e), 20000, replace=False) if len(Xtr_e) > 20000 else np.arange(len(Xtr_e))
    sk, su, sv = (fit(Xtr_e[it], ytr[it], E) for E in (Xk_e, Xu_e, Xval_e))
    ik = np.random.choice(len(sk), 5000, replace=False) if len(sk) > 5000 else np.arange(len(sk))
    iu = np.random.choice(len(su), 5000, replace=False) if len(su) > 5000 else np.arange(len(su))
    auroc = roc_auc_score(np.r_[np.zeros(len(ik)), np.ones(len(iu))], np.r_[sk[ik], su[iu]])
    it2 = np.random.choice(len(Xtr_e), 20000, replace=False) if len(Xtr_e) > 20000 else np.arange(len(Xtr_e))
    tk, tu = (fit(Xtr_e[it2], ytr[it2], E) for E in (Xk_e, Xu_e))
    tpr_diag = float((tu > np.percentile(tk, 95)).mean())
    tau = np.percentile(sv, 95)
    return {"auroc": float(auroc), "tpr5": tpr_diag,
            "val_tau_known_reject": float((sk > tau).mean()), "val_tau_recall": float((su > tau).mean())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda")
    ap.add_argument("--reference", action="store_true"); ap.add_argument("--mitm", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    todo = REFS if args.reference else MITM if args.mitm else SETS
    dest = RES / "cbm/device_openset.json"
    out = json.loads(dest.read_text()) if dest.exists() else {}
    T = lambda X: torch.tensor(X, dtype=torch.float32, device=dev)
    for name, (base, fname, fams) in todo.items():
        (Xtr, ytr, _), _, nf, nk, nc, _ = load_dataset(base)
        (Xk, _, _), _, _, Xval, _ = load_test_data(base)
        Xu = unknown_rows(base, fname, fams, load_vocab(base)["feature_cols"])
        cbm_fit = lambda tr, y, E: mahalanobis_scores(tr, E, train_labels=y)
        rows = []
        for vname, _ in VARIANTS:
            arch = "JointCBM" if vname.startswith("JointCBM") else vname
            for s in range(5):
                f = RES / f"cbm/ms/{base}_{vname}_s{s}.pt"
                if not f.exists():
                    continue
                m = build(arch, nf, nc, nk)
                m.load_state_dict(torch.load(f, map_location=dev)["model_state_dict"]); m.to(dev).eval()
                with torch.no_grad():
                    E = [m.get_embedding(T(X)).cpu().numpy() for X in (Xtr, Xval, Xk, Xu)]
                rows.append({"model": vname, "seed": s, **operating_points(cbm_fit, E[0], ytr, E[1], E[2], E[3])})
        if is_ctu(base):
            cols = nesy_feature_cols(base)
            NXtr, Nytr, NXk, _, _ = nesy_rows(base, cols)
            vdf = pd.read_parquet(data_dir(base) / "val.parquet")
            vdf = vdf[vdf["label_id"].isin(load_vocab(base)["known_ids"])]
            NXval = vdf[cols].values.astype(np.float32)
            NXu = unknown_rows(base, fname, fams, cols)
            for model, tag in (("NeSy", ""), ("NeSy_a0.1", "_a0.1"), ("RuleOnly", "")):
                for s in range(5):
                    f = RES / f"nesy/{base}_{'rule_only' if model == 'RuleOnly' else 'nesy'}_s{s}{tag}.pt"
                    if not f.exists():
                        continue
                    m, _ = load_model(f, dev); m.eval()
                    with torch.no_grad():
                        E = [m.get_embedding(T(X), k=K_EVAL).cpu().numpy() for X in (NXtr, NXval, NXk, NXu)]
                    rows.append({"model": model, "seed": s,
                                 **operating_points(_mahalanobis_scores_perclass, E[0], Nytr, E[1], E[2], E[3])})
        summary = {}
        for m in dict.fromkeys(r["model"] for r in rows):
            z = [r for r in rows if r["model"] == m]
            summary[m] = {k: [float(np.mean([r[k] for r in z])), float(np.std([r[k] for r in z]))]
                          for k in ("auroc", "tpr5", "val_tau_known_reject", "val_tau_recall")}
            summary[m]["n"] = len(z)
            s_ = summary[m]
            print(f"[{name}] {m:14s} AUROC {s_['auroc'][0]:.3f}  TPR(test thr) {s_['tpr5'][0]:.3f}  "
                  f"val-tau: known rejected {s_['val_tau_known_reject'][0]:.3f}, recall {s_['val_tau_recall'][0]:.3f}", flush=True)
        out[name] = {"models_from": base, "unknown_file": fname, "families": fams, "n_unknown": int(len(Xu)),
                     "rows": rows, "summary": summary}
        dest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
