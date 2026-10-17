#!/usr/bin/env python3
# Device-disjoint open-set test. On the device-disjoint splits the known test flows come from the
# held-out device, while test_unknown.parquet is the main split's unknown pool, most of it from
# devices seen in training. Here every saved model is re-scored against test_unknown_device.parquet,
# the held-out device's own unknown-family flows, so known and unknown test flows both come from
# the unseen device. Scoring follows the main table: per-class Mahalanobis fitted on 20,000
# training vectors, AUROC on up to 5,000 known vs 5,000 unknown flows (np.random.seed(42)),
# TPR at the 95th percentile of known scores (cbm.evaluate.tpr_at_fpr; nesy/run_shared_eval.py).
# --reference scores the main-split models the same way against the main split's own flows of the
# family that dominates the held-out devices' unknowns (seen-device baseline): ctuk_cc = CTU C&C
# (99.5% of host .198's unknown flows), ton_dos = TON_IoT DoS (all but 512 of the TON folds').
# The TON reference takes 2,000 DoS flows as the TON loader does (np.random.default_rng(42)).
# Writes results/cbm/device_openset.json.
#
# usage: python -m cbm.run_device_openset --device cuda
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
from cbm.evaluate import load_test_data, ood_auroc, tpr_at_fpr
from nesy.evaluate import load_model, K_EVAL, _mahalanobis_scores_perclass
from nesy.run_shared_eval import rows as nesy_rows, subsets
from dsets import data_dir, nesy_feature_cols, is_ctu

RES = ROOT / "results"
DATASETS = ["ctud_h198", "ctud_o_h198", "tond_a30", "tond_a31"]
UNK = "test_unknown_device.parquet"


REFERENCES = {"ctuk_cc": "ctuk", "ton_dos": "ton"}


def unknown_frame(ds):
    if ds == "ctuk_cc":                       # reference: the main split's C&C flows
        df = pd.read_parquet(data_dir("ctuk") / "test_unknown.parquet")
        return df[df["detailed_label"] == "C&C"].reset_index(drop=True)
    if ds == "ton_dos":                       # reference: the main split's DoS flows, 2,000 as the loader
        v = json.loads((data_dir("ton") / "vocab.json").read_text())
        df = pd.read_parquet(data_dir("ton") / "test_unknown.parquet")
        idx = np.where(df["label_id"].values == v["label_to_id"]["dos"])[0]
        idx = np.sort(np.random.default_rng(42).choice(idx, 2000, replace=False))
        return df.iloc[idx].reset_index(drop=True)
    return pd.read_parquet(data_dir(ds) / UNK)


def nesy_device_rows(ds):
    base = REFERENCES.get(ds, ds)
    cols = nesy_feature_cols(base)
    Xtr, ytr, Xk, yk, _ = nesy_rows(base, cols)
    df = unknown_frame(ds)
    if len(df) > 10000:                       # same cap as cbm.evaluate.load_test_data (CTU)
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    return Xtr, ytr, Xk, df[cols].values.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--reference", action="store_true", help="seen-device baselines (ctuk_cc, ton_dos)")
    args = ap.parse_args()
    if args.reference:
        args.datasets = list(REFERENCES)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dest = RES / "cbm/device_openset.json"
    out = json.loads(dest.read_text()) if dest.exists() else {}
    for ds in args.datasets:
        base = REFERENCES.get(ds, ds)
        (Xtr, ytr, _), _, nf, nk, nc, _ = load_dataset(base)
        if ds in REFERENCES:
            (Xk, _, _), _, _, _, _ = load_test_data(base)
            df = unknown_frame(ds)
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42).reset_index(drop=True)
            Xu = df[json.loads((data_dir(base) / "vocab.json").read_text())["feature_cols"]].values.astype(np.float32)
        else:
            (Xk, _, _), _, Xu, _, _ = load_test_data(ds, unknown_file=UNK)
        # the CBM loader caps CTU unknowns at 10,000 and samples CIC/TON unknowns per class
        rows = []
        for vname, _ in VARIANTS:
            arch = "JointCBM" if vname.startswith("JointCBM") else vname
            for s in range(5):
                f = RES / f"cbm/ms/{base}_{vname}_s{s}.pt"
                if not f.exists():
                    continue
                m = build(arch, nf, nc, nk)
                m.load_state_dict(torch.load(f, map_location=dev)["model_state_dict"]); m.to(dev).eval()
                np.random.seed(42)
                au = float(ood_auroc(m, Xtr, ytr, Xk, Xu, dev))
                tp = float(tpr_at_fpr(m, Xtr, ytr, Xk, Xu, dev))
                rows.append({"model": vname, "seed": s, "auroc": au, "tpr5": tp})
        if is_ctu(base):
            NXtr, Nytr, NXk, NXu = nesy_device_rows(ds)
            it, ik, iu, it2 = subsets(len(NXtr), len(NXk), len(NXu))
            T = lambda X: torch.tensor(X, dtype=torch.float32, device=dev)
            for model, tag in (("NeSy", ""), ("NeSy_a0.1", "_a0.1"), ("RuleOnly", "")):
                for s in range(5):
                    name = "rule_only" if model == "RuleOnly" else "nesy"
                    f = RES / f"nesy/{base}_{name}_s{s}{tag}.pt"
                    if not f.exists():
                        continue
                    m, _ = load_model(f, dev); m.eval()
                    with torch.no_grad():
                        htr, hk, hu = (m.get_embedding(T(X), k=K_EVAL).cpu().numpy() for X in (NXtr, NXk, NXu))
                    mk = _mahalanobis_scores_perclass(htr[it], Nytr[it], hk[ik])
                    mu = _mahalanobis_scores_perclass(htr[it], Nytr[it], hu[iu])
                    tk = _mahalanobis_scores_perclass(htr[it2], Nytr[it2], hk)
                    tu = _mahalanobis_scores_perclass(htr[it2], Nytr[it2], hu)
                    rows.append({"model": model, "seed": s,
                                 "auroc": float(roc_auc_score(np.r_[np.zeros(len(mk)), np.ones(len(mu))], np.r_[mk, mu])),
                                 "tpr5": float((tu > np.percentile(tk, 95)).mean())})
        summary = {}
        for m in dict.fromkeys(r["model"] for r in rows):
            a = np.array([r["auroc"] for r in rows if r["model"] == m]); t = np.array([r["tpr5"] for r in rows if r["model"] == m])
            summary[m] = {"auroc": [float(a.mean()), float(a.std())], "tpr5": [float(t.mean()), float(t.std())], "n": len(a)}
            print(f"[{ds}] {m:14s} AUROC {a.mean():.3f}±{a.std():.3f}  TPR {t.mean():.3f}±{t.std():.3f}  n={len(a)}", flush=True)
        out[ds] = {"n_unknown": int(len(Xu)), "rows": rows, "summary": summary}
        dest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
