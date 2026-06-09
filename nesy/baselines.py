# baseline comparisons for nesy-nids.
#
# baselines:
# 1. decisiontree - sklearn dt (rule-based, natural comparison for nesy)
# 2. randomforest - rf ensemble (upper bound for rule-based approaches)
# 3. shap-mlp     - load existing mlpbaseline checkpoint, compute shap feature importance
#
# usage:
# python -m nesy.baselines --dataset ctu
# python -m nesy.baselines --dataset cic

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "nesy" / "results"


# data loading

def load_ctu(split):
    data_dir = PROJECT_ROOT / "data"
    vocab = json.loads((data_dir / "vocab.json").read_text())
    feature_cols = vocab["feature_cols"]
    known_ids = vocab["known_ids"]
    df = pd.read_parquet(data_dir / f"{split}.parquet")
    mask = df["label_id"].isin(known_ids)
    df = df[mask].reset_index(drop=True)
    X = df[feature_cols].values.astype(np.float32)
    id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}
    y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
    id_to_label = {i: vocab["id_to_label"][str(orig)]
                   for i, orig in enumerate(sorted(known_ids))}
    return X, y, id_to_label, feature_cols


def load_cic(split):
    data_dir = PROJECT_ROOT / "data" / "cic"
    vocab = json.loads((data_dir / "vocab.json").read_text())
    feature_cols = vocab["feature_cols"]
    l2i = vocab["label_to_id"]
    known_classes = vocab["known_classes"]
    known_ids = [l2i[c] for c in known_classes]
    df = pd.read_parquet(data_dir / f"{split}.parquet")
    mask = df["label_id"].isin(known_ids)
    df = df[mask].reset_index(drop=True)
    X = df[feature_cols].values.astype(np.float32)
    id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}
    y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
    id_to_label = {i: c for i, (orig, c) in enumerate(
        sorted([(lid, cls) for cls, lid in l2i.items() if lid in known_ids])
    )}
    return X, y, id_to_label, feature_cols


def load_unknown(dataset: str, n_sample: int = 10000):
    if dataset == "ctu":
        data_dir = PROJECT_ROOT / "data"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        known_ids = set(vocab["known_ids"])
        df = pd.read_parquet(data_dir / "test_unknown.parquet")
        mask = ~df["label_id"].isin(known_ids)
        df = df[mask].reset_index(drop=True)
        if len(df) > n_sample:
            df = df.sample(n=n_sample, random_state=42).reset_index(drop=True)
        return df[feature_cols].values.astype(np.float32)
    else:
        data_dir = PROJECT_ROOT / "data" / "cic"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        l2i = vocab["label_to_id"]
        known_ids = set([l2i[c] for c in vocab["known_classes"]])
        test_path = data_dir / "test_unknown.parquet"
        label_col = pd.read_parquet(test_path, columns=["label_id"])["label_id"]
        unknown_mask = ~label_col.isin(known_ids)
        unknown_ids_all = label_col[unknown_mask]
        per_class = max(1, n_sample // max(1, len(unknown_ids_all.unique())))
        sampled_idx = []
        for lid in unknown_ids_all.unique():
            idx = unknown_ids_all[unknown_ids_all == lid].index.tolist()
            sampled_idx.extend(idx[:per_class])
        sampled_idx = sampled_idx[:n_sample]
        df = pd.read_parquet(test_path, columns=feature_cols + ["label_id"])
        return df.iloc[sampled_idx][feature_cols].values.astype(np.float32)


# ood scoring

# 1 - max(predict_proba). high = low confidence = likely ood.
def ood_score_proba(model, X):
    proba = model.predict_proba(X)
    return 1.0 - proba.max(axis=1)


# mahalanobis distance ood score in raw feature space.
def mahalanobis_ood(X_known_train, X_known_test, X_unknown, cap=20000):
    from numpy.linalg import pinv
    if len(X_known_train) > cap:
        idx = np.random.choice(len(X_known_train), cap, replace=False)
        X_fit = X_known_train[idx]
    else:
        X_fit = X_known_train

    # regularize to avoid singular covariance
    cov = np.cov(X_fit.T) + 1e-4 * np.eye(X_fit.shape[1])
    cov_inv = pinv(cov)
    mu = X_fit.mean(axis=0)

    def score(X):
        if len(X) > 5000:
            X = X[np.random.choice(len(X), 5000, replace=False)]
        diff = X - mu
        return np.sum(diff @ cov_inv * diff, axis=1)

    return score(X_known_test), score(X_unknown)


def auroc(score_known, score_unknown):
    scores = np.concatenate([score_known, score_unknown])
    labels = np.concatenate([np.zeros(len(score_known)), np.ones(len(score_unknown))])
    return roc_auc_score(labels, scores)


def tpr_at_fpr(score_known, score_unknown, target_fpr=0.05):
    thresh = np.percentile(score_known, (1 - target_fpr) * 100)
    return float((score_unknown >= thresh).mean())


# main evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ctu", "cic"], required=True)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  NeSy-NIDS Baselines - {args.dataset.upper()}")
    print(f"{'='*60}\n")

    load_fn = load_ctu if args.dataset == "ctu" else load_cic
    X_tr, y_tr, id_to_label, feature_cols = load_fn("train")
    X_val, y_val, _, _ = load_fn("val")
    X_unk = load_unknown(args.dataset)

    print(f"  Train: {X_tr.shape}  Val: {X_val.shape}  Unknown: {X_unk.shape}")
    print(f"  Classes: {id_to_label}")

    results = {}

    # decision tree
    print("\n--- Decision Tree ---")
    best_dt_f1, best_dt = -1.0, None
    for depth in [5, 10, 15, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_tr, y_tr)
        f1 = f1_score(y_val, dt.predict(X_val), average="weighted", zero_division=0)
        print(f"  depth={depth}: val_F1={f1:.4f}")
        if f1 > best_dt_f1:
            best_dt_f1, best_dt = f1, dt

    # ood: both max-proba-confidence and mahalanobis
    sc_known_conf  = ood_score_proba(best_dt, X_val)
    sc_unk_conf    = ood_score_proba(best_dt, X_unk)
    sc_known_maha, sc_unk_maha = mahalanobis_ood(X_tr, X_val, X_unk)

    auroc_conf = auroc(sc_known_conf, sc_unk_conf)
    auroc_maha = auroc(sc_known_maha, sc_unk_maha)
    tpr_conf   = tpr_at_fpr(sc_known_conf, sc_unk_conf)
    tpr_maha   = tpr_at_fpr(sc_known_maha, sc_unk_maha)
    depth_used = best_dt.get_depth()
    n_leaves   = best_dt.get_n_leaves()

    print(f"\n  Best DT (depth={depth_used}, leaves={n_leaves}):")
    print(f"    F1:           {best_dt_f1:.4f}")
    print(f"    AUROC (conf): {auroc_conf:.4f}  TPR@5%={tpr_conf:.4f}")
    print(f"    AUROC (maha): {auroc_maha:.4f}  TPR@5%={tpr_maha:.4f}")

    # top features by importance
    top_k = 10
    imp = best_dt.feature_importances_
    top_idx = np.argsort(imp)[::-1][:top_k]
    print(f"\n  Top-{top_k} DT feature importances:")
    for rank, fi in enumerate(top_idx):
        print(f"    {rank+1:2d}. {feature_cols[fi]:<28} {imp[fi]:.4f}")

    results["DecisionTree"] = {
        "f1": best_dt_f1, "auroc_conf": auroc_conf, "auroc_maha": auroc_maha,
        "tpr5_conf": tpr_conf, "tpr5_maha": tpr_maha,
        "depth": depth_used, "n_leaves": n_leaves,
    }

    # random forest
    print("\n--- Random Forest (100 trees, max_depth=10) ---")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_f1 = f1_score(y_val, rf.predict(X_val), average="weighted", zero_division=0)

    sc_known_rf  = ood_score_proba(rf, X_val)
    sc_unk_rf    = ood_score_proba(rf, X_unk)
    rf_auroc = auroc(sc_known_rf, sc_unk_rf)
    rf_tpr   = tpr_at_fpr(sc_known_rf, sc_unk_rf)

    print(f"  F1={rf_f1:.4f}  AUROC={rf_auroc:.4f}  TPR@5%={rf_tpr:.4f}")

    top_idx_rf = np.argsort(rf.feature_importances_)[::-1][:top_k]
    print(f"\n  Top-{top_k} RF feature importances:")
    for rank, fi in enumerate(top_idx_rf):
        print(f"    {rank+1:2d}. {feature_cols[fi]:<28} {rf.feature_importances_[fi]:.4f}")

    results["RandomForest"] = {
        "f1": rf_f1, "auroc": rf_auroc, "tpr5": rf_tpr,
    }

    # summary table
    print(f"\n{'='*60}")
    print(f"  Baseline Summary - {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"  {'Model':<20}  {'F1':>7}  {'AUROC':>7}  {'TPR@5%':>7}")
    print(f"  {'-'*50}")
    print(f"  {'DecisionTree':<20}  {best_dt_f1:>7.4f}  {auroc_maha:>7.4f}  {tpr_maha:>7.4f}")
    print(f"  {'RandomForest':<20}  {rf_f1:>7.4f}  {rf_auroc:>7.4f}  {rf_tpr:>7.4f}")

    # save
    import json as _json
    out = RESULTS_DIR / f"{args.dataset}_baselines.json"
    # cast numpy types to python natives for json serialisation
    def _cast(obj):
        if isinstance(obj, dict):
            return {k: _cast(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    out.write_text(_json.dumps(_cast(results), indent=2))
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
