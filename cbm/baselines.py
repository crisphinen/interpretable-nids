import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm.concepts import (
    CTU_CONCEPTS, CIC_CONCEPTS,
    ctu_concept_labels, cic_concept_labels,
)
from cbm.model import MLPBaseline, JointCBM

RESULTS_DIR = PROJECT_ROOT / "cbm" / "results"
EMBED_DIM = 64
DEVICE = torch.device("cpu")
MAX_SHAP_BACKGROUND = 200
MAX_SHAP_EXPLAIN    = 500
MAX_MAHAL_TRAIN     = 20000
MAX_UNKNOWN         = 10000


def load_dataset(dataset: str):
    if dataset == "ctu":
        data_dir = PROJECT_ROOT / "data"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        known_ids = vocab["known_ids"]
        id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}

        def _load(split):
            df = pd.read_parquet(data_dir / f"{split}.parquet")
            df = df[df["label_id"].isin(known_ids)].reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
            C = ctu_concept_labels(df)
            return X, y, C

        def _unk():
            df = pd.read_parquet(data_dir / "test_unknown.parquet")
            X = df[feature_cols].values.astype(np.float32)
            if len(X) > MAX_UNKNOWN:
                X = X[np.random.default_rng(42).choice(len(X), MAX_UNKNOWN, replace=False)]
            return X

        concept_names = CTU_CONCEPTS

    elif dataset == "cic":
        data_dir = PROJECT_ROOT / "data" / "cic"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        l2i = vocab["label_to_id"]
        known_classes = vocab["known_classes"]
        known_ids = [l2i[c] for c in known_classes]
        id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}

        def _load(split):
            df = pd.read_parquet(data_dir / f"{split}.parquet")
            df = df[df["label_id"].isin(known_ids)].reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
            C = cic_concept_labels(df)
            return X, y, C

        def _unk():
            print("  Loading CIC unknown (stratified subsample)...")
            df_ids = pd.read_parquet(data_dir / "test_unknown.parquet", columns=["label_id"])
            rng = np.random.default_rng(42)
            per_cls = max(1, MAX_UNKNOWN // max(1, df_ids["label_id"].nunique()))
            parts = []
            for lid in df_ids["label_id"].unique():
                idx = np.where(df_ids["label_id"].values == lid)[0]
                parts.append(rng.choice(idx, min(len(idx), per_cls), replace=False))
            selected = np.sort(np.concatenate(parts))
            df_full = pd.read_parquet(data_dir / "test_unknown.parquet")
            return df_full.iloc[selected][feature_cols].values.astype(np.float32)

        concept_names = CIC_CONCEPTS

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    X_tr, y_tr, C_tr   = _load("train")
    X_val, y_val, C_val = _load("val")
    X_te, y_te, C_te   = _load("test_known")
    X_unk               = _unk()

    return (X_tr, y_tr, C_tr,
            X_val, y_val, C_val,
            X_te, y_te, C_te,
            X_unk, concept_names, feature_cols)



def mahal_auroc_tpr(X_train, X_known_test, X_unknown, fpr_target=0.05):
    n = min(len(X_train), MAX_MAHAL_TRAIN)
    idx = np.random.default_rng(42).choice(len(X_train), n, replace=False)
    cov = EmpiricalCovariance(assume_centered=False)
    cov.fit(X_train[idx])

    def _cap(X, cap=5000):
        if len(X) > cap:
            return X[np.random.default_rng(0).choice(len(X), cap, replace=False)]
        return X

    sc_k = cov.mahalanobis(_cap(X_known_test))
    sc_u = cov.mahalanobis(_cap(X_unknown))
    labels = np.concatenate([np.zeros(len(sc_k)), np.ones(len(sc_u))])
    scores = np.concatenate([sc_k, sc_u])
    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float("nan")
    thresh = np.percentile(sc_k, (1 - fpr_target) * 100)
    tpr = float((sc_u > thresh).mean())
    return auc, tpr



def run_decision_tree(X_tr, y_tr, X_val, y_val, X_te, y_te,
                      X_unk, feature_cols, dataset):
    print("\nBaseline 1: Decision Tree")
    best_f1, best_dt = -1.0, None
    for depth in [5, 10, 15, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_tr, y_tr)
        f1 = f1_score(y_val, dt.predict(X_val), average="weighted", zero_division=0)
        print(f"  depth={str(depth):<5}: val_F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_dt = f1, dt

    test_f1 = f1_score(y_te, best_dt.predict(X_te), average="weighted", zero_division=0)
    auroc, tpr = mahal_auroc_tpr(X_tr, X_te, X_unk)

    print(f"\n  Best DT (depth={best_dt.get_depth()}, leaves={best_dt.get_n_leaves()})")
    print(f"  Test F1:      {test_f1:.4f}")
    print(f"  OOD AUROC:    {auroc:.4f}  (Mahalanobis, raw features)")
    print(f"  TPR @ 5% FPR: {tpr:.4f}")

    imp = best_dt.feature_importances_
    top10 = np.argsort(imp)[::-1][:10]
    print(f"\n  Top-10 features by DT importance:")
    for r, fi in enumerate(top10):
        print(f"    {r+1:2d}. {feature_cols[fi]:<30}  {imp[fi]:.4f}")

    return {
        "test_f1": float(test_f1),
        "ood_auroc": float(auroc),
        "tpr_at_5pct_fpr": float(tpr),
        "depth": int(best_dt.get_depth()),
        "n_leaves": int(best_dt.get_n_leaves()),
        "top10_features": [feature_cols[fi] for fi in top10],
        "top10_importances": [float(imp[fi]) for fi in top10],
    }


def run_shap_mlp(X_tr, y_tr, X_te, y_te, feature_cols, concept_names, dataset):
    import shap

    print("\nBaseline 2: SHAP-MLP")
    ckpt_path = RESULTS_DIR / f"{dataset}_MLPBaseline.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] MLPBaseline checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    n_features = ckpt["n_features"]
    n_classes  = ckpt["n_classes"]
    n_concepts = ckpt["n_concepts"]

    model = MLPBaseline(n_features, n_classes, EMBED_DIM)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Verify F1
    with torch.no_grad():
        logits, _ = model(torch.tensor(X_te, dtype=torch.float32))
    preds = logits.argmax(1).numpy()
    test_f1 = f1_score(y_te, preds, average="weighted", zero_division=0)
    print(f"  MLPBaseline test F1: {test_f1:.4f}")

    # SHAP: GradientExplainer needs a torch model wrapper returning logits
    class _Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            logits, _ = self.m(x)
            return logits

    rng = np.random.default_rng(42)
    bg_idx  = rng.choice(len(X_tr), min(MAX_SHAP_BACKGROUND, len(X_tr)), replace=False)
    ex_idx  = rng.choice(len(X_te), min(MAX_SHAP_EXPLAIN, len(X_te)), replace=False)

    X_bg = torch.tensor(X_tr[bg_idx], dtype=torch.float32)
    X_ex = torch.tensor(X_te[ex_idx], dtype=torch.float32)

    print(f"  Running GradientExplainer (background={len(X_bg)}, explain={len(X_ex)})...")
    explainer = shap.GradientExplainer(_Wrapper(model), X_bg)
    shap_values = explainer.shap_values(X_ex)  # list of (N, F) per class

    # Mean absolute SHAP across classes and samples → global feature importance
    shap_arr = np.stack(shap_values, axis=0)       # (C, N, F)
    mean_abs_shap = np.abs(shap_arr).mean(axis=(0, 1))  # (F,)

    top10_shap = np.argsort(mean_abs_shap)[::-1][:10]
    print(f"\n  Top-10 features by mean |SHAP|:")
    for r, fi in enumerate(top10_shap):
        print(f"    {r+1:2d}. {feature_cols[fi]:<30}  {mean_abs_shap[fi]:.6f}")

    print(f"\n  Key limitation for paper:")
    print(f"  SHAP gives feature attributions — not concept predictions.")
    print(f"  Cannot intervene on SHAP values; cannot compute 'concept accuracy'.")
    print(f"  JointCBM provides {len(concept_names)} named concepts with structural bottleneck (interventionable).")

    return {
        "test_f1": float(test_f1),
        "top10_features_by_shap": [feature_cols[fi] for fi in top10_shap],
        "top10_mean_abs_shap": [float(mean_abs_shap[fi]) for fi in top10_shap],
        "n_background": int(len(X_bg)),
        "n_explained": int(len(X_ex)),
        "note": (
            "SHAP provides local post-hoc feature attributions. "
            "No concept bottleneck — intervention accuracy is undefined. "
            "Compare with JointCBM per-concept accuracy and intervention delta."
        ),
    }


def run_posthoc_cbm(X_tr, y_tr, C_tr,
                    X_val, y_val, C_val,
                    X_te, y_te, C_te,
                    X_unk, concept_names, feature_cols, dataset):
    print("\nBaseline 3: Post-hoc CBM (frozen MLP + linear probes)")
    ckpt_path = RESULTS_DIR / f"{dataset}_MLPBaseline.pt"
    if not ckpt_path.exists():
        print(f"  [SKIP] MLPBaseline checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    n_features = ckpt["n_features"]
    n_classes  = ckpt["n_classes"]
    n_concepts = ckpt["n_concepts"]

    model = MLPBaseline(n_features, n_classes, EMBED_DIM)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    @torch.no_grad()
    def _embed(X, bs=4096):
        parts = []
        for i in range(0, len(X), bs):
            parts.append(model.get_embedding(
                torch.tensor(X[i:i+bs], dtype=torch.float32)
            ).numpy())
        return np.concatenate(parts)

    print("  Extracting frozen embeddings...")
    E_tr  = _embed(X_tr)
    E_val = _embed(X_val)
    E_te  = _embed(X_te)
    E_unk = _embed(X_unk)

    print(f"  Fitting {len(concept_names)} linear concept probes...")
    concept_preds_val = np.zeros((len(X_val), len(concept_names)), dtype=np.float32)
    concept_preds_te  = np.zeros((len(X_te),  len(concept_names)), dtype=np.float32)
    concept_preds_unk = np.zeros((len(X_unk), len(concept_names)), dtype=np.float32)
    concept_accs      = {}

    for j, cname in enumerate(concept_names):
        clf = LogisticRegression(max_iter=200, C=1.0, random_state=42, n_jobs=1)
        clf.fit(E_tr, C_tr[:, j])
        probe_acc = (clf.predict(E_te) == C_te[:, j]).mean()
        concept_accs[cname] = float(probe_acc)

        # Soft probabilities as concept predictions (probability of concept=1)
        concept_preds_val[:, j] = clf.predict_proba(E_val)[:, 1]
        concept_preds_te[:, j]  = clf.predict_proba(E_te)[:, 1]
        concept_preds_unk[:, j] = clf.predict_proba(E_unk)[:, 1]

    mean_concept_acc = np.mean(list(concept_accs.values()))
    print(f"\n  Per-concept probe accuracy:")
    for cname, acc in concept_accs.items():
        flag = " ← unsafe (<0.97)" if acc < 0.97 else ""
        print(f"    {cname:<28} {acc:.4f}{flag}")
    print(f"  Mean concept accuracy: {mean_concept_acc:.4f}")

    print("\n  Training linear classifier on concept vectors...")
    clf_head = LogisticRegression(max_iter=500, C=1.0, random_state=42, n_jobs=1)
    clf_head.fit(concept_preds_val, y_val)
    preds_te = clf_head.predict(concept_preds_te)
    test_f1 = f1_score(y_te, preds_te, average="weighted", zero_division=0)
    print(f"  Post-hoc CBM test F1: {test_f1:.4f}")

    auroc, tpr = mahal_auroc_tpr(
        concept_preds_val,   # use val concept predictions as "train" for Mahalanobis
        concept_preds_te,
        concept_preds_unk,
    )
    print(f"  OOD AUROC (concept Mahalanobis): {auroc:.4f}")
    print(f"  TPR @ 5% FPR:                    {tpr:.4f}")

    # summary comparison
    print(f"\n  Post-hoc CBM vs JointCBM comparison (concept accuracy):")
    print(f"  {'Concept':<28}  {'PostHoc':>8}  {'JointCBM (γ=0.5 from eval)':>10}")
    cic_g05 = {
        "is_high_rate": 0.9972, "is_syn_flood": 0.9957,
        "is_udp_dominant": 0.9987, "is_short_connection": 0.9229,
        "is_large_payload": 0.9696, "is_port_scan": 0.9851,
        "is_high_variance": 0.9439, "is_persistent": 0.9990,
    }
    ctu_g05 = {}  # add if running CTU
    joint_ref = cic_g05 if dataset == "cic" else ctu_g05
    for cname, ph_acc in concept_accs.items():
        j_acc = joint_ref.get(cname, float("nan"))
        delta = ph_acc - j_acc if not np.isnan(j_acc) else float("nan")
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "    N/A"
        print(f"  {cname:<28}  {ph_acc:>8.4f}  {j_acc:>8.4f}  Δ={delta_str}")

    return {
        "test_f1": float(test_f1),
        "ood_auroc": float(auroc),
        "tpr_at_5pct_fpr": float(tpr),
        "mean_concept_accuracy": float(mean_concept_acc),
        "concept_accuracies": concept_accs,
        "note": (
            "Post-hoc CBM: linear probes on frozen MLPBaseline embeddings. "
            "Concept accuracy reflects how well a frozen representation "
            "separates human-defined concepts vs JointCBM's end-to-end training."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ctu", "cic"], required=True)
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP computation (can be slow on large test sets)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  CBM-NIDS Baselines — {args.dataset.upper()}")
    print(f"{'='*65}")

    (X_tr, y_tr, C_tr,
     X_val, y_val, C_val,
     X_te, y_te, C_te,
     X_unk, concept_names, feature_cols) = load_dataset(args.dataset)

    print(f"\n  Train:   {X_tr.shape}")
    print(f"  Val:     {X_val.shape}")
    print(f"  Test:    {X_te.shape}")
    print(f"  Unknown: {X_unk.shape}")
    print(f"  Concepts ({len(concept_names)}): {concept_names}")

    results = {}

    results["DecisionTree"] = run_decision_tree(
        X_tr, y_tr, X_val, y_val, X_te, y_te,
        X_unk, feature_cols, args.dataset,
    )

    if not args.skip_shap:
        results["SHAP_MLP"] = run_shap_mlp(
            X_tr, y_tr, X_te, y_te,
            feature_cols, concept_names, args.dataset,
        )
    else:
        print("\nBaseline 2: SHAP-MLP [SKIPPED]")

    results["PostHocCBM"] = run_posthoc_cbm(
        X_tr, y_tr, C_tr,
        X_val, y_val, C_val,
        X_te, y_te, C_te,
        X_unk, concept_names, feature_cols, args.dataset,
    )

    print(f"\n{'='*65}")
    print(f"  BASELINE SUMMARY — {args.dataset.upper()}")
    print(f"{'='*65}")
    print(f"  {'Model':<20}  {'Test F1':>8}  {'OOD AUROC':>10}  {'TPR@5%':>8}")
    print(f"  {'-'*55}")
    for name, res in results.items():
        if res is None:
            continue
        f1  = res.get("test_f1", float("nan"))
        auc = res.get("ood_auroc", float("nan"))
        tpr = res.get("tpr_at_5pct_fpr", float("nan"))
        print(f"  {name:<20}  {f1:>8.4f}  {auc:>10.4f}  {tpr:>8.4f}")

    # Save
    out_path = RESULTS_DIR / f"{args.dataset}_cbm_baselines.json"

    def _cast(obj):
        if isinstance(obj, dict):  return {k: _cast(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_cast(v) for v in obj]
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    out_path.write_text(json.dumps(_cast(results), indent=2))
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
