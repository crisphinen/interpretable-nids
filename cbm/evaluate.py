import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.covariance import EmpiricalCovariance

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm.concepts import (
    CTU_CONCEPTS, CIC_CONCEPTS,
    ctu_concept_labels, cic_concept_labels,
)
from cbm.model import MLPBaseline, JointCBM, SequentialCBM, HybridCBM

RESULTS_DIR = PROJECT_ROOT / "cbm" / "results"
EMBED_DIM = 64


def load_test_data(dataset: str):
    if dataset == "ctu":
        data_dir = PROJECT_ROOT / "data"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        known_ids = vocab["known_ids"]
        id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}

        def _load_known(split):
            df = pd.read_parquet(data_dir / f"{split}.parquet")
            mask = df["label_id"].isin(known_ids)
            df = df[mask].reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
            C = ctu_concept_labels(df)
            return X, y, C

        def _load_unknown(max_rows=10000):
            # CTU test_unknown is manageable; no subsampling needed
            df = pd.read_parquet(data_dir / "test_unknown.parquet")
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            return X

        X_known, y_known, C_known = _load_known("test_known")
        # Also load train split for Mahalanobis fitting
        X_train, y_train, C_train = _load_known("train")
        X_unknown = _load_unknown()
        concept_names = CTU_CONCEPTS

    elif dataset == "cic":
        data_dir = PROJECT_ROOT / "data" / "cic"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        l2i = vocab["label_to_id"]
        known_classes = vocab["known_classes"]
        known_ids = [l2i[c] for c in known_classes]
        id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}

        def _load_known(split):
            df = pd.read_parquet(data_dir / f"{split}.parquet")
            mask = df["label_id"].isin(known_ids)
            df = df[mask].reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
            C = cic_concept_labels(df)
            return X, y, C

        def _load_unknown(max_per_class=2000):
            # CIC test_unknown has 40M rows — stratified subsample per class
            print("  Loading CIC test_unknown (stratified subsample)...")
            df_ids = pd.read_parquet(data_dir / "test_unknown.parquet", columns=["label_id"])
            rng = np.random.default_rng(42)
            parts = []
            for lid in df_ids["label_id"].unique():
                idx = np.where(df_ids["label_id"].values == lid)[0]
                if len(idx) > max_per_class:
                    idx = rng.choice(idx, max_per_class, replace=False)
                parts.append(idx)
            selected = np.sort(np.concatenate(parts))
            df_full = pd.read_parquet(data_dir / "test_unknown.parquet")
            df = df_full.iloc[selected].reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            print(f"  Loaded {len(X)} unknown samples ({len(df_ids['label_id'].unique())} classes)")
            return X

        X_known, y_known, C_known = _load_known("test_known")
        X_train, y_train, C_train = _load_known("train")
        X_unknown = _load_unknown()
        concept_names = CIC_CONCEPTS

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return (X_known, y_known, C_known), (X_train, y_train, C_train), X_unknown, concept_names


def rebuild_model(model_name: str, n_features: int, n_classes: int, n_concepts: int):
    if model_name == "MLPBaseline":
        return MLPBaseline(n_features, n_classes, EMBED_DIM)
    elif model_name == "JointCBM":
        return JointCBM(n_features, n_concepts, n_classes, EMBED_DIM)
    elif model_name == "SequentialCBM":
        return SequentialCBM(n_features, n_concepts, n_classes, EMBED_DIM)
    elif model_name == "HybridCBM":
        return HybridCBM(n_features, n_concepts, n_classes, EMBED_DIM)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def mahalanobis_scores(train_vecs: np.ndarray, test_vecs: np.ndarray) -> np.ndarray:
    # Use a single global covariance for numerical stability with small n_concepts
    cov = EmpiricalCovariance(assume_centered=False)
    cov.fit(train_vecs)

    # Per-class means
    scores = cov.mahalanobis(test_vecs)  # shape (N,)
    return scores  # higher = more anomalous


def ood_auroc(
    model, X_train: np.ndarray, y_train: np.ndarray,
    X_known: np.ndarray, X_unknown: np.ndarray, device: torch.device
) -> float:
    model.eval()
    with torch.no_grad():
        emb_train = model.get_embedding(
            torch.tensor(X_train, dtype=torch.float32, device=device)
        ).cpu().numpy()
        emb_known = model.get_embedding(
            torch.tensor(X_known, dtype=torch.float32, device=device)
        ).cpu().numpy()
        emb_unknown = model.get_embedding(
            torch.tensor(X_unknown, dtype=torch.float32, device=device)
        ).cpu().numpy()

    # Limit size for covariance fitting
    MAX_TRAIN = 20000
    if len(emb_train) > MAX_TRAIN:
        idx = np.random.choice(len(emb_train), MAX_TRAIN, replace=False)
        emb_train = emb_train[idx]

    scores_known = mahalanobis_scores(emb_train, emb_known)
    scores_unknown = mahalanobis_scores(emb_train, emb_unknown)

    MAX_EVAL = 5000
    if len(scores_known) > MAX_EVAL:
        idx = np.random.choice(len(scores_known), MAX_EVAL, replace=False)
        scores_known = scores_known[idx]
    if len(scores_unknown) > MAX_EVAL:
        idx = np.random.choice(len(scores_unknown), MAX_EVAL, replace=False)
        scores_unknown = scores_unknown[idx]

    all_scores = np.concatenate([scores_known, scores_unknown])
    all_labels = np.concatenate([
        np.zeros(len(scores_known)),
        np.ones(len(scores_unknown))
    ])

    try:
        auroc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auroc = float("nan")
    return auroc


def tpr_at_fpr(
    model, X_train: np.ndarray,
    X_known: np.ndarray, X_unknown: np.ndarray,
    device: torch.device, fpr_target: float = 0.05
) -> float:
    model.eval()
    with torch.no_grad():
        emb_train = model.get_embedding(
            torch.tensor(X_train, dtype=torch.float32, device=device)
        ).cpu().numpy()
        emb_known = model.get_embedding(
            torch.tensor(X_known, dtype=torch.float32, device=device)
        ).cpu().numpy()
        emb_unknown = model.get_embedding(
            torch.tensor(X_unknown, dtype=torch.float32, device=device)
        ).cpu().numpy()

    MAX_TRAIN = 20000
    if len(emb_train) > MAX_TRAIN:
        idx = np.random.choice(len(emb_train), MAX_TRAIN, replace=False)
        emb_train = emb_train[idx]

    scores_known = mahalanobis_scores(emb_train, emb_known)
    scores_unknown = mahalanobis_scores(emb_train, emb_unknown)

    threshold = np.percentile(scores_known, (1 - fpr_target) * 100)
    tpr = (scores_unknown > threshold).mean()
    return float(tpr)


def concept_accuracies(
    model, X: np.ndarray, C_gt: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        concept_preds = model.get_embedding(
            torch.tensor(X, dtype=torch.float32, device=device)
        ).cpu().numpy()
    # Binarize at 0.5
    binary_preds = (concept_preds > 0.5).astype(np.float32)
    accs = (binary_preds == C_gt).mean(axis=0)
    return accs


def intervention_experiment(
    model, X: np.ndarray, y: np.ndarray, C_gt: np.ndarray,
    device: torch.device, concept_names: list
) -> dict:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    C_t = torch.tensor(C_gt, dtype=torch.float32, device=device)
    y_np = y

    model.eval()

    # Baseline accuracy
    with torch.no_grad():
        logits_base, _ = model(X_t)
    preds_base = logits_base.argmax(dim=1).cpu().numpy()
    acc_base = (preds_base == y_np).mean()

    deltas = {}
    for j, cname in enumerate(concept_names):
        with torch.no_grad():
            logits_int = model.intervene(X_t, j, C_t[:, j])
        preds_int = logits_int.argmax(dim=1).cpu().numpy()
        acc_int = (preds_int == y_np).mean()
        deltas[cname] = float(acc_int - acc_base)

    return deltas, float(acc_base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ctu", "cic"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+",
                        choices=["MLPBaseline", "JointCBM", "SequentialCBM", "HybridCBM"],
                        default=["MLPBaseline", "JointCBM", "SequentialCBM", "HybridCBM"])
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Load checkpoint trained with this gamma value")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    np.random.seed(42)
    gamma_tag = f"_g{args.gamma}" if args.gamma > 0 else ""

    print(f"\n{'='*60}")
    print(f"  Evaluation — dataset={args.dataset}  gamma={args.gamma}  device={device}")
    print(f"{'='*60}\n")

    (X_known, y_known, C_known), (X_train, y_train, C_train), X_unknown, concept_names = \
        load_test_data(args.dataset)

    model_names = args.models
    has_concepts = {"JointCBM", "SequentialCBM", "HybridCBM"}

    all_results = {}

    for model_name in model_names:
        ckpt_path = RESULTS_DIR / f"{args.dataset}_{model_name}{gamma_tag}.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {model_name} — checkpoint not found: {ckpt_path}")
            continue

        print(f"\n--- Evaluating {model_name} ---")
        ckpt = torch.load(ckpt_path, map_location=device)

        n_features = ckpt["n_features"]
        n_classes = ckpt["n_classes"]
        n_concepts = ckpt["n_concepts"]

        model = rebuild_model(model_name, n_features, n_classes, n_concepts)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        results = {"model": model_name, "dataset": args.dataset}

        X_t = torch.tensor(X_known, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()
        weighted_f1 = f1_score(y_known, preds, average="weighted", zero_division=0)
        results["weighted_f1"] = float(weighted_f1)
        print(f"  Weighted F1 (test_known): {weighted_f1:.4f}")

        if model_name in has_concepts:
            con_accs = concept_accuracies(model, X_known, C_known, device)
            results["concept_accuracies"] = {
                name: float(acc) for name, acc in zip(concept_names, con_accs)
            }
            results["mean_concept_accuracy"] = float(con_accs.mean())
            print(f"  Mean concept accuracy: {con_accs.mean():.4f}")
            for name, acc in zip(concept_names, con_accs):
                print(f"    {name:<28}: {acc:.4f}")
        else:
            results["concept_accuracies"] = None
            results["mean_concept_accuracy"] = None

        try:
            auroc = ood_auroc(model, X_train, y_train, X_known, X_unknown, device)
            results["ood_auroc"] = float(auroc)
            print(f"  OOD AUROC (Mahalanobis): {auroc:.4f}")
        except Exception as e:
            results["ood_auroc"] = None
            print(f"  OOD AUROC failed: {e}")

        try:
            tpr = tpr_at_fpr(model, X_train, X_known, X_unknown, device, fpr_target=0.05)
            results["tpr_at_5pct_fpr"] = float(tpr)
            print(f"  TPR @ 5% FPR: {tpr:.4f}")
        except Exception as e:
            results["tpr_at_5pct_fpr"] = None
            print(f"  TPR@5%FPR failed: {e}")

        if model_name in has_concepts:
            try:
                deltas, acc_base = intervention_experiment(
                    model, X_known, y_known, C_known, device, concept_names
                )
                results["intervention_base_acc"] = acc_base
                results["intervention_deltas"] = deltas
                print(f"  Intervention experiment (base acc={acc_base:.4f}):")
                for cname, delta in sorted(deltas.items(), key=lambda x: -abs(x[1])):
                    sign = "+" if delta >= 0 else ""
                    print(f"    {cname:<28}: {sign}{delta:.4f}")
            except Exception as e:
                results["intervention_deltas"] = None
                print(f"  Intervention failed: {e}")
        else:
            results["intervention_deltas"] = None

        all_results[model_name] = results

    print(f"\n{'='*70}")
    print(f"  SUMMARY — {args.dataset.upper()}")
    print(f"{'='*70}")
    print(f"  {'Model':<18}  {'F1':>7}  {'AUROC':>7}  {'TPR@5%':>7}  {'MeanConceptAcc':>14}")
    print(f"  {'-'*65}")
    for mname, res in all_results.items():
        f1_str = f"{res['weighted_f1']:.4f}" if res.get("weighted_f1") is not None else "   N/A"
        auroc_str = f"{res['ood_auroc']:.4f}" if res.get("ood_auroc") is not None else "   N/A"
        tpr_str = f"{res['tpr_at_5pct_fpr']:.4f}" if res.get("tpr_at_5pct_fpr") is not None else "   N/A"
        mca_str = f"{res['mean_concept_accuracy']:.4f}" if res.get("mean_concept_accuracy") is not None else "           N/A"
        print(f"  {mname:<18}  {f1_str:>7}  {auroc_str:>7}  {tpr_str:>7}  {mca_str:>14}")

    # Save results JSON
    out_path = RESULTS_DIR / f"{args.dataset}_eval_results{gamma_tag}.json"
    import json as _json
    with open(out_path, "w") as f:
        _json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
