# evaluation script for cbm variants on ctu-iot-23 and cic-iot-2023.
#
# computes:
# - weighted f1 on test_known
# - concept prediction accuracy per concept (cbm models)
# - ood auroc via mahalanobis distance on concept vectors
# - intervention experiment: concept-by-concept correction delta
# - unknown detection: tpr @ 5% fpr threshold
#
# usage:
# python -m cbm.evaluate --dataset ctu
# python -m cbm.evaluate --dataset cic

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

RESULTS_DIR = PROJECT_ROOT / "results" / "cbm"
EMBED_DIM = 64


# data

# load test_known and test_unknown splits with concept labels.
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
            # ctu test_unknown is manageable; no subsampling needed
            df = pd.read_parquet(data_dir / "test_unknown.parquet")
            if len(df) > max_rows:
                df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            return X

        X_known, y_known, C_known = _load_known("test_known")
        # also load train split for mahalanobis fitting
        X_train, y_train, C_train = _load_known("train")
        X_val_known, _, _ = _load_known("val")
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
            # cic test_unknown has 40M rows - stratified subsample per class
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
        X_val_known, _, _ = _load_known("val")
        X_unknown = _load_unknown()
        concept_names = CIC_CONCEPTS

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return (X_known, y_known, C_known), (X_train, y_train, C_train), X_unknown, X_val_known, concept_names


def _load_val_known(dataset: str) -> np.ndarray:
    """Return raw feature array for val_known split (used for OOD calibration)."""
    if dataset == "ctu":
        data_dir = PROJECT_ROOT / "data"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        known_ids = set(vocab["known_ids"])
        df = pd.read_parquet(data_dir / "val.parquet")
        df = df[df["label_id"].isin(known_ids)].reset_index(drop=True)
        return df[feature_cols].values.astype(np.float32)
    elif dataset == "cic":
        data_dir = PROJECT_ROOT / "data" / "cic"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        l2i = vocab["label_to_id"]
        known_ids = set([l2i[c] for c in vocab["known_classes"]])
        df = pd.read_parquet(data_dir / "val.parquet")
        df = df[df["label_id"].isin(known_ids)].reset_index(drop=True)
        return df[feature_cols].values.astype(np.float32)
    else:
        raise ValueError(dataset)


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


# mahalanobis ood detection

# per-class Mahalanobis scorer (Lee et al. 2018).
# fits one Gaussian per class with shared (pooled) covariance, scores each
# test sample as the minimum squared Mahalanobis distance to any class centroid.
def mahalanobis_scores(train_vecs: np.ndarray, test_vecs: np.ndarray,
                        train_labels: np.ndarray = None,
                        eps: float = 1e-4) -> np.ndarray:
    from numpy.linalg import pinv
    if train_labels is not None and len(np.unique(train_labels)) >= 2:
        classes = np.unique(train_labels)
        residuals = np.concatenate([
            train_vecs[train_labels == c] - train_vecs[train_labels == c].mean(axis=0)
            for c in classes if (train_labels == c).sum() >= 2
        ], axis=0)
        cov = np.cov(residuals.T) + eps * np.eye(residuals.shape[1])
        cov_inv = pinv(cov)
        mus = {c: train_vecs[train_labels == c].mean(axis=0)
               for c in classes if (train_labels == c).sum() >= 2}
        dists = np.stack([
            np.sum((test_vecs - mu) @ cov_inv * (test_vecs - mu), axis=1)
            for mu in mus.values()
        ], axis=1)
        return dists.min(axis=1)
    else:
        # global fallback (no labels available, e.g. MLP baseline)
        cov = EmpiricalCovariance(assume_centered=False)
        cov.fit(train_vecs)
        return cov.mahalanobis(test_vecs)


def energy_score(logits: np.ndarray) -> np.ndarray:
    """Energy OOD score: E(x) = -log sum_c exp(l_c(x)).
    Higher value = less evidence for any known class = more OOD."""
    max_l = logits.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(logits - max_l).sum(axis=1)) + max_l.squeeze(1)
    return -log_sum_exp


def _get_embeddings_and_logits(model, X: np.ndarray, device: torch.device,
                                cap: int = 20000):
    """Return (embeddings, logits) for X, with optional size cap."""
    if len(X) > cap:
        idx = np.random.choice(len(X), cap, replace=False)
        X = X[idx]
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xt)
        emb = model.get_embedding(Xt)
    return emb.cpu().numpy(), logits.cpu().numpy()


# fit mahalanobis on train concept vectors.
# score known (label=0) vs unknown (label=1) test samples.
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

    # limit size for covariance fitting
    MAX_TRAIN = 20000
    y_train_fit = y_train
    if len(emb_train) > MAX_TRAIN:
        idx = np.random.choice(len(emb_train), MAX_TRAIN, replace=False)
        emb_train = emb_train[idx]
        y_train_fit = y_train[idx]

    scores_known = mahalanobis_scores(emb_train, emb_known, train_labels=y_train_fit)
    scores_unknown = mahalanobis_scores(emb_train, emb_unknown, train_labels=y_train_fit)

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


def combined_ood_scores(
    mahal: np.ndarray, energy: np.ndarray,
    val_mahal: np.ndarray, val_energy: np.ndarray,
) -> np.ndarray:
    """z-normalise Mahalanobis and energy using val in-distribution stats, then average.
    Returns s_comb = 0.5*(s_tilde + E_tilde); higher = more OOD."""
    mu_s, sig_s = val_mahal.mean(), val_mahal.std() + 1e-8
    mu_e, sig_e = val_energy.mean(), val_energy.std() + 1e-8
    return 0.5 * ((mahal - mu_s) / sig_s + (energy - mu_e) / sig_e)


def ood_full(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val_known: np.ndarray,
    X_known: np.ndarray,
    X_unknown: np.ndarray,
    device: torch.device,
    fpr_target: float = 0.05,
    tau_percentile: float = 95.0,
) -> dict:
    """Compute Mahalanobis (per-class) and energy OOD scores.

    Uses y_train labels for per-class Mahalanobis fitting (Lee et al. 2018).
    Uses X_val_known to calibrate rejection threshold tau = Q_{tau_percentile}.

    Returns dict with auroc and tpr@fpr_target for Mahalanobis,
    plus tau and the open-set rejection rate on unknowns using mahal > tau.
    """
    CAP = 5000
    MAX_TRAIN = 20000

    # subsample training data with consistent label tracking
    y_train_fit = y_train
    X_train_fit = X_train
    if len(X_train) > MAX_TRAIN:
        idx = np.random.choice(len(X_train), MAX_TRAIN, replace=False)
        X_train_fit = X_train[idx]
        y_train_fit = y_train[idx]

    emb_train, _ = _get_embeddings_and_logits(model, X_train_fit, device, cap=MAX_TRAIN + 1)
    emb_val, logits_val = _get_embeddings_and_logits(model, X_val_known, device, cap=CAP)
    emb_known, logits_known = _get_embeddings_and_logits(model, X_known, device, cap=CAP)
    emb_unk, logits_unk = _get_embeddings_and_logits(model, X_unknown, device, cap=CAP)

    # per-class mahalanobis scores
    s_val = mahalanobis_scores(emb_train, emb_val, train_labels=y_train_fit)
    s_known = mahalanobis_scores(emb_train, emb_known, train_labels=y_train_fit)
    s_unk = mahalanobis_scores(emb_train, emb_unk, train_labels=y_train_fit)

    # energy scores (for ablation reporting only)
    e_known = energy_score(logits_known)
    e_unk = energy_score(logits_unk)

    # tau: 95th percentile of val Mahalanobis scores (Eq. tau in paper)
    tau = float(np.percentile(s_val, tau_percentile))

    def _auroc(scores_in, scores_out):
        labels = np.concatenate([np.zeros(len(scores_in)), np.ones(len(scores_out))])
        scores = np.concatenate([scores_in, scores_out])
        try:
            return float(roc_auc_score(labels, scores))
        except Exception:
            return float("nan")

    def _tpr(scores_in, scores_out, fpr):
        thr = np.percentile(scores_in, (1 - fpr) * 100)
        return float((scores_out >= thr).mean())

    return {
        "mahal_auroc":    _auroc(s_known, s_unk),
        "energy_auroc":   _auroc(e_known, e_unk),
        "mahal_tpr5":     _tpr(s_known, s_unk, fpr_target),
        "energy_tpr5":    _tpr(e_known, e_unk, fpr_target),
        "tau":            tau,
        "unk_rejection_rate": float((s_unk > tau).mean()),
        "known_rejection_rate": float((s_known > tau).mean()),
    }


# return tpr (unknown detection rate) at a given fpr on known samples.
def tpr_at_fpr(
    model, X_train: np.ndarray, y_train: np.ndarray,
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

    y_train_fit = y_train
    MAX_TRAIN = 20000
    if len(emb_train) > MAX_TRAIN:
        idx = np.random.choice(len(emb_train), MAX_TRAIN, replace=False)
        emb_train = emb_train[idx]
        y_train_fit = y_train[idx]

    scores_known = mahalanobis_scores(emb_train, emb_known, train_labels=y_train_fit)
    scores_unknown = mahalanobis_scores(emb_train, emb_unknown, train_labels=y_train_fit)

    threshold = np.percentile(scores_known, (1 - fpr_target) * 100)
    tpr = (scores_unknown > threshold).mean()
    return float(tpr)


# concept accuracy

# per-concept binary accuracy between predicted and ground-truth concept labels.
def concept_accuracies(
    model, X: np.ndarray, C_gt: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        concept_preds = model.get_embedding(
            torch.tensor(X, dtype=torch.float32, device=device)
        ).cpu().numpy()
    # binarize at 0.5
    binary_preds = (concept_preds > 0.5).astype(np.float32)
    accs = (binary_preds == C_gt).mean(axis=0)
    return accs


# intervention experiment

# for each concept j:
# - set concept_j = ground_truth for all test samples
# - measure accuracy change vs. baseline (no intervention)
# returns dict: concept_name -> delta_accuracy
def intervention_experiment(
    model, X: np.ndarray, y: np.ndarray, C_gt: np.ndarray,
    device: torch.device, concept_names: list
) -> dict:
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    C_t = torch.tensor(C_gt, dtype=torch.float32, device=device)
    y_np = y

    model.eval()

    # baseline accuracy
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


# main

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
    print(f"  Evaluation - dataset={args.dataset}  gamma={args.gamma}  device={device}")
    print(f"{'='*60}\n")

    (X_known, y_known, C_known), (X_train, y_train, C_train), X_unknown, X_val_known, concept_names = \
        load_test_data(args.dataset)

    model_names = args.models
    has_concepts = {"JointCBM", "SequentialCBM", "HybridCBM"}

    all_results = {}

    for model_name in model_names:
        ckpt_path = RESULTS_DIR / f"{args.dataset}_{model_name}{gamma_tag}.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {model_name} - checkpoint not found: {ckpt_path}")
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

        # 1. weighted f1 on test_known
        X_t = torch.tensor(X_known, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = model(X_t)
        preds = logits.argmax(dim=1).cpu().numpy()
        weighted_f1 = f1_score(y_known, preds, average="weighted", zero_division=0)
        results["weighted_f1"] = float(weighted_f1)
        print(f"  Weighted F1 (test_known): {weighted_f1:.4f}")

        # 2. concept accuracies (cbm models only)
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

        # 3. ood auroc (mahalanobis on concept / embedding vectors)
        try:
            auroc = ood_auroc(model, X_train, y_train, X_known, X_unknown, device)
            results["ood_auroc"] = float(auroc)
            print(f"  OOD AUROC (Mahalanobis): {auroc:.4f}")
        except Exception as e:
            results["ood_auroc"] = None
            print(f"  OOD AUROC failed: {e}")

        # 4. tpr at 5% fpr + tau calibration
        try:
            tpr = tpr_at_fpr(model, X_train, y_train, X_known, X_unknown, device, fpr_target=0.05)
            results["tpr_at_5pct_fpr"] = float(tpr)
            print(f"  TPR @ 5% FPR: {tpr:.4f}")
        except Exception as e:
            results["tpr_at_5pct_fpr"] = None
            print(f"  TPR@5%FPR failed: {e}")

        # 4b. energy OOD scoring (ablation) + tau calibration
        try:
            ood_res = ood_full(
                model, X_train, y_train, X_val_known, X_known, X_unknown, device
            )
            results["energy_auroc"]       = ood_res["energy_auroc"]
            results["tau"]                = ood_res["tau"]
            results["unk_rejection_rate"] = ood_res["unk_rejection_rate"]
            results["known_rejection_rate"] = ood_res["known_rejection_rate"]
            print(f"  OOD AUROC (Energy, ablation): {ood_res['energy_auroc']:.4f}")
            print(f"  Rejection threshold tau: {ood_res['tau']:.4f}  "
                  f"(unk rejected: {ood_res['unk_rejection_rate']*100:.1f}%  "
                  f"known rejected: {ood_res['known_rejection_rate']*100:.1f}%)")
        except Exception as e:
            results["energy_auroc"] = None
            print(f"  Energy OOD failed: {e}")

        # 5. intervention experiment (cbm models only)
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

    # summary table
    print(f"\n{'='*90}")
    print(f"  SUMMARY - {args.dataset.upper()}")
    print(f"{'='*90}")
    print(f"  {'Model':<18}  {'F1':>7}  {'AUROC':>7}  {'TPR@5%':>7}  {'Energy':>7}  {'MCA':>7}")
    print(f"  {'-'*72}")
    for mname, res in all_results.items():
        f1_str    = f"{res['weighted_f1']:.4f}"           if res.get("weighted_f1")           is not None else "   N/A"
        mahal_str = f"{res['ood_auroc']:.4f}"              if res.get("ood_auroc")              is not None else "   N/A"
        tpr_str   = f"{res['tpr_at_5pct_fpr']:.4f}"       if res.get("tpr_at_5pct_fpr")       is not None else "   N/A"
        eng_str   = f"{res['energy_auroc']:.4f}"           if res.get("energy_auroc")           is not None else "   N/A"
        mca_str   = f"{res['mean_concept_accuracy']:.4f}"  if res.get("mean_concept_accuracy")  is not None else "   N/A"
        print(f"  {mname:<18}  {f1_str:>7}  {mahal_str:>7}  {tpr_str:>7}  {eng_str:>7}  {mca_str:>7}")

    # save results json
    out_path = RESULTS_DIR / f"{args.dataset}_eval_results{gamma_tag}.json"
    import json as _json
    with open(out_path, "w") as f:
        _json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
