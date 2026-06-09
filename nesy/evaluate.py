# evaluation script for nesy-nids.
#
# metrics:
# - known f1 (weighted)
# - ood auroc (rule-confidence score)
# - tpr@5%fpr
# - rule crispness (fraction of hard activations at k=10)
# - per-rule class selectivity (which class activates each rule)
# - rule editing: set one rule to 1/0, measure accuracy change (like cbm interventions)
# - gate value alpha (rule vs neural balance)
#
# usage:
# python -m nesy.evaluate --dataset ctu
# python -m nesy.evaluate --dataset cic
# python -m nesy.evaluate --dataset ctu --multi_seed 5

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nesy.model import NeSyNIDS, RuleOnlyNIDS, CTU_RULES, CIC_RULES

RESULTS_DIR = PROJECT_ROOT / "nesy" / "results"
K_EVAL = 10.0   # hard k for evaluation and rule analysis


# data loading

def load_known_val(dataset: str, device: torch.device):
    if dataset == "ctu":
        data_dir = PROJECT_ROOT / "data"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        known_ids = vocab["known_ids"]
        df = pd.read_parquet(data_dir / "val.parquet")
        mask = df["label_id"].isin(known_ids)
        df = df[mask].reset_index(drop=True)
        X = df[feature_cols].values.astype(np.float32)
        id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}
        y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
        id_to_label = {i: vocab["id_to_label"][str(orig)] for i, orig in enumerate(sorted(known_ids))}
    elif dataset == "cic":
        data_dir = PROJECT_ROOT / "data" / "cic"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        l2i = vocab["label_to_id"]
        known_classes = vocab["known_classes"]
        known_ids = [l2i[c] for c in known_classes]
        df = pd.read_parquet(data_dir / "val.parquet")
        mask = df["label_id"].isin(known_ids)
        df = df[mask].reset_index(drop=True)
        X = df[feature_cols].values.astype(np.float32)
        id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}
        y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
        id_to_label = {i: c for i, (orig, c) in enumerate(
            sorted([(lid, cls) for cls, lid in l2i.items() if lid in known_ids])
        )}
    else:
        raise ValueError(dataset)

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    return Xt, yt, id_to_label


# load unknown (ood) test samples. subsample to n_sample.
def load_unknown_test(dataset: str, device: torch.device, n_sample: int = 10000):
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
        X = df[feature_cols].values.astype(np.float32)

    elif dataset == "cic":
        data_dir = PROJECT_ROOT / "data" / "cic"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        l2i = vocab["label_to_id"]
        known_ids = set([l2i[c] for c in vocab["known_classes"]])
        # cic unknown test is huge - stratified subsample
        test_path = data_dir / "test_unknown.parquet"
        label_col = pd.read_parquet(test_path, columns=["label_id"])["label_id"]
        unknown_mask = ~label_col.isin(known_ids)
        unknown_ids_all = label_col[unknown_mask]
        per_class = max(1, n_sample // len(unknown_ids_all.unique()))
        sampled_idx = []
        for lid in unknown_ids_all.unique():
            idx = unknown_ids_all[unknown_ids_all == lid].index.tolist()
            sampled_idx.extend(idx[:per_class])
        sampled_idx = sampled_idx[:n_sample]
        df = pd.read_parquet(test_path, columns=feature_cols + ["label_id"])
        df = df.iloc[sampled_idx].reset_index(drop=True)
        X = df[feature_cols].values.astype(np.float32)

    return torch.tensor(X, dtype=torch.float32, device=device)


# model loading

def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dataset = ckpt["dataset"]
    n_features = ckpt["n_features"]
    n_classes = ckpt["n_classes"]
    model_type = ckpt.get("model_type", "nesy")
    rule_templates = CTU_RULES if dataset == "ctu" else CIC_RULES

    if model_type == "nesy":
        model = NeSyNIDS(n_features, n_classes, rule_templates)
    else:
        model = RuleOnlyNIDS(n_features, n_classes, rule_templates)

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


# core metrics

@torch.no_grad()
def compute_known_f1(model, X_known, y_known):
    logits, _ = model(X_known, k=K_EVAL)
    preds = logits.argmax(dim=1).cpu().numpy()
    return f1_score(y_known.cpu().numpy(), preds, average="weighted", zero_division=0)


# fit class-conditional gaussian on emb_known, score test samples.
# uses global covariance (pooled), per-class mean.
def _mahalanobis_scores(emb_known: np.ndarray, emb_unknown: np.ndarray,
                        cap_train: int = 20000):
    from numpy.linalg import pinv
    if len(emb_known) > cap_train:
        idx = np.random.choice(len(emb_known), cap_train, replace=False)
        emb_fit = emb_known[idx]
    else:
        emb_fit = emb_known

    cov = np.cov(emb_fit.T) + 1e-6 * np.eye(emb_fit.shape[1])
    cov_inv = pinv(cov)
    mu = emb_fit.mean(axis=0)  # global mean (single-class mahalanobis)

    def maha(E):
        diff = E - mu
        return np.sum(diff @ cov_inv * diff, axis=1)

    return maha(emb_known), maha(emb_unknown)


# ood auroc using mahalanobis distance on rule activation vectors.
# rule activations form an interpretable embedding space; mahalanobis on
# this space measures how far a sample is from the known-traffic manifold.
@torch.no_grad()
def compute_ood_auroc(model, X_known, X_unknown):
    emb_known  = model.get_embedding(X_known,  k=K_EVAL).cpu().numpy()
    emb_unknown = model.get_embedding(X_unknown, k=K_EVAL).cpu().numpy()

    if len(emb_known) > 5000:
        idx = np.random.choice(len(emb_known), 5000, replace=False)
        emb_known_s = emb_known[idx]
    else:
        emb_known_s = emb_known

    if len(emb_unknown) > 5000:
        idx = np.random.choice(len(emb_unknown), 5000, replace=False)
        emb_unknown_s = emb_unknown[idx]
    else:
        emb_unknown_s = emb_unknown

    score_known, score_unknown = _mahalanobis_scores(emb_known_s, emb_unknown_s)

    scores = np.concatenate([score_known, score_unknown])
    labels = np.concatenate([
        np.zeros(len(score_known)),
        np.ones(len(score_unknown))
    ])
    return roc_auc_score(labels, scores)


# tpr@5%fpr for ood detection using mahalanobis on rule activations.
def tpr_at_fpr(model, X_known, X_unknown, target_fpr=0.05):
    with torch.no_grad():
        emb_known  = model.get_embedding(X_known,  k=K_EVAL).cpu().numpy()
        emb_unknown = model.get_embedding(X_unknown, k=K_EVAL).cpu().numpy()

    if len(emb_known) > 5000:
        emb_known = emb_known[np.random.choice(len(emb_known), 5000, replace=False)]
    if len(emb_unknown) > 5000:
        emb_unknown = emb_unknown[np.random.choice(len(emb_unknown), 5000, replace=False)]

    score_known, score_unknown = _mahalanobis_scores(emb_known, emb_unknown)
    threshold = np.percentile(score_known, (1 - target_fpr) * 100)
    tpr = (score_unknown >= threshold).mean()
    return float(tpr)


# rule analysis

# for each rule, fraction of samples with activation in [0, 0.1) or (0.9, 1].
# high = rule is making crisp binary decisions.
@torch.no_grad()
def rule_crispness(model, X: torch.Tensor, k: float = K_EVAL) -> Dict[str, float]:
    scores = model.rule_bank.get_rule_activations(X, k)  # (N, M)
    result = {}
    for r_idx, rule in enumerate(model.rule_bank.rules):
        s = scores[:, r_idx].cpu().numpy()
        crisp = ((s < 0.1) | (s > 0.9)).mean()
        result[rule.name] = float(crisp)
    return result


# per-rule mean activation per class. shows which rules are class-selective.
@torch.no_grad()
def rule_class_selectivity(model, X: torch.Tensor, y: torch.Tensor,
                           id_to_label: dict, k: float = K_EVAL):
    scores = model.rule_bank.get_rule_activations(X, k)  # (N, M)
    n_classes = len(id_to_label)
    result = {}
    for r_idx, rule in enumerate(model.rule_bank.rules):
        s = scores[:, r_idx].cpu()
        class_means = {}
        for cls_id in range(n_classes):
            mask = (y.cpu() == cls_id)
            if mask.any():
                class_means[id_to_label.get(cls_id, str(cls_id))] = float(s[mask].mean())
            else:
                class_means[id_to_label.get(cls_id, str(cls_id))] = 0.0
        result[rule.name] = class_means
    return result


# rule editing: for each rule r, force activation to 1.0 (rule fires) and 0.0 (rule suppressed).
# measure accuracy change. analogous to cbm concept interventions.
# a rule is 'editable' if forcing it high for the class that should activate it increases accuracy.
@torch.no_grad()
def rule_editing_experiment(model, X: torch.Tensor, y: torch.Tensor, k: float = K_EVAL):
    # baseline accuracy
    logits_base, rule_scores = model(X, k=k)
    base_acc = (logits_base.argmax(dim=1) == y).float().mean().item()

    results = {}
    for r_idx, rule in enumerate(model.rule_bank.rules):
        # get rule activations
        s = rule_scores[:, r_idx]  # (N,)

        # force rule to 1.0 (rule always fires)
        rule_scores_forced_hi = rule_scores.clone()
        rule_scores_forced_hi[:, r_idx] = 1.0

        # force rule to 0.0 (rule never fires)
        rule_scores_forced_lo = rule_scores.clone()
        rule_scores_forced_lo[:, r_idx] = 0.0

        # recompute logits using forced rule scores
        # we need to go through the W matrix directly
        if hasattr(model, 'rule_bank'):
            weighted_hi = rule_scores_forced_hi * torch.sigmoid(
                model.rule_bank.rule_importance
            ).unsqueeze(0)
            weighted_lo = rule_scores_forced_lo * torch.sigmoid(
                model.rule_bank.rule_importance
            ).unsqueeze(0)
            logits_hi_rule = model.rule_bank.W(weighted_hi)
            logits_lo_rule = model.rule_bank.W(weighted_lo)

            if hasattr(model, 'gate'):
                alpha = torch.sigmoid(model.gate)
                neural_logits = model.neural_fallback(X)
                logits_hi = alpha * logits_hi_rule + (1.0 - alpha) * neural_logits
                logits_lo = alpha * logits_lo_rule + (1.0 - alpha) * neural_logits
            else:
                logits_hi = logits_hi_rule
                logits_lo = logits_lo_rule
        else:
            continue

        acc_hi = (logits_hi.argmax(dim=1) == y).float().mean().item()
        acc_lo = (logits_lo.argmax(dim=1) == y).float().mean().item()

        results[rule.name] = {
            "base_acc": base_acc,
            "acc_forced_hi": acc_hi,
            "acc_forced_lo": acc_lo,
            "delta_hi": acc_hi - base_acc,
            "delta_lo": acc_lo - base_acc,
        }
    return results


# print helpers

def print_rule_summary(crispness, selectivity, editing, id_to_label):
    print(f"\n{'-'*60}")
    print(f"  Rule Analysis (k={K_EVAL})")
    print(f"{'-'*60}")

    rule_names = list(crispness.keys())
    class_names = list(next(iter(selectivity.values())).keys())

    # crispness table
    print(f"\n  {'Rule':<30}  {'Crisp':>7}  {'DAcc(hi)':>9}  {'DAcc(lo)':>9}")
    print(f"  {'-'*62}")
    for rname in rule_names:
        crisp = crispness.get(rname, 0.0)
        edit = editing.get(rname, {})
        dhi = edit.get("delta_hi", float('nan'))
        dlo = edit.get("delta_lo", float('nan'))
        print(f"  {rname:<30}  {crisp:>7.3f}  {dhi:>+9.4f}  {dlo:>+9.4f}")

    # selectivity table
    print(f"\n  Rule class selectivity (mean activation per class at k={K_EVAL}):")
    header = f"  {'Rule':<30}" + "".join(f"{c[:10]:>12}" for c in class_names)
    print(header)
    print(f"  {'-'*80}")
    for rname in rule_names:
        sel = selectivity.get(rname, {})
        row = f"  {rname:<30}" + "".join(f"{sel.get(c, 0.0):>12.3f}" for c in class_names)
        print(row)


# multi-seed evaluation

# load models from n_seeds seeds, report mean +/- std.
def multi_seed_eval(dataset: str, n_seeds: int, device: torch.device):
    X_known, y_known, id_to_label = load_known_val(dataset, device)
    X_unknown = load_unknown_test(dataset, device)

    results = {"f1": [], "auroc": [], "tpr5": []}
    for seed in range(n_seeds):
        ckpt_path = RESULTS_DIR / f"{dataset}_nesy_s{seed}.pt"
        if not ckpt_path.exists():
            print(f"  Missing: {ckpt_path} - skipping")
            continue
        model, _ = load_model(ckpt_path, device)
        f1   = compute_known_f1(model, X_known, y_known)
        auroc = compute_ood_auroc(model, X_known, X_unknown)
        tpr  = tpr_at_fpr(model, X_known, X_unknown)
        results["f1"].append(f1)
        results["auroc"].append(auroc)
        results["tpr5"].append(tpr)
        print(f"  seed={seed}  F1={f1:.4f}  AUROC={auroc:.4f}  TPR@5%={tpr:.4f}")

    if results["f1"]:
        print(f"\n  Multi-seed summary ({len(results['f1'])} seeds):")
        for k in results:
            vals = results[k]
            print(f"    {k}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    return results


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ctu", "cic"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi_seed", type=int, default=0,
                        help="Evaluate N seeds and report mean+/-std (0=off)")
    parser.add_argument("--model", choices=["nesy", "rule_only"], default="nesy")
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    print(f"\n{'='*60}")
    print(f"  NeSy-NIDS Evaluation - {args.dataset}  model={args.model}")
    print(f"{'='*60}\n")

    if args.multi_seed > 0:
        multi_seed_eval(args.dataset, args.multi_seed, device)
        return

    # single seed evaluation
    ckpt_path = RESULTS_DIR / f"{args.dataset}_{args.model}_s{args.seed}.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Run training first: python -m nesy.train --dataset {args.dataset}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    model, ckpt = load_model(ckpt_path, device)

    print("Loading data...")
    X_known, y_known, id_to_label = load_known_val(args.dataset, device)
    X_unknown = load_unknown_test(args.dataset, device)

    print(f"\n  Known samples: {len(X_known)}")
    print(f"  Unknown (OOD) samples: {len(X_unknown)}")

    # core metrics
    print("\nComputing metrics...")
    f1    = compute_known_f1(model, X_known, y_known)
    auroc = compute_ood_auroc(model, X_known, X_unknown)
    tpr5  = tpr_at_fpr(model, X_known, X_unknown)

    # rule analysis (only for models with rule_bank)
    if hasattr(model, 'rule_bank'):
        crisp   = rule_crispness(model, X_known)
        select  = rule_class_selectivity(model, X_known, y_known, id_to_label)
        editing = rule_editing_experiment(model, X_known, y_known)
        mean_crisp = np.mean(list(crisp.values()))
        gate_val = model.get_gate_value() if hasattr(model, 'gate') else float('nan')
    else:
        crisp = select = editing = {}
        mean_crisp = gate_val = float('nan')

    # results table
    print(f"\n{'='*60}")
    print(f"  Results - {args.dataset.upper()}  {args.model}  seed={args.seed}")
    print(f"{'='*60}")
    print(f"  Known F1 (weighted) : {f1:.4f}")
    print(f"  OOD AUROC           : {auroc:.4f}")
    print(f"  TPR @ 5% FPR        : {tpr5:.4f}")
    print(f"  Mean rule crispness : {mean_crisp:.4f}")
    if not np.isnan(gate_val):
        print(f"  Gate alpha (rule weight): {gate_val:.4f}")

    if hasattr(model, 'rule_bank'):
        print_rule_summary(crisp, select, editing, id_to_label)

        # learned thresholds vs initial
        print(f"\n  Learned thresholds:")
        for rule in model.rule_bank.rules:
            thetas = rule.thresholds.detach().cpu().tolist()
            conds = list(zip(rule.feature_indices, rule.condition_types))
            theta_str = ", ".join(f"feat[{fi}]{ct}theta={th:.3f}"
                                  for (fi, ct), th in zip(conds, thetas))
            print(f"    {rule.name:<30}: {theta_str}")

    # save results
    results_file = RESULTS_DIR / f"{args.dataset}_{args.model}_s{args.seed}_eval.json"
    import json as _json
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "known_f1": f1,
        "ood_auroc": auroc,
        "tpr_at_5fpr": tpr5,
        "mean_crispness": float(mean_crisp),
        "gate_alpha": float(gate_val),
        "crispness_per_rule": crisp,
        "editing_results": {k: {kk: float(vv) for kk, vv in v.items()}
                            for k, v in editing.items()},
    }
    results_file.write_text(_json.dumps(summary, indent=2))
    print(f"\n  Saved: {results_file}")

    # export selectivity matrix for figures.py heatmap
    if select:
        sel_file = RESULTS_DIR / f"{args.dataset}_{args.model}_s{args.seed}_selectivity.json"
        sel_file.write_text(_json.dumps(select, indent=2))
        print(f"  Saved: {sel_file}")


if __name__ == "__main__":
    main()
