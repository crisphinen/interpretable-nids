# training script for nesy-nids on ctu-iot-23 and cic-iot-2023.
#
# usage:
# python -m nesy.train --dataset ctu
# python -m nesy.train --dataset cic
# python -m nesy.train --dataset ctu --seed 42 --lambda_sparse 0.01
# python -m nesy.train --dataset ctu --model rule_only

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nesy.model import (
    NeSyNIDS, RuleOnlyNIDS,
    CTU_RULES, CIC_RULES,
)

# config

LR            = 1e-3
N_EPOCHS      = 60
PATIENCE      = 12
BATCH_SIZE    = 512
LAMBDA_SPARSE = 0.0   # rule importance sparsity regulariser (default off)
K_INIT        = 1.0   # initial gate steepness
K_FINAL       = 10.0  # final gate steepness (hard gates)
K_WARMUP      = 30    # epochs to ramp k from K_INIT to K_FINAL

RESULTS_DIR = PROJECT_ROOT / "nesy" / "results"


# k-annealing schedule

# linear ramp from K_INIT to K_FINAL over K_WARMUP epochs.
def get_k(epoch: int) -> float:
    frac = min(epoch / K_WARMUP, 1.0)
    return K_INIT + frac * (K_FINAL - K_INIT)


# data loading (reuse cbm's preprocessing)

def load_dataset(dataset: str):
    if dataset == "ctu":
        data_dir = PROJECT_ROOT / "data"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        known_ids = vocab["known_ids"]

        import pandas as pd
        def _load(split):
            df = pd.read_parquet(data_dir / f"{split}.parquet")
            mask = df["label_id"].isin(known_ids)
            df = df[mask].reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}
            y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
            return X, y

        X_tr, y_tr = _load("train")
        X_val, y_val = _load("val")
        n_classes = len(np.unique(y_tr))
        rule_templates = CTU_RULES

    elif dataset == "cic":
        data_dir = PROJECT_ROOT / "data" / "cic"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        l2i = vocab["label_to_id"]
        known_classes = vocab["known_classes"]
        known_ids = [l2i[c] for c in known_classes]

        import pandas as pd
        def _load(split):
            df = pd.read_parquet(data_dir / f"{split}.parquet")
            mask = df["label_id"].isin(known_ids)
            df = df[mask].reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}
            y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
            return X, y

        X_tr, y_tr = _load("train")
        X_val, y_val = _load("val")
        n_classes = len(np.unique(y_tr))
        rule_templates = CIC_RULES

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_features = X_tr.shape[1]
    print(f"  Train: {X_tr.shape}, Val: {X_val.shape}")
    print(f"  Classes: {n_classes}, Rules: {len(rule_templates)}")
    return X_tr, y_tr, X_val, y_val, n_features, n_classes, rule_templates


def make_loader(X, y, device, shuffle=True):
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# training loop

def compute_val_f1(model, loader, device, k):
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits, _ = model(xb, k=k)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_y.append(yb.cpu().numpy())
    return f1_score(
        np.concatenate(all_y), np.concatenate(all_preds),
        average="weighted", zero_division=0
    )


def train_nesy(model, loader_tr, loader_val, device, label,
               lambda_sparse=0.0, lambda_alpha=0.0):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, N_EPOCHS + 1):
        k = get_k(epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in loader_tr:
            optimizer.zero_grad()
            logits, rule_scores = model(xb, k=k)
            loss = ce_loss(logits, yb)

            # sparsity regulariser: encourage rules to have low activation on average
            # (rules should fire selectively, not always-on)
            if lambda_sparse > 0.0 and hasattr(model, 'rule_bank'):
                mean_activation = rule_scores.mean()
                loss = loss + lambda_sparse * mean_activation

            # alpha-regularisation: penalise low rule weight, pushing model to rely
            # more on rules and less on neural fallback (higher interpretability)
            if lambda_alpha > 0.0 and hasattr(model, 'gate'):
                alpha = torch.sigmoid(model.gate)
                loss = loss + lambda_alpha * (1.0 - alpha) ** 2

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_f1 = compute_val_f1(model, loader_val, device, k)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / max(n_batches, 1)
            gate_str = ""
            if hasattr(model, 'gate'):
                gate_str = f"  alpha={model.get_gate_value():.3f}"
            print(f"  [{label}] Epoch {epoch:3d}/{N_EPOCHS}  k={k:.1f}  "
                  f"loss={avg_loss:.4f}  val_f1={val_f1:.4f}  "
                  f"patience={patience_counter}/{PATIENCE}{gate_str}")

        if patience_counter >= PATIENCE:
            print(f"  [{label}] Early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model, best_val_f1


# main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ctu", "cic"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda_sparse", type=float, default=0.0)
    parser.add_argument("--lambda_alpha", type=float, default=0.0,
                        help="Penalty on (1-alpha)^2 to push model toward higher rule weight.")
    parser.add_argument("--model", choices=["nesy", "rule_only"], default="nesy")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    print(f"\n{'='*60}")
    print(f"  NeSy-NIDS Training")
    print(f"  dataset={args.dataset}  model={args.model}  seed={args.seed}  device={device}")
    print(f"  lambda_sparse={args.lambda_sparse}  lambda_alpha={args.lambda_alpha}")
    print(f"{'='*60}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_tr, y_tr, X_val, y_val, n_features, n_classes, rule_templates = load_dataset(args.dataset)

    loader_tr  = make_loader(X_tr, y_tr, device, shuffle=True)
    loader_val = make_loader(X_val, y_val, device, shuffle=False)

    # build model
    if args.model == "nesy":
        model = NeSyNIDS(n_features, n_classes, rule_templates).to(device)
    else:
        model = RuleOnlyNIDS(n_features, n_classes, rule_templates).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    label = f"{args.model}_{args.dataset}_s{args.seed}"
    model, best_f1 = train_nesy(
        model, loader_tr, loader_val, device, label,
        lambda_sparse=args.lambda_sparse,
        lambda_alpha=args.lambda_alpha,
    )

    print(f"\n  Best val F1: {best_f1:.4f}")

    # save checkpoint - tags encode non-default hyperparams
    sparse_tag = f"_sp{args.lambda_sparse}" if args.lambda_sparse > 0 else ""
    alpha_tag  = f"_a{args.lambda_alpha}"   if args.lambda_alpha  > 0 else ""
    ckpt_path = RESULTS_DIR / f"{args.dataset}_{args.model}_s{args.seed}{sparse_tag}{alpha_tag}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_type": args.model,
        "dataset": args.dataset,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_rules": len(rule_templates),
        "rule_names": [t.name for t in rule_templates],
        "val_f1": best_f1,
        "seed": args.seed,
        "lambda_sparse": args.lambda_sparse,
        "lambda_alpha": args.lambda_alpha,
    }, ckpt_path)
    print(f"  Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
