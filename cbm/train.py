import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm.concepts import (
    CTU_CONCEPTS, CIC_CONCEPTS,
    ctu_concept_labels, cic_concept_labels,
)
from cbm.model import MLPBaseline, JointCBM, SequentialCBM, HybridCBM



EMBED_DIM = 64
LR = 1e-3
N_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 512
LAMBDA_CONCEPT = 0.5
RESULTS_DIR = PROJECT_ROOT / "cbm" / "results"


def load_dataset(dataset: str):
    if dataset == "ctu":
        data_dir = PROJECT_ROOT / "data"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        known_ids = vocab["known_ids"]

        def _load(split):
            df = pd.read_parquet(data_dir / f"{split}.parquet")
            mask = df["label_id"].isin(known_ids)
            df = df[mask].reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            # Remap label_ids to [0, n_classes-1]
            id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}
            y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
            C = ctu_concept_labels(df)
            return X, y, C

        X_tr, y_tr, C_tr = _load("train")
        X_val, y_val, C_val = _load("val")
        n_concepts = len(CTU_CONCEPTS)
        concept_names = CTU_CONCEPTS

    elif dataset == "cic":
        data_dir = PROJECT_ROOT / "data" / "cic"
        vocab = json.loads((data_dir / "vocab.json").read_text())
        feature_cols = vocab["feature_cols"]
        l2i = vocab["label_to_id"]
        known_classes = vocab["known_classes"]
        known_ids = [l2i[c] for c in known_classes]

        def _load(split):
            df = pd.read_parquet(data_dir / f"{split}.parquet")
            mask = df["label_id"].isin(known_ids)
            df = df[mask].reset_index(drop=True)
            X = df[feature_cols].values.astype(np.float32)
            id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}
            y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)
            C = cic_concept_labels(df)
            return X, y, C

        X_tr, y_tr, C_tr = _load("train")
        X_val, y_val, C_val = _load("val")
        n_concepts = len(CIC_CONCEPTS)
        concept_names = CIC_CONCEPTS

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_features = X_tr.shape[1]
    n_classes = len(np.unique(y_tr))

    print(f"  Train: {X_tr.shape}, Val: {X_val.shape}")
    print(f"  Features: {n_features}, Classes: {n_classes}, Concepts: {n_concepts}")

    return (X_tr, y_tr, C_tr), (X_val, y_val, C_val), n_features, n_classes, n_concepts, concept_names


def make_tensors(X, y, C, device):
    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
        torch.tensor(C, dtype=torch.float32, device=device),
    )


def make_loader(X_t, y_t, C_t, shuffle=True):
    ds = TensorDataset(X_t, y_t, C_t)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)



def compute_val_f1(model, loader, device):
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            logits, _ = model(xb)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_y.append(yb.cpu().numpy())
    preds = np.concatenate(all_preds)
    ys = np.concatenate(all_y)
    return f1_score(ys, preds, average="weighted", zero_division=0)


def train_joint(
    model, loader_tr, loader_val, device, label: str,
    lambda_concept: float = LAMBDA_CONCEPT,
    gamma_leakage: float = 0.0,
    concept_only: bool = False,
):
    """
    Generic joint training loop.

    concept_only=True  →  only BCE on concepts (stage 1 of SequentialCBM)
    concept_only=False →  CE + lambda*BCE + gamma*MSE  (JointCBM / HybridCBM)

    gamma_leakage: weight on concept-fidelity MSE regularizer.
      Penalises ||concept_preds - concept_gt||^2, removing the incentive
      for the model to encode class information in soft concept deviations
      (concept leakage). gamma=0 reproduces the original JointCBM.
    """
    ce_loss  = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=LR)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb, cb in loader_tr:
            optimizer.zero_grad()
            logits, concept_preds = model(xb)

            if concept_only:
                # Stage 1: concept prediction only
                loss = bce_loss(concept_preds, cb)
            else:
                loss = ce_loss(logits, yb)
                if concept_preds is not None:
                    loss = loss + lambda_concept * bce_loss(concept_preds, cb)
                    if gamma_leakage > 0.0:
                        # Concept-fidelity regularizer: penalise soft deviations
                        # from binary ground truth, eliminating leakage channel.
                        loss = loss + gamma_leakage * mse_loss(concept_preds, cb)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if concept_only:
            # For stage-1, track BCE loss improvement as proxy
            val_f1 = -avg_loss  # negative loss so "higher is better"
        else:
            val_f1 = compute_val_f1(model, loader_val, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    [{label}] Epoch {epoch:3d}/{N_EPOCHS}  loss={avg_loss:.4f}  "
                  f"val_f1={val_f1:.4f}  patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"    [{label}] Early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model


def train_classifier_stage2(model, loader_tr, loader_val, device, label: str):
    model.freeze_concept_stage()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb, cb in loader_tr:
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = ce_loss(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        val_f1 = compute_val_f1(model, loader_val, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"    [{label} stage2] Epoch {epoch:3d}/{N_EPOCHS}  loss={avg_loss:.4f}  "
                  f"val_f1={val_f1:.4f}  patience={patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"    [{label} stage2] Early stop at epoch {epoch}")
            break

    model.unfreeze_all()
    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ctu", "cic"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+",
                        choices=["MLPBaseline", "JointCBM", "SequentialCBM", "HybridCBM"],
                        default=["MLPBaseline", "JointCBM", "SequentialCBM", "HybridCBM"],
                        help="Which models to train (default: all)")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Concept-fidelity regulariser weight (0=off). "
                             "Applies to JointCBM and HybridCBM only.")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    gamma_tag = f"_g{args.gamma}" if args.gamma > 0 else ""
    print(f"\n{'='*60}")
    print(f"  Training CBM variants — dataset={args.dataset}  device={device}  gamma={args.gamma}")
    print(f"{'='*60}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    (X_tr, y_tr, C_tr), (X_val, y_val, C_val), n_features, n_classes, n_concepts, concept_names = \
        load_dataset(args.dataset)

    X_tr_t, y_tr_t, C_tr_t = make_tensors(X_tr, y_tr, C_tr, device)
    X_val_t, y_val_t, C_val_t = make_tensors(X_val, y_val, C_val, device)

    loader_tr = make_loader(X_tr_t, y_tr_t, C_tr_t, shuffle=True)
    loader_val = make_loader(X_val_t, y_val_t, C_val_t, shuffle=False)

    all_models = {
        "MLPBaseline":   MLPBaseline(n_features, n_classes, EMBED_DIM),
        "JointCBM":      JointCBM(n_features, n_concepts, n_classes, EMBED_DIM),
        "SequentialCBM": SequentialCBM(n_features, n_concepts, n_classes, EMBED_DIM),
        "HybridCBM":     HybridCBM(n_features, n_concepts, n_classes, EMBED_DIM),
    }
    models_to_train = [(name, all_models[name]) for name in args.models]

    for model_name, model in models_to_train:
        model = model.to(device)
        t0 = time.time()
        print(f"\n--- Training {model_name} (gamma={args.gamma}) ---")

        if model_name == "SequentialCBM":
            print("  Stage 1: concept head training")
            model = train_joint(
                model, loader_tr, loader_val, device,
                label=f"{model_name}-stage1",
                concept_only=True,
            )
            print("  Stage 2: classifier training (concepts frozen)")
            model = train_classifier_stage2(model, loader_tr, loader_val, device, label=model_name)
        else:
            use_gamma = args.gamma if model_name != "MLPBaseline" else 0.0
            model = train_joint(
                model, loader_tr, loader_val, device,
                label=model_name,
                lambda_concept=LAMBDA_CONCEPT if model_name != "MLPBaseline" else 0.0,
                gamma_leakage=use_gamma,
            )

        final_f1 = compute_val_f1(model, loader_val, device)
        elapsed = time.time() - t0
        print(f"  -> Final val F1: {final_f1:.4f}  ({elapsed:.1f}s)")

        ckpt_path = RESULTS_DIR / f"{args.dataset}_{model_name}{gamma_tag}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "dataset": args.dataset,
            "n_features": n_features,
            "n_classes": n_classes,
            "n_concepts": n_concepts,
            "concept_names": concept_names,
            "embed_dim": EMBED_DIM,
            "val_f1": final_f1,
            "gamma": args.gamma,
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    print(f"\nDone — dataset={args.dataset}  gamma={args.gamma}.")


if __name__ == "__main__":
    main()
