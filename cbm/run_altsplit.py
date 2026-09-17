#!/usr/bin/env python3
# Reconstructed alternative-split sensitivity driver (original was ad-hoc, lost).
# Re-derives train/val/test_known/test_unknown from pooled data under an
# alternative known-class set, then trains every CBM variant across N seeds and
# reports known-class val F1 + per-class-Mahalanobis OOD AUROC (same scoring as
# the main pipeline, via model.get_embedding + cbm.evaluate.ood_auroc).
#
# usage: python -m cbm.run_altsplit --dataset ctu --seeds 5 --device cuda
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cbm.concepts import (ctu_concept_labels, cic_concept_labels,
                          CTU_CONCEPTS, CIC_CONCEPTS)
from cbm.model import MLPBaseline, JointCBM, SequentialCBM, HybridCBM
from cbm.train import (train_joint, train_classifier_stage2, make_tensors,
                       make_loader, compute_val_f1, EMBED_DIM, LAMBDA_CONCEPT)
from cbm.evaluate import ood_auroc

PROJECT_ROOT = Path(__file__).parent.parent
SPLIT_SEED = 42          # data split fixed across model seeds (matches main run)

# alternative known sets (label_ids), and the split provenance per dataset
ALT = {
    "ctu": {
        "dir": PROJECT_ROOT / "data",
        "known_ids": [0, 2, 3, 4],           # Benign, DDoS, Okiru, C&C
        "samples_per_class": 23000,
        "unk_cap": None,
        "concepts": ctu_concept_labels, "concept_names": CTU_CONCEPTS,
    },
    "cic": {
        "dir": PROJECT_ROOT / "data" / "cic",
        "known_ids": [0, 3, 4, 7, 21],       # Benign, Recon-PortScan, VulnScan, DDoS-ICMP, Mirai-udpplain
        "samples_per_class": 50000,
        "unk_cap": 2000,
        "concepts": cic_concept_labels, "concept_names": CIC_CONCEPTS,
    },
}


def load_pooled(cfg):
    """Concatenate all split parquets so any label can be re-partitioned."""
    dfs = []
    for sp in ["train", "val", "test_known", "test_unknown"]:
        p = cfg["dir"] / f"{sp}.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)
    return df.drop_duplicates().reset_index(drop=True)


def build_altsplit(cfg):
    vocab = json.loads((cfg["dir"] / "vocab.json").read_text())
    feats = vocab["feature_cols"]
    df = load_pooled(cfg)

    known = cfg["known_ids"]
    id_map = {orig: i for i, orig in enumerate(sorted(known))}
    rng = np.random.RandomState(SPLIT_SEED)

    def _prep(sub):
        X = sub[feats].values.astype(np.float32)
        y = np.array([id_map[l] for l in sub["label_id"].values], dtype=np.int64)
        C = cfg["concepts"](sub)
        return X, y, C

    tr_parts, va_parts, te_parts = [], [], []
    for lid in sorted(known):
        rows = df[df["label_id"] == lid]
        idx = rng.permutation(len(rows))
        n = min(len(rows), cfg["samples_per_class"])
        idx = idx[:n]
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        tr_parts.append(rows.iloc[idx[:n_tr]])
        va_parts.append(rows.iloc[idx[n_tr:n_tr + n_va]])
        te_parts.append(rows.iloc[idx[n_tr + n_va:]])
    tr = pd.concat(tr_parts); va = pd.concat(va_parts); te = pd.concat(te_parts)

    # unknown pool = every label not in the alt known set
    unk_parts = []
    for lid in sorted(df["label_id"].unique()):
        if lid in known:
            continue
        rows = df[df["label_id"] == lid]
        if cfg["unk_cap"] and len(rows) > cfg["unk_cap"]:
            rows = rows.iloc[rng.permutation(len(rows))[:cfg["unk_cap"]]]
        unk_parts.append(rows)
    unk = pd.concat(unk_parts)

    Xtr, ytr, Ctr = _prep(tr)
    Xva, yva, Cva = _prep(va)
    Xte, yte, Cte = _prep(te)
    Xun = unk[feats].values.astype(np.float32)
    n_classes = len(known)
    n_concepts = len(cfg["concept_names"])
    print(f"  alt known={sorted(known)}  train={Xtr.shape} val={Xva.shape} "
          f"test_known={Xte.shape} unknown={Xun.shape}  classes={n_classes}")
    return dict(Xtr=Xtr, ytr=ytr, Ctr=Ctr, Xva=Xva, yva=yva, Cva=Cva,
                Xte=Xte, Xun=Xun, n_features=Xtr.shape[1],
                n_classes=n_classes, n_concepts=n_concepts)


def train_variant(name, gamma, d, device):
    nf, nc, nk = d["n_features"], d["n_concepts"], d["n_classes"]
    if name == "MLPBaseline":
        model = MLPBaseline(nf, nk, EMBED_DIM).to(device)
    elif name == "JointCBM":
        model = JointCBM(nf, nc, nk, EMBED_DIM).to(device)
    elif name == "SequentialCBM":
        model = SequentialCBM(nf, nc, nk, EMBED_DIM).to(device)
    elif name == "HybridCBM":
        model = HybridCBM(nf, nc, nk, EMBED_DIM).to(device)

    Xtr_t, ytr_t, Ctr_t = make_tensors(d["Xtr"], d["ytr"], d["Ctr"], device)
    Xva_t, yva_t, Cva_t = make_tensors(d["Xva"], d["yva"], d["Cva"], device)
    ltr = make_loader(Xtr_t, ytr_t, Ctr_t, shuffle=True)
    lva = make_loader(Xva_t, yva_t, Cva_t, shuffle=False)

    if name == "SequentialCBM":
        model = train_joint(model, ltr, lva, device, label=name, concept_only=True)
        model = train_classifier_stage2(model, ltr, lva, device, label=name)
    else:
        model = train_joint(model, ltr, lva, device, label=name,
                            lambda_concept=LAMBDA_CONCEPT if name != "MLPBaseline" else 0.0,
                            gamma_leakage=gamma)

    f1 = compute_val_f1(model, lva, device)
    au = ood_auroc(model, d["Xtr"], d["ytr"], d["Xte"], d["Xun"], device)
    return float(f1), float(au)


VARIANTS = [("MLPBaseline", 0.0), ("JointCBM_g0", 0.0), ("JointCBM_g0.1", 0.1),
            ("JointCBM_g0.5", 0.5), ("JointCBM_g1.0", 1.0),
            ("SequentialCBM", 0.0), ("HybridCBM", 0.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ctu", "cic"], required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    cfg = ALT[args.dataset]
    print(f"[{args.dataset}] building alternative split ...")
    d = build_altsplit(cfg)

    seeds = list(range(args.seeds))
    out = {"known_ids": sorted(cfg["known_ids"]), "seeds": seeds, "results": {}}
    for vname, gamma in VARIANTS:
        base = "JointCBM" if vname.startswith("JointCBM") else vname
        f1s, aus = [], []
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            f1, au = train_variant(base, gamma, d, device)
            f1s.append(f1); aus.append(au)
            print(f"  {vname:16} seed {s}: F1={f1:.4f} AUROC={au:.4f}")
        out["results"][vname] = {
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)), "f1_all": f1s,
            "auroc_mean": float(np.mean(aus)), "auroc_std": float(np.std(aus)), "auroc_all": aus,
        }
    dest = PROJECT_ROOT / "results" / "cbm" / f"altsplit_{args.dataset}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"[{args.dataset}] wrote {dest}")


if __name__ == "__main__":
    main()
