#!/usr/bin/env python3
# Temporal (session-respecting) split driver for CTU-IoT-23.
# The saved parquet splits drop the capture timestamp, so a temporal partition
# cannot be rebuilt from data/*.parquet alone.
# We therefore re-pull the known-class rows (with `ts`) from the DuckDB source
# table `preprocessed_sorted`, encode categoricals with the SAME saved vocab as
# the main pipeline (data/vocab.json), and order each known class by capture
# timestamp over its WHOLE timeline: earliest 70% -> train pool, next 15% -> val
# pool, latest 15% -> test pool (test strictly follows train in time). Each pool
# is then subsampled to a per-class budget derived from SAMPLES_PER_CLASS
# (train 0.70, val 0.15, test 0.15) with a fixed split seed, so the split stays
# class-balanced and comparable to the main run while keeping train and test
# genuinely far apart in time. The held-out unknown families are unchanged ->
# reuse data/test_unknown.parquet verbatim.
#
# Reports known-class TEST F1 (weighted) + per-class-Mahalanobis OOD AUROC, the
# same scoring as the main pipeline (model.get_embedding + cbm.evaluate.ood_auroc).
#
# usage: python -m cbm.run_temporal --seeds 3 --device cuda
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
import config as C  # DB_PATH, DUCKDB_BIN, KNOWN_CLASSES, NUMERIC/CAT_FEATURES, ...

from cbm.concepts import ctu_concept_labels, CTU_CONCEPTS
from cbm.model import MLPBaseline, JointCBM, SequentialCBM, HybridCBM
from cbm.train import (train_joint, train_classifier_stage2, make_tensors,
                       make_loader, compute_val_f1, EMBED_DIM, LAMBDA_CONCEPT)
from cbm.evaluate import ood_auroc

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test = remaining 0.15 (latest)
SPLIT_SEED = 42  # fixes the within-pool subsample; constant across model seeds


def pull_known_with_ts():
    """COPY known-class rows (features + ts + label) out of preprocessed_sorted."""
    in_clause = ", ".join(f"'{c}'" for c in C.KNOWN_CLASSES)
    select_cols = ", ".join(C.NUMERIC_FEATURES + C.CAT_FEATURES + ["ts", "detailed_label"])
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf:
        tmp = tf.name
    sql = f"""
    COPY (
        SELECT {select_cols}
        FROM preprocessed_sorted
        WHERE detailed_label IN ({in_clause})
    ) TO '{tmp}' (FORMAT PARQUET, ROW_GROUP_SIZE 100000);
    """
    print("Querying preprocessed_sorted via DuckDB CLI (with ts) ...")
    res = subprocess.run([C.DUCKDB_BIN, C.DB_PATH], input=sql, text=True,
                         capture_output=True)
    if res.returncode != 0:
        print("DuckDB stderr:", res.stderr)
        sys.exit(1)
    df = pd.read_parquet(tmp)
    Path(tmp).unlink(missing_ok=True)
    print(f"  pulled {len(df):,} known-class rows")
    return df


def encode_like_main(df, vocab):
    """Reproduce data/02_make_splits.py feature encoding on the raw pull."""
    for col in C.CAT_FEATURES:
        cmap = vocab["cat_vocabs"][col]
        df[col + "_id"] = df[col].map(cmap).fillna(0).astype(np.float32)
    for col in C.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(np.float32)
    df["label_id"] = df["detailed_label"].map(vocab["label_to_id"]).astype(np.int64)
    return df


def build_temporal_split(mode="strict"):
    """mode='strict': split each class over its WHOLE timeline (train=earliest
    70%, test=latest 15%), then subsample each pool to budget -> train and test
    are far apart in time. mode='mild': cap to the earliest SAMPLES_PER_CLASS
    rows first, then split that early window 70/15/15 -> a gentler shift.
    mode='sample': draw the SAME random SAMPLES_PER_CLASS sample the main split
    uses, then order THAT sample by ts and split 70/15/15 -> a moderate shift
    (train spans early->mid of the sample, test the latest of it)."""
    vocab = json.loads((DATA_DIR / "vocab.json").read_text())
    feats = vocab["feature_cols"]
    df = encode_like_main(pull_known_with_ts(), vocab)

    known_ids = sorted(vocab["label_to_id"][l] for l in C.KNOWN_CLASSES)
    id_map = {orig: i for i, orig in enumerate(known_ids)}  # -> contiguous 0..K-1
    spc = int(vocab.get("samples_per_class", C.SAMPLES_PER_CLASS))
    budget = {"train": int(TRAIN_FRAC * spc), "val": int(VAL_FRAC * spc)}
    budget["test"] = spc - budget["train"] - budget["val"]
    rng = np.random.RandomState(SPLIT_SEED)

    def _prep(sub):
        X = sub[feats].values.astype(np.float32)
        y = np.array([id_map[l] for l in sub["label_id"].values], dtype=np.int64)
        Cc = ctu_concept_labels(sub)
        return X, y, Cc

    def _cap(pool, k):
        if len(pool) <= k:
            return pool
        return pool.iloc[np.sort(rng.permutation(len(pool))[:k])]

    tr_parts, va_parts, te_parts = [], [], []
    for lid in known_ids:
        rows = df[df["label_id"] == lid].sort_values("ts", kind="mergesort")
        if mode == "mild":
            rows = rows.iloc[:spc]  # restrict to earliest window before splitting
        elif mode == "sample":
            # same random per-class sample as the main split, then re-order by ts
            take = min(len(rows), spc)
            rows = rows.sample(n=take, random_state=SPLIT_SEED).sort_values(
                "ts", kind="mergesort")
        n = len(rows)
        n_tr, n_va = int(TRAIN_FRAC * n), int(VAL_FRAC * n)
        tr = _cap(rows.iloc[:n_tr], budget["train"])            # earliest 70%
        va = _cap(rows.iloc[n_tr:n_tr + n_va], budget["val"])   # middle 15%
        te = _cap(rows.iloc[n_tr + n_va:], budget["test"])      # latest 15%
        tr_parts.append(tr); va_parts.append(va); te_parts.append(te)
        lbl = vocab["id_to_label"][str(lid)]
        print(f"  {lbl:16} n={n:8}  train={len(tr)} val={len(va)} test={len(te)}")
    tr = pd.concat(tr_parts); va = pd.concat(va_parts); te = pd.concat(te_parts)

    # unknown families unchanged: reuse the saved, already-encoded parquet
    unk = pd.read_parquet(DATA_DIR / "test_unknown.parquet")

    Xtr, ytr, Ctr = _prep(tr)
    Xva, yva, Cva = _prep(va)
    Xte, yte, Cte = _prep(te)
    Xun = unk[feats].values.astype(np.float32)
    print(f"  train={Xtr.shape} val={Xva.shape} test_known={Xte.shape} "
          f"unknown={Xun.shape}  classes={len(known_ids)}")
    ddos_idx = id_map[vocab["label_to_id"]["DDoS"]]  # contiguous index of DDoS
    return dict(Xtr=Xtr, ytr=ytr, Ctr=Ctr, Xva=Xva, yva=yva, Cva=Cva,
                Xte=Xte, yte=yte, Cte=Cte, Xun=Xun, n_features=Xtr.shape[1],
                n_classes=len(known_ids), n_concepts=len(CTU_CONCEPTS),
                ddos_idx=ddos_idx, class_names=[vocab["id_to_label"][str(l)]
                                                for l in known_ids])


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
    Xte_t, yte_t, Cte_t = make_tensors(d["Xte"], d["yte"], d["Cte"], device)
    ltr = make_loader(Xtr_t, ytr_t, Ctr_t, shuffle=True)
    lva = make_loader(Xva_t, yva_t, Cva_t, shuffle=False)
    lte = make_loader(Xte_t, yte_t, Cte_t, shuffle=False)

    if name == "SequentialCBM":
        model = train_joint(model, ltr, lva, device, label=name, concept_only=True)
        model = train_classifier_stage2(model, ltr, lva, device, label=name)
    else:
        model = train_joint(model, ltr, lva, device, label=name,
                            lambda_concept=LAMBDA_CONCEPT if name != "MLPBaseline" else 0.0,
                            gamma_leakage=gamma)

    f1 = compute_val_f1(model, lte, device)  # TEST-known weighted F1 (all classes)
    # per-class + coherent (DDoS-excluded) F1 on the temporal test set
    model.eval()
    with torch.no_grad():
        out = model(Xte_t)
        logits = out[0] if isinstance(out, tuple) else out
        pred = logits.argmax(1).cpu().numpy()
    yte = d["yte"]
    labels = list(range(d["n_classes"]))
    f1_pc = f1_score(yte, pred, labels=labels, average=None, zero_division=0)
    coherent = [c for c in labels if c != d["ddos_idx"]]
    f1_coh = f1_score(yte, pred, labels=coherent, average="weighted", zero_division=0)
    au = ood_auroc(model, d["Xtr"], d["ytr"], d["Xte"], d["Xun"], device)
    return float(f1), float(f1_coh), [float(x) for x in f1_pc], float(au)


VARIANTS = [("MLPBaseline", 0.0), ("JointCBM_g0", 0.0), ("JointCBM_g0.1", 0.1),
            ("JointCBM_g0.5", 0.5), ("JointCBM_g1.0", 1.0),
            ("SequentialCBM", 0.0), ("HybridCBM", 0.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["strict", "mild", "sample"], default="strict")
    ap.add_argument("--out", default=None, help="override output JSON path")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"[ctu] building temporal (session-respecting) split [mode={args.mode}] ...")
    d = build_temporal_split(mode=args.mode)

    seeds = list(range(args.seeds))
    out = {"known": list(C.KNOWN_CLASSES), "mode": args.mode, "seeds": seeds,
           "class_names": d["class_names"], "ddos_idx": d["ddos_idx"],
           "note": "f1_* = all-class weighted; f1_coherent_* = weighted over "
                   "temporally-coherent families (DDoS excluded); f1_perclass_mean "
                   "aligns with class_names.", "results": {}}
    for vname, gamma in VARIANTS:
        base = "JointCBM" if vname.startswith("JointCBM") else vname
        f1s, cohs, pcs, aus = [], [], [], []
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            f1, coh, pc, au = train_variant(base, gamma, d, device)
            f1s.append(f1); cohs.append(coh); pcs.append(pc); aus.append(au)
            print(f"  {vname:16} seed {s}: F1={f1:.4f} coh3={coh:.4f} "
                  f"AUROC={au:.4f}  per-class={[round(x,3) for x in pc]}")
        out["results"][vname] = {
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)), "f1_all": f1s,
            "f1_coherent_mean": float(np.mean(cohs)), "f1_coherent_std": float(np.std(cohs)),
            "f1_coherent_all": cohs,
            "f1_perclass_mean": [float(x) for x in np.mean(pcs, axis=0)],
            "auroc_mean": float(np.mean(aus)), "auroc_std": float(np.std(aus)),
            "auroc_all": aus,
        }
    dest = Path(args.out) if args.out else PROJECT_ROOT / "results" / "cbm" / "temporal_ctu.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"[ctu] wrote {dest}")


if __name__ == "__main__":
    main()
