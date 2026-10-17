# Build the TON_IoT splits: 5 known classes x 50,000 flows (stratified 70/15/15), every held-out
# family capped at 100,000 as the unknown pool. Deterministic: rows are ordered by a hash of
# (file name, row number, seed) before sampling, so reruns give identical files.
# 18 numeric features are z-scored with a scaler fit on the training split; 5 categoricals are
# integer-encoded from known training rows (0 = unseen). raw_* columns keep the raw Zeek values
# the concept rules read.
#
# usage: python 07_make_splits_ton.py
import json
import os

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ton_config import (DATA_DIR, CSV_GLOB, KNOWN_CLASSES, UNKNOWN_CLASSES, NUMERIC_FEATURES,
                        DNS_FLAGS, CAT_FEATURES, CONCEPT_RAW, SAMPLES_PER_CLASS, UNKNOWN_CAP,
                        VAL_FRAC, TEST_FRAC, RANDOM_SEED)

COLS = sorted(set(NUMERIC_FEATURES + CAT_FEATURES + CONCEPT_RAW))


KEY = ["ts", "src_ip", "src_port", "dst_ip", "dst_port", "filename"]


def pull(con, label, limit):
    # order independent of CSV read order: seeded hash of the flow identity, ties broken on every
    # kept column (rows still tied are identical in everything kept)
    sel = ", ".join(f'"{c}"' for c in COLS)
    key = ", ".join(KEY)
    return con.execute(
        f"select {sel}, type from src where type = ? "
        f"order by hash({key}, {RANDOM_SEED}), {sel} limit {limit}", [label]).df()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"create view src as select * from read_csv('{CSV_GLOB}', union_by_name=true, "
                f"all_varchar=true, filename=true)")
    known = pd.concat([pull(con, c, SAMPLES_PER_CLASS) for c in KNOWN_CLASSES], ignore_index=True)
    unknown = pd.concat([pull(con, c, UNKNOWN_CAP) for c in UNKNOWN_CLASSES], ignore_index=True)
    for name, df in (("known", known), ("unknown", unknown)):
        print(f"{name}: {len(df):,}  " + "  ".join(f"{k}={v:,}" for k, v in df["type"].value_counts().items()))

    label_to_id = {c: i for i, c in enumerate(KNOWN_CLASSES + UNKNOWN_CLASSES)}
    for df in (known, unknown):
        df["label_id"] = df["type"].map(label_to_id)
        for c in DNS_FLAGS:
            df[c] = (df[c] == "T").astype(np.float32)
        for c in sorted(set(NUMERIC_FEATURES) - set(DNS_FLAGS)):
            df[c] = pd.to_numeric(df[c].replace("-", None), errors="coerce").fillna(0).astype(np.float32)
        for c in CONCEPT_RAW:
            df["raw_" + c] = df[c]

    train, tmp = train_test_split(known, test_size=VAL_FRAC + TEST_FRAC,
                                  stratify=known["label_id"], random_state=RANDOM_SEED)
    val, test_known = train_test_split(tmp, test_size=0.5, stratify=tmp["label_id"], random_state=RANDOM_SEED)
    print(f"split: train {len(train):,}  val {len(val):,}  test_known {len(test_known):,}  unknown {len(unknown):,}")

    cat_vocabs = {}
    for c in CAT_FEATURES:
        cat_vocabs[c] = {v: i + 1 for i, v in enumerate(sorted(train[c].dropna().unique().tolist()))}
    scaler = StandardScaler().fit(train[NUMERIC_FEATURES].values)
    frames = {"train": train, "val": val, "test_known": test_known, "test_unknown": unknown}
    for k, df in frames.items():
        df = df.copy()
        df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES].values).astype(np.float32)
        for c in CAT_FEATURES:
            df[c + "_id"] = df[c].map(cat_vocabs[c]).fillna(0).astype(np.float32)
        frames[k] = df

    feature_cols = NUMERIC_FEATURES + [c + "_id" for c in CAT_FEATURES]
    keep = feature_cols + ["raw_" + c for c in CONCEPT_RAW] + ["label_id"]
    for k, df in frames.items():
        df[keep].reset_index(drop=True).to_parquet(os.path.join(DATA_DIR, f"{k}.parquet"), index=False)
    vocab = {"feature_cols": feature_cols, "numeric_zscored": NUMERIC_FEATURES, "categorical": CAT_FEATURES,
             "label_to_id": label_to_id, "known_classes": KNOWN_CLASSES, "unknown_classes": UNKNOWN_CLASSES,
             "cat_vocabs": cat_vocabs, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
             "samples_per_class": SAMPLES_PER_CLASS, "unknown_cap": UNKNOWN_CAP, "seed": RANDOM_SEED}
    with open(os.path.join(DATA_DIR, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"saved {DATA_DIR} ({len(feature_cols)} features)")


if __name__ == "__main__":
    main()
