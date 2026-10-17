# build the leak-free CTU-IoT-23 splits (supplement, "CTU-IoT-23
# Feature Derivation"): features re-selected by random-forest importance on known-class
# training rows only, so no held-out family informs the feature list.
#
# writes two datasets over the SAME rows:
#   data/ctuk/   known-only feature list (target: 28 numeric + 3 categorical = 31)
#   data/ctuko/  the submitted 34 numeric + 3 categorical list (config.NUMERIC_FEATURES)
# so ctuko vs ctuk isolates the feature-selection source from the sample.
#
# pipeline:
# 1. draw SAMPLES_PER_CLASS flows per known class in a deterministic order
#    (hash of rowid and seed), and take every held-out flow as the unknown pool.
# 2. stratified 70/15/15 split of the known sample.
# 3. integer-encode conn_state / history / service from known training rows (0 = unseen).
# 4. class-weighted random forest on known training rows over every numeric candidate;
#    keep the features covering CUM_IMPORTANCE of total importance.
# 5. save parquets (all candidate columns) and a vocab.json per dataset.
#
# usage: python 06_make_kfs_splits.py [--sample-seed 42] [--rf-seed 42]
#        python 06_make_kfs_splits.py --split alt       -> data/ctualt, data/ctualt_o
#        python 06_make_kfs_splits.py --split temporal  -> data/ctut,   data/ctut_o
#        python 06_make_kfs_splits.py --split device --holdout-host 192.168.1.198 --tag _h198
#            -> data/ctud_h198, data/ctud_o_h198: every known-class flow of the held-out host is
#               test-only; train/val come from the other hosts

import argparse
import json
import os
import sys

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import (
    DB_PATH, KNOWN_CLASSES, UNKNOWN_PRIMARY, UNKNOWN_RARE, NUMERIC_FEATURES, CAT_FEATURES,
    SAMPLES_PER_CLASS, VAL_FRAC, TEST_FRAC, RANDOM_SEED,
)

CUM_IMPORTANCE = 0.99
# the known-only list as documented in the supplement: the submitted 34 minus 11 dropped,
# plus 5 added = 28 numeric. Features near the 99% cutoff have importances within noise of
# each other (~0.003), so a re-run forest can land a few features differently; by default
# the documented list is used and the re-run selection is only recorded.
PAPER_DROPPED = ["HTTP", "HTTPS", "SSH", "IRC", "resp_pkts", "fin_count", "Radius", "Std",
                 "Variance", "resp_ip_bytes", "rst_flag_number"]
PAPER_ADDED = ["TCP", "Protocol_Type", "Time_To_Live", "cwr_flag_number", "syn_flag_number"]
PAPER_KFS_NUMERIC = [f for f in NUMERIC_FEATURES if f not in PAPER_DROPPED] + PAPER_ADDED
NON_FEATURES = {"src_ip", "ts", "detailed_label", *CAT_FEATURES}
# features the expert NeSy-NIDS rules read; restored for NeSy if the forest drops them
NESY_RULE_FEATURES = ["IAT", "Rate", "orig_pkts", "duration", "resp_pkts", "syn_count", "fin_count"]


def candidate_columns(con):
    rows = con.execute(
        "select column_name from information_schema.columns "
        "where table_name = 'preprocessed_sorted' order by ordinal_position").fetchall()
    return [r[0] for r in rows if r[0] not in NON_FEATURES]


# alternative known set (main paper, Robustness splits): C&C promoted, C&C-HeartBeat demoted
ALT_KNOWN = ["Benign", "C&C", "DDoS", "Okiru"]
ALT_UNKNOWN_PRIMARY = ["C&C-HeartBeat", "Attack", "C&C-HeartBeat-Attack"]


def pull(con, cols, label, limit, seed, extra=(), host=None, exclude_host=None):
    sel = ", ".join(f'"{c}"' for c in cols + CAT_FEATURES + ["detailed_label", *extra])
    order = f"hash(rowid, {seed})"
    lim = f"limit {limit}" if limit else ""
    where, params = "detailed_label = ?", [label]
    if host:
        where += " and src_ip = ?"; params.append(host)
    if exclude_host:
        where += " and src_ip <> ?"; params.append(exclude_host)
    q = f"select {sel} from preprocessed_sorted where {where} order by {order} {lim}"
    return con.execute(q, params).df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--rf-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--n-trees", type=int, default=300)
    ap.add_argument("--use-rerun-selection", action="store_true",
                    help="use this run's forest selection instead of the documented list")
    ap.add_argument("--tag", default="", help="suffix for the output dirs, e.g. _r1 -> data/ctuk_r1/")
    ap.add_argument("--holdout-host", help="device split: source IP whose known-class flows are test-only")
    ap.add_argument("--split", choices=["random", "alt", "temporal", "device"], default="random",
                    help="random: main split (ctuk/ctuko); alt: alternative known set (ctualt); "
                         "temporal: same sample as main, each class split by capture time (ctut)")
    args = ap.parse_args()
    known_classes = ALT_KNOWN if args.split == "alt" else KNOWN_CLASSES
    unknown_primary = ALT_UNKNOWN_PRIMARY if args.split == "alt" else UNKNOWN_PRIMARY
    ts = ("ts",) if args.split == "temporal" else ()

    con = duckdb.connect(DB_PATH, read_only=True)
    cands = candidate_columns(con)
    print(f"{len(cands)} numeric candidate features")

    # 1. deterministic sample: known classes capped, unknowns in full
    known_parts = []
    n_test = int(TEST_FRAC * SAMPLES_PER_CLASS)
    for cls in known_classes:
        if args.split == "device":
            te = pull(con, cands, cls, n_test, args.sample_seed, host=args.holdout_host).assign(_part="test")
            tv = pull(con, cands, cls, SAMPLES_PER_CLASS - n_test, args.sample_seed,
                      exclude_host=args.holdout_host).assign(_part="trainval")
            part = pd.concat([tv, te], ignore_index=True)
        else:
            part = pull(con, cands, cls, SAMPLES_PER_CLASS, args.sample_seed, ts)
        print(f"  known   {cls:28s} {len(part):>7,}")
        known_parts.append(part)
    unk_parts = []
    for cls in unknown_primary + UNKNOWN_RARE:
        part = pull(con, cands, cls, None, args.sample_seed)
        if len(part):
            print(f"  unknown {cls:28s} {len(part):>7,}")
            unk_parts.append(part)
    known = pd.concat(known_parts, ignore_index=True)
    unknown = pd.concat(unk_parts, ignore_index=True)

    labels = known_classes + unknown_primary + UNKNOWN_RARE
    label_to_id = {l: i for i, l in enumerate(labels)}
    for df in (known, unknown):
        df["label_id"] = df["detailed_label"].map(label_to_id)
        df[cands] = df[cands].apply(pd.to_numeric, errors="coerce").fillna(-1).astype(np.float32)

    # 2. 70/15/15: stratified at random, or per class by capture time (earliest 70% train,
    # next 15% val, latest 15% test; timestamps compared only within a class)
    if args.split == "device":
        tv = known[known["_part"] == "trainval"]
        test_known = known[known["_part"] == "test"].reset_index(drop=True)
        train, val = train_test_split(tv, test_size=TEST_FRAC / (1 - TEST_FRAC),
                                      stratify=tv["label_id"], random_state=args.sample_seed)
    elif args.split == "temporal":
        parts = {"train": [], "val": [], "test": []}
        for lid, g in known.groupby("label_id", sort=True):
            g = g.sort_values("ts", kind="mergesort")
            n = len(g); a = int(0.70 * n); b = a + int(0.15 * n)
            parts["train"].append(g.iloc[:a]); parts["val"].append(g.iloc[a:b]); parts["test"].append(g.iloc[b:])
        train, val, test_known = (pd.concat(parts[k], ignore_index=True) for k in ("train", "val", "test"))
    else:
        train, tmp = train_test_split(known, test_size=VAL_FRAC + TEST_FRAC,
                                      stratify=known["label_id"], random_state=args.sample_seed)
        val, test_known = train_test_split(tmp, test_size=0.5,
                                           stratify=tmp["label_id"], random_state=args.sample_seed)
    print(f"split: train {len(train):,}  val {len(val):,}  test_known {len(test_known):,}  "
          f"test_unknown {len(unknown):,}")

    # 3. categoricals from known training rows only
    cat_vocabs = {}
    for col in CAT_FEATURES:
        vals = sorted(train[col].dropna().unique().tolist())
        cat_vocabs[col] = {v: i + 1 for i, v in enumerate(vals)}
        for df in (train, val, test_known, unknown):
            df[col + "_id"] = df[col].map(cat_vocabs[col]).fillna(0).astype(np.float32)
    cat_ids = [c + "_id" for c in CAT_FEATURES]

    # 4. known-only random-forest selection
    rf = RandomForestClassifier(n_estimators=args.n_trees, class_weight="balanced",
                                n_jobs=-1, random_state=args.rf_seed)
    rf.fit(train[cands].values, train["label_id"].values)
    imp = pd.Series(rf.feature_importances_, index=cands).sort_values(ascending=False)
    cum = imp.cumsum() / imp.sum()
    n_keep = int((cum < CUM_IMPORTANCE).sum()) + 1
    rerun_numeric = imp.index[:n_keep].tolist()
    print(f"\nre-run forest keeps {len(rerun_numeric)} numeric features")
    print(f"  vs documented list, extra:   {[f for f in rerun_numeric if f not in PAPER_KFS_NUMERIC]}")
    print(f"  vs documented list, missing: {[f for f in PAPER_KFS_NUMERIC if f not in rerun_numeric]}")
    # robustness splits always use their own re-selection; the documented list is their control
    use_rerun = args.use_rerun_selection or args.split != "random"
    kfs_numeric = rerun_numeric if use_rerun else PAPER_KFS_NUMERIC
    print(f"using {'re-run' if use_rerun else 'documented'} list: "
          f"{len(kfs_numeric)} numeric + {len(cat_ids)} categorical = {len(kfs_numeric) + len(cat_ids)}")
    dropped = [f for f in NUMERIC_FEATURES if f not in kfs_numeric]
    added = [f for f in kfs_numeric if f not in NUMERIC_FEATURES]
    nesy_restore = [f for f in NESY_RULE_FEATURES if f not in kfs_numeric]
    print(f"  NeSy rule features restored: {nesy_restore}")

    # 5. save both datasets over the same rows
    keep_cols = cands + cat_ids + ["label_id", "detailed_label"]
    common = {
        "cat_vocabs": cat_vocabs,
        "label_to_id": label_to_id,
        "id_to_label": {str(i): l for l, i in label_to_id.items()},
        "known_classes": known_classes,
        "unknown_primary": unknown_primary,
        "unknown_rare": UNKNOWN_RARE,
        "known_ids": [label_to_id[l] for l in known_classes],
        "samples_per_class": SAMPLES_PER_CLASS,
        "sample_seed": args.sample_seed,
    }
    selected = {"feature_cols": kfs_numeric + cat_ids,
                "nesy_feature_cols": kfs_numeric + nesy_restore + cat_ids,
                "selection": {"method": "class-weighted RF on known-class training rows",
                              "list_used": "rerun" if use_rerun else "documented",
                               "rerun_numeric": rerun_numeric,
                               "cum_importance": CUM_IMPORTANCE, "n_trees": args.n_trees,
                               "rf_seed": args.rf_seed,
                               "importance": {k: float(v) for k, v in imp.items()},
                               "dropped_vs_submitted": dropped, "added_vs_submitted": added,
                               "nesy_restored": nesy_restore}}
    if args.split == "random":
        datasets = {"ctuk": selected,
                    "ctuko": {"feature_cols": NUMERIC_FEATURES + cat_ids,
                              "nesy_feature_cols": NUMERIC_FEATURES + cat_ids,
                              "selection": {"method": "submitted list (selected on all CTU labels)"}}}
    else:
        # same-sample control: the documented 31-feature known-only list on these rows
        doc_restore = [f for f in NESY_RULE_FEATURES if f not in PAPER_KFS_NUMERIC]
        base = {"alt": "ctualt", "temporal": "ctut", "device": "ctud"}[args.split]
        datasets = {base: selected,
                    base + "_o": {"feature_cols": PAPER_KFS_NUMERIC + cat_ids,
                                  "nesy_feature_cols": PAPER_KFS_NUMERIC + doc_restore + cat_ids,
                                  "selection": {"method": "documented known-only list (main split), "
                                                          "same-sample control"}}}
        keep_cols += list(ts)
        unknown = unknown.assign(**({"ts": np.nan} if ts else {}))
    for name, extra in datasets.items():
        name += args.tag
        out = os.path.join(ROOT, "data", name)
        os.makedirs(out, exist_ok=True)
        for split, df in (("train", train), ("val", val), ("test_known", test_known),
                          ("test_unknown", unknown)):
            df[keep_cols].to_parquet(os.path.join(out, f"{split}.parquet"), index=False)
        vocab = {**common, **extra, "n_features": len(extra["feature_cols"])}
        with open(os.path.join(out, "vocab.json"), "w") as f:
            json.dump(vocab, f, indent=2)
        print(f"saved data/{name}/ ({vocab['n_features']} features)")


if __name__ == "__main__":
    main()
