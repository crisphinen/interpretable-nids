#!/usr/bin/env python3
# Restored-features control split (main paper, Open-Set Detection, CTU paragraph).
# data/ctuk_n holds the same rows as data/ctuk (the parquet files are symlinks), but the MLP/CBM
# input is NeSy-NIDS's own 33 columns: the 31 known-class-only features plus the two rule
# counters (resp_pkts, fin_count) that NeSy-NIDS's expert rules read. Training the MLP and CBMs
# on it tests whether those two counters explain NeSy-NIDS's higher CTU AUROC.
#
# usage: python 08_make_ctuk_n.py            (after 06_make_kfs_splits.py has written data/ctuk)
#        python -m cbm.run_multiseed --dataset ctuk_n --seeds 5 --device cuda \
#            --variants MLPBaseline JointCBM_g0 SequentialCBM HybridCBM
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    v = json.loads((ROOT / "data/ctuk/vocab.json").read_text())
    cols = v["nesy_feature_cols"]
    d = ROOT / "data" / "ctuk_n"
    d.mkdir(exist_ok=True)
    for s in ("train", "val", "test_known", "test_unknown"):
        if not (d / f"{s}.parquet").is_symlink():
            os.symlink(f"../ctuk/{s}.parquet", d / f"{s}.parquet")
    nv = {**v, "feature_cols": cols, "nesy_feature_cols": cols, "n_features": len(cols),
          "variant_note": "ctuk rows, CBM input = NeSy columns (31 + resp_pkts, fin_count)"}
    (d / "vocab.json").write_text(json.dumps(nv, indent=2))
    print(f"data/ctuk_n: {len(cols)} columns, added {sorted(set(cols) - set(v['feature_cols']))}")


if __name__ == "__main__":
    main()
