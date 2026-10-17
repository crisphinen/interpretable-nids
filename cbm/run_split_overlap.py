#!/usr/bin/env python3
# Train/test overlap of the known classes per split (main paper, Device-disjoint splits;
# supplement, device sections). For each known class: the number of distinct test feature vectors,
# and the share of test flows whose feature vector (model inputs, rounded to four decimals) occurs
# exactly among the training flows of the same class. Writes results/cbm/split_overlap.json.
#
# usage: python -m cbm.run_split_overlap
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cbm.evaluate import load_test_data
from dsets import load_vocab

DATASETS = ["ctuk", "ctud_h198", "ctud_h111", "ctud_h108", "ton", "tond_a30", "tond_a31"]


def main():
    out = {}
    for ds in DATASETS:
        (Xte, yte, _), (Xtr, ytr, _), _, _, _ = load_test_data(ds)
        names = load_vocab(ds)["known_classes"]
        per = {}
        for c, name in enumerate(names):
            tr = {r.tobytes() for r in Xtr[ytr == c].round(4)}
            te = Xte[yte == c].round(4)
            per[name] = {"n_test": int(len(te)),
                         "distinct_test_vectors": int(len(np.unique(te, axis=0))),
                         "share_with_same_class_train_duplicate": float(np.mean([r.tobytes() in tr for r in te]))}
        mean = float(np.mean([v["share_with_same_class_train_duplicate"] for v in per.values()]))
        out[ds] = {"per_class": per, "mean_share": mean}
        print(f"{ds:10s} mean share {mean:.2f}  " + "  ".join(
            f"{k} {v['share_with_same_class_train_duplicate']:.2f} ({v['distinct_test_vectors']}/{v['n_test']})"
            for k, v in per.items()), flush=True)
    (ROOT / "results/cbm/split_overlap.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
