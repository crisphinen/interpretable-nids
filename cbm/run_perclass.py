#!/usr/bin/env python3
# Per-class known-test F1 (supplement tab:perclass): MLP and JointCBM (gamma=0), seeds 0-4, from the
# multiseed checkpoints. Writes results/cbm/perclass_{ds}.json.
#
# usage: python -m cbm.run_perclass --datasets ctuk cicr --device cpu
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from cbm.evaluate import load_test_data, rebuild_model
from dsets import load_vocab, is_ctu

RES = Path(__file__).parent.parent / "results" / "cbm"


def class_names(ds):
    v = load_vocab(ds)
    ids = v["known_ids"] if is_ctu(ds) else [v["label_to_id"][c] for c in v["known_classes"]]
    i2l = {int(k): l for k, l in v["id_to_label"].items()} if "id_to_label" in v else {i: l for l, i in v["label_to_id"].items()}
    return [i2l[i] for i in sorted(ids)]


@torch.no_grad()
def run(ds, device):
    (Xte, yte, _), *_ = load_test_data(ds)
    X = torch.tensor(Xte, dtype=torch.float32, device=device)
    names, out = class_names(ds), {}
    for v in ("MLPBaseline", "JointCBM_g0"):
        per = []
        for s in range(5):
            ck = torch.load(RES / "ms" / f"{ds}_{v}_s{s}.pt", map_location=device, weights_only=False)
            m = rebuild_model("JointCBM" if v.startswith("JointCBM") else v, ck["n_features"], ck["n_classes"],
                              ck["n_concepts"]).to(device)
            m.load_state_dict(ck["model_state_dict"]); m.eval()
            p = m(X)[0].argmax(1).cpu().numpy()
            per.append(f1_score(yte, p, labels=list(range(len(names))), average=None, zero_division=0))
        out[v] = dict(zip(names, np.mean(per, 0).tolist()))
    for c in names:
        print(f"[{ds}] {c:22s} MLP {out['MLPBaseline'][c]:.3f}  JointCBM {out['JointCBM_g0'][c]:.3f}")
    (RES / f"perclass_{ds}.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ctuk", "cicr"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    for ds in args.datasets:
        run(ds, torch.device(args.device))


if __name__ == "__main__":
    main()
