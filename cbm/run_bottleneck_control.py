#!/usr/bin/env python3
# Compression control: an ordinary neural network with an 8-unit representation. It is the
# JointCBM architecture (encoder d->256->256->64, 8 sigmoid units, linear classifier) trained with
# the classification loss only (no concept labels: lambda = gamma = 0), so its 8 units are learned
# freely rather than tied to the declared concepts. Training, evaluation flows and scoring follow
# cbm/run_multiseed.py (per-class Mahalanobis on the 8 units; np.random.seed(42) before scoring).
# Checkpoints: results/cbm/ms/{ds}_Bottleneck8_s{seed}.pt; rows: results/cbm/bottleneck8_{ds}.json.
#
# usage: python -m cbm.run_bottleneck_control --datasets ctuk cicr ton --device cuda
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from cbm.model import JointCBM
from cbm.train import load_dataset, train_joint, make_tensors, make_loader, compute_val_f1, EMBED_DIM
from cbm.evaluate import load_test_data, ood_auroc, tpr_at_fpr

RES, CK = ROOT / "results" / "cbm", ROOT / "results" / "cbm" / "ms"


def run(ds, device, seeds):
    (Xtr, ytr, Ctr), (Xva, yva, Cva), nf, nk, nc, _ = load_dataset(ds)
    (Xte, yte, _), _, Xun, _, _ = load_test_data(ds)
    a, b, c = make_tensors(Xtr, ytr, Ctr, device); d, e, f = make_tensors(Xva, yva, Cva, device)
    ltr = make_loader(a, b, c, shuffle=True); lva = make_loader(d, e, f, shuffle=False)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    dest = RES / f"bottleneck8_{ds}.json"
    rows = json.loads(dest.read_text()) if dest.exists() else []
    done = {r["seed"] for r in rows}
    for s in seeds:
        if s in done:
            continue
        torch.manual_seed(s); np.random.seed(s)
        m = JointCBM(nf, nc, nk, EMBED_DIM).to(device)
        m = train_joint(m, ltr, lva, device, label=f"Bottleneck8-s{s}", lambda_concept=0.0, gamma_leakage=0.0)
        m.eval()
        torch.save({"model_state_dict": m.state_dict(), "variant": "Bottleneck8", "seed": s,
                    "n_features": nf, "n_classes": nk, "n_concepts": nc}, CK / f"{ds}_Bottleneck8_s{s}.pt")
        with torch.no_grad():
            preds = m(Xte_t)[0].argmax(1).cpu().numpy()
        np.random.seed(42)
        r = {"seed": s, "val_f1": float(compute_val_f1(m, lva, device)),
             "test_f1": float(f1_score(yte, preds, average="weighted", zero_division=0)),
             "auroc": float(ood_auroc(m, Xtr, ytr, Xte, Xun, device)),
             "tpr5": float(tpr_at_fpr(m, Xtr, ytr, Xte, Xun, device))}
        rows.append(r); dest.write_text(json.dumps(rows, indent=2))
        print(f"  [{ds}] s{s}: F1={r['test_f1']:.4f} AUROC={r['auroc']:.4f} TPR={r['tpr5']:.3f}", flush=True)
    g = lambda k: np.array([r[k] for r in rows])
    print(f"[{ds}] Bottleneck8 n={len(rows)} F1 {g('test_f1').mean():.4f} AUROC {g('auroc').mean():.3f}±{g('auroc').std():.3f} "
          f"TPR {g('tpr5').mean():.3f}±{g('tpr5').std():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ctuk", "cicr", "ton"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for ds in args.datasets:
        run(ds, dev, list(range(args.seeds)))


if __name__ == "__main__":
    main()
