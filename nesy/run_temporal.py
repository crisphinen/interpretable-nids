#!/usr/bin/env python3
# Temporal (session-respecting) split for NeSy-NIDS on CTU-IoT-23.
#
# Closes the last gap in the temporal evaluation (Table tab:temporal): the CBM
# variants and the MLP were reported on the timestamp-ordered split, but
# NeSy-NIDS was not. To keep the comparison honest we reuse the *identical*
# split built by cbm.run_temporal.build_temporal_split (mode='sample': the same
# per-class random sample as the main run, re-ordered by capture timestamp,
# earliest 70% train / middle 15% val / latest 15% test; unknown families
# unchanged). Only the model and its OOD scorer differ.
#
# NeSy is trained with the standard k-annealed schedule (nesy.train.train_nesy)
# and evaluated exactly as nesy.evaluate does on the main split:
#   - known-class weighted F1 on the temporal test (all classes),
#   - per-class F1 + weighted F1 over the three temporally-coherent families
#     (Benign, C&C-HeartBeat, Okiru; DDoS excluded — it is temporally bimodal,
#     a documented dataset property, see the paper),
#   - per-class Mahalanobis OOD AUROC on the binary rule activations, fitted on
#     the temporal-train activations, scored test_known vs test_unknown.
#
# usage: python -m nesy.run_temporal --seeds 3 --device cuda
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cbm.run_temporal import build_temporal_split
from nesy.model import NeSyNIDS, CTU_RULES
from nesy.train import train_nesy, make_loader
from nesy.evaluate import compute_ood_auroc_perclass, K_EVAL

RESULTS_DIR = PROJECT_ROOT / "results" / "nesy"

# NeSy analogue of the CBM gamma sweep: the main model (lambda_alpha=0, the
# main-table NeSy row) and the alpha-regularised model (lambda_alpha=0.1, the
# "NeSy + alpha-reg" row).
VARIANTS = [("NeSyNIDS", 0.0), ("NeSyNIDS_a0.1", 0.1)]


def train_variant(lambda_alpha, d, device):
    model = NeSyNIDS(d["n_features"], d["n_classes"], CTU_RULES).to(device)
    loader_tr = make_loader(d["Xtr"], d["ytr"], device, shuffle=True)
    loader_va = make_loader(d["Xva"], d["yva"], device, shuffle=False)
    model, _ = train_nesy(model, loader_tr, loader_va, device, label="nesy_temporal",
                          lambda_alpha=lambda_alpha)

    # temporal test-known: all-class weighted F1, per-class F1, coherent-3 F1
    model.eval()
    Xte_t = torch.tensor(d["Xte"], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, _ = model(Xte_t, k=K_EVAL)
        pred = logits.argmax(dim=1).cpu().numpy()
    yte = d["yte"]
    labels = list(range(d["n_classes"]))
    f1_all = f1_score(yte, pred, labels=labels, average="weighted", zero_division=0)
    f1_pc = f1_score(yte, pred, labels=labels, average=None, zero_division=0)
    coherent = [c for c in labels if c != d["ddos_idx"]]
    f1_coh = f1_score(yte, pred, labels=coherent, average="weighted", zero_division=0)

    # per-class Mahalanobis OOD AUROC on rule activations (fit on temporal train)
    Xtr_t = torch.tensor(d["Xtr"], dtype=torch.float32, device=device)
    ytr_t = torch.tensor(d["ytr"], dtype=torch.long, device=device)
    Xun_t = torch.tensor(d["Xun"], dtype=torch.float32, device=device)
    au = compute_ood_auroc_perclass(model, Xtr_t, ytr_t, Xte_t, Xun_t)
    return float(f1_all), float(f1_coh), [float(x) for x in f1_pc], float(au)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["strict", "mild", "sample"], default="sample")
    ap.add_argument("--out", default=None, help="override output JSON path")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"[ctu/nesy] building temporal split [mode={args.mode}] (identical to CBM temporal) ...")
    d = build_temporal_split(mode=args.mode)

    seeds = list(range(args.seeds))
    out = {"known": d["class_names"], "mode": args.mode, "seeds": seeds,
           "class_names": d["class_names"], "ddos_idx": d["ddos_idx"],
           "note": "NeSy-NIDS on the same temporal split as tab:temporal. "
                   "f1_* = all-class weighted; f1_coherent_* = weighted over "
                   "temporally-coherent families (DDoS excluded); f1_perclass_mean "
                   "aligns with class_names. OOD = per-class Mahalanobis on rule "
                   "activations, fit on temporal train.", "results": {}}
    for vname, la in VARIANTS:
        f1s, cohs, pcs, aus = [], [], [], []
        for s in seeds:
            torch.manual_seed(s); np.random.seed(s)
            f1, coh, pc, au = train_variant(la, d, device)
            f1s.append(f1); cohs.append(coh); pcs.append(pc); aus.append(au)
            print(f"  {vname:16} seed {s}: F1={f1:.4f} coh3={coh:.4f} "
                  f"AUROC={au:.4f}  per-class={[round(x,3) for x in pc]}")
        out["results"][vname] = {
            "lambda_alpha": la,
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)), "f1_all": f1s,
            "f1_coherent_mean": float(np.mean(cohs)), "f1_coherent_std": float(np.std(cohs)),
            "f1_coherent_all": cohs,
            "f1_perclass_mean": [float(x) for x in np.mean(pcs, axis=0)],
            "auroc_mean": float(np.mean(aus)), "auroc_std": float(np.std(aus)),
            "auroc_all": aus,
        }
    dest = Path(args.out) if args.out else RESULTS_DIR / "temporal_ctu.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"[ctu/nesy] wrote {dest}")


if __name__ == "__main__":
    main()
