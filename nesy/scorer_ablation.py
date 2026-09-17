#!/usr/bin/env python3
# Reconstructed scorer ablation (Table tab:scorer): OOD AUROC of five scoring
# rules on the binary rule-activation space of the five NeSy-NIDS seeds.
# Protocol matches nesy.evaluate: class statistics fitted on training
# activations, AUROC on test_known vs test_unknown. mahal/energy/comb are read
# from the per-seed eval JSONs (run nesy.evaluate first); hamming/jaccard are
# computed here.
#   mahal   per-class Mahalanobis (min over classes)           [Lee et al. 2018]
#   hamming Hamming distance to nearest class majority-vote binary centroid
#   jaccard 1 - Jaccard similarity to nearest class majority-vote centroid
#   energy  logit-space energy score -logsumexp(logits)
#   comb    z-normalised average of mahal and energy (val-calibrated)
# Also recomputes results/nesy/r2_fig_quant.json["fig3"] (within-class
# activation variance vs Mahalanobis AUROC) with the new AUROCs.
#
# usage: python -m nesy.scorer_ablation --device cuda
import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from nesy.evaluate import (load_known_split, load_unknown_test, load_model,
                           K_EVAL, MAX_FIT, MAX_EVAL, RESULTS_DIR)


def _auroc(s_in, s_out):
    y = np.concatenate([np.zeros(len(s_in)), np.ones(len(s_out))])
    return float(roc_auc_score(y, np.concatenate([s_in, s_out])))


def _centroid_dists(emb_tr, y_tr, emb, kind):
    cents = np.stack([(emb_tr[y_tr == c].mean(0) > 0.5).astype(np.float32)
                      for c in np.unique(y_tr)])            # (C, R) majority vote
    e = emb[:, None, :]; c = cents[None, :, :]
    if kind == "hamming":
        d = (e != c).sum(-1)
    else:  # jaccard distance
        inter = ((e == 1) & (c == 1)).sum(-1); union = ((e == 1) | (c == 1)).sum(-1)
        d = 1.0 - inter / np.maximum(union, 1)
    return d.min(1)


def run(ds, device):
    X_tr, y_tr, _ = load_known_split(ds, device, "train")
    X_va, y_va, _ = load_known_split(ds, device, "val")
    X_te, y_te, _ = load_known_split(ds, device, "test_known")
    X_un = load_unknown_test(ds, device)
    out = {k: [] for k in ["mahal", "hamming", "jaccard", "energy", "comb"]}
    wcv = []
    rng = np.random.RandomState(42)
    for s in range(5):
        model, _ = load_model(RESULTS_DIR / f"{ds}_nesy_s{s}.pt", device)
        with torch.no_grad():
            e_tr = model.get_embedding(X_tr, k=K_EVAL).cpu().numpy()
            e_te = model.get_embedding(X_te, k=K_EVAL).cpu().numpy()
            e_un = model.get_embedding(X_un, k=K_EVAL).cpu().numpy()
        ytr = y_tr.cpu().numpy()
        if len(e_tr) > MAX_FIT:
            idx = rng.choice(len(e_tr), MAX_FIT, replace=False); e_tr, ytr = e_tr[idx], ytr[idx]
        if len(e_te) > MAX_EVAL: e_te = e_te[rng.choice(len(e_te), MAX_EVAL, replace=False)]
        if len(e_un) > MAX_EVAL: e_un = e_un[rng.choice(len(e_un), MAX_EVAL, replace=False)]
        for kind in ["hamming", "jaccard"]:
            out[kind].append(_auroc(_centroid_dists(e_tr, ytr, e_te, kind),
                                    _centroid_dists(e_tr, ytr, e_un, kind)))
        # mahal / energy / comb: same protocol as here, taken from the per-seed
        # eval JSON written by nesy.evaluate so Table tab:main and tab:scorer agree.
        ev = json.loads((RESULTS_DIR / f"{ds}_nesy_s{s}_eval.json").read_text())
        out["mahal"].append(ev["ood_auroc_mahal"]); out["energy"].append(ev["ood_auroc_energy"])
        out["comb"].append(ev["ood_auroc_comb_avg"])
        # within-class activation variance (val split, as for the selectivity figure)
        with torch.no_grad():
            e_va = model.get_embedding(X_va, k=K_EVAL).cpu().numpy()
        yva = y_va.cpu().numpy()
        wcv.append(float(np.mean([e_va[yva == c].var(0).mean() for c in np.unique(yva)])))
        print(f"{ds} s{s}: " + "  ".join(f"{k}={out[k][-1]:.4f}" for k in out) + f"  wcv={wcv[-1]:.4f}")
    res = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "all": [float(x) for x in v]}
           for k, v in out.items()}
    return res, wcv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    res, pts = {}, []
    for ds in ["ctu", "cic"]:
        res[ds], wcv = run(ds, device)
        pts += [{"ds": ds, "seed": s, "wcv": wcv[s], "auroc": res[ds]["mahal"]["all"][s]} for s in range(5)]
        print(f"{ds}: " + "  ".join(f"{k}={v['mean']:.3f}±{v['std']:.3f}" for k, v in res[ds].items()))
    (RESULTS_DIR / "scorer_ablation.json").write_text(json.dumps(res, indent=2))
    fq_path = RESULTS_DIR / "r2_fig_quant.json"
    fq = json.loads(fq_path.read_text()) if fq_path.exists() else {}
    r, p = pearsonr([q["wcv"] for q in pts], [q["auroc"] for q in pts])
    fq["fig3"] = {"pearson_r": float(r), "p": float(p), "points": pts}
    fq_path.write_text(json.dumps(fq, indent=2))
    print(f"fig3: r={r:.4f} p={p:.2e}"); print("wrote scorer_ablation.json, r2_fig_quant.json")


if __name__ == "__main__":
    main()
