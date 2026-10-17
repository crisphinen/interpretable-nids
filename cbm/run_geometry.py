#!/usr/bin/env python3
# Known-class geometry vs OOD AUROC (main tab:signflip, fig:confounding; supplement tab:loco; the
# NeSy ten-model correlation and the k_T sweep).
#
# CBM part: the 30 concept-space models per corpus (JointCBM gamma 0/0.1/0.5/1.0, SequentialCBM,
# HybridCBM x seeds 0-4) on ctuk (CTU, known-only features), cicr (CIC) and ton (TON_IoT). From the
# concept activations of <=20,000 known training flows (and the class labels only):
#   within_var   mean diagonal of the pooled within-class covariance S_W
#   fisher       tr(S_B) / tr(S_W)
#   ridge_lda    tr((S_W + 1e-4 I)^-1 S_B)
#   silhouette   Euclidean silhouette score on a 5,000-flow subsample
#   nc_margin    mean (distance to nearest other-class centroid - distance to own centroid)
#   entropy      mean binary entropy of the soft concept activations
#   f1           closed-set weighted test F1 (multiseed rows)
#   unk_ratio    mean per-class Mahalanobis score of 5,000 unknown / 5,000 known test flows
#                (uses unknown flows: a proxy of AUROC, not a predictor)
# Correlations with the per-seed Mahalanobis AUROC: Pearson r over the 30 models (tab:signflip),
# over the six configuration means (r6), the range over leave-one-configuration-out (LOCO), and
# whether dropping SequentialCBM changes the sign (tab:loco).
# NeSy part: within-class variance of the hard rule activations on the validation split (as
# nesy/scorer_ablation.py) vs Mahalanobis AUROC, for the ten main models (ctuk + cicr) and for the
# k_T sweep {2, 5, 10, 20} x 5 seeds per corpus (nesy/run_kt_sweep.sh).
# Writes results/cbm/geometry.json and results/figures/confounding.{pdf,png}.
#
# usage: python -m cbm.run_geometry --device cuda [--skip-cbm] [--skip-nesy]
import argparse, json
from pathlib import Path

import numpy as np
import torch
from numpy.linalg import pinv
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "cbm"
CORPORA = {"ctuk": "CTU", "cicr": "CIC", "ton": "TON"}
CONFIGS = ["JointCBM_g0", "JointCBM_g0.1", "JointCBM_g0.5", "JointCBM_g1.0", "SequentialCBM", "HybridCBM"]
STATS = ["within_var", "fisher", "ridge_lda", "silhouette", "nc_margin", "entropy", "f1", "unk_ratio"]


def known_stats(E, y, rng):
    classes = np.unique(y); mu = E.mean(0)
    Sw = np.zeros((E.shape[1],) * 2); Sb = np.zeros_like(Sw); cents = {}
    for c in classes:
        Ec = E[y == c]; mc = Ec.mean(0); cents[c] = mc
        Sw += (Ec - mc).T @ (Ec - mc); Sb += len(Ec) * np.outer(mc - mu, mc - mu)
    Sw /= len(E); Sb /= len(E)
    C = np.stack([cents[c] for c in classes]); D = np.linalg.norm(E[:, None, :] - C[None], axis=2)
    own = D[np.arange(len(E)), np.searchsorted(classes, y)]
    D_other = D.copy(); D_other[np.arange(len(E)), np.searchsorted(classes, y)] = np.inf
    i = rng.choice(len(E), min(5000, len(E)), replace=False)
    p = np.clip(E, 1e-7, 1 - 1e-7)
    return {"within_var": float(np.mean(np.diag(Sw))),
            "fisher": float(np.trace(Sb) / np.trace(Sw)) if np.trace(Sw) > 0 else 0.0,
            "ridge_lda": float(np.trace(pinv(Sw + 1e-4 * np.eye(Sw.shape[0])) @ Sb)),
            "silhouette": float(silhouette_score(E[i], y[i])) if len(np.unique(y[i])) > 1 else 0.0,
            "nc_margin": float(np.mean(D_other.min(1) - own)),
            "entropy": float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))}


def r(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")


@torch.no_grad()
def cbm_part(device):
    from cbm.evaluate import load_test_data, rebuild_model, mahalanobis_scores
    out = {}
    for ds in CORPORA:
        rng = np.random.default_rng(0)
        (Xk, _, _), (Xtr, ytr, _), Xun, _, _ = load_test_data(ds)
        if len(Xtr) > 20000:
            i = rng.choice(len(Xtr), 20000, replace=False); Xtr, ytr = Xtr[i], ytr[i]
        Xk = Xk[rng.choice(len(Xk), min(5000, len(Xk)), replace=False)]
        Xun = Xun[rng.choice(len(Xun), min(5000, len(Xun)), replace=False)]
        seeds = {(x["variant"], x["seed"]): x for x in json.loads((RES / f"multiseed_{ds}_v2_seeds.json").read_text())}
        rows = []
        for v in CONFIGS:
            for s in range(5):
                ck = torch.load(RES / "ms" / f"{ds}_{v}_s{s}.pt", map_location=device, weights_only=False)
                m = rebuild_model("JointCBM" if v.startswith("JointCBM") else v,
                                  ck["n_features"], ck["n_classes"], ck["n_concepts"]).to(device)
                m.load_state_dict(ck["model_state_dict"]); m.eval()
                E = lambda X: m.get_embedding(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
                etr = E(Xtr)
                st = known_stats(etr, ytr, rng)
                sk = mahalanobis_scores(etr, E(Xk), train_labels=ytr)
                su = mahalanobis_scores(etr, E(Xun), train_labels=ytr)
                st["unk_ratio"] = float(su.mean() / max(sk.mean(), 1e-9))
                st["f1"] = seeds[(v, s)]["test_f1"]
                rows.append({"config": v, "seed": s, "auroc": seeds[(v, s)]["auroc"], **st})
        corr = {}
        for k in STATS:
            cm = [(np.mean([x[k] for x in rows if x["config"] == c]), np.mean([x["auroc"] for x in rows if x["config"] == c]))
                  for c in CONFIGS]
            loco = [r([x[k] for x in rows if x["config"] != c], [x["auroc"] for x in rows if x["config"] != c]) for c in CONFIGS]
            no_seq = r([x[k] for x in rows if x["config"] != "SequentialCBM"],
                       [x["auroc"] for x in rows if x["config"] != "SequentialCBM"])
            full = r([x[k] for x in rows], [x["auroc"] for x in rows])
            corr[k] = {"r30": full, "r6": r(*zip(*cm)), "loco": [min(loco), max(loco)], "r_no_seq": no_seq,
                       "sign_flips_without_seq": bool(np.sign(no_seq) != np.sign(full))}
        out[ds] = {"rows": rows, "corr": corr,
                   "mean_within_var": float(np.mean([x["within_var"] for x in rows])),
                   "mean_auroc": float(np.mean([x["auroc"] for x in rows]))}
        print(f"[{CORPORA[ds]}] " + "  ".join(f"{k} {corr[k]['r30']:+.2f}" for k in STATS))
    dm = [(out[d]["mean_within_var"], out[d]["mean_auroc"]) for d in CORPORA]
    out["dataset_means_r"] = r(*zip(*dm))
    print(f"dataset means (within_var vs AUROC): r = {out['dataset_means_r']:+.2f}")
    return out


@torch.no_grad()
def nesy_part(device):
    from nesy.evaluate import load_known_split, load_model, K_EVAL, RESULTS_DIR
    def wcv(ckpt, Xva, yva):
        m, _ = load_model(ckpt, device)
        e = m.get_embedding(Xva, k=K_EVAL).cpu().numpy()
        return float(np.mean([e[yva == c].var(0).mean() for c in np.unique(yva)]))
    out = {"ten_model": [], "kt": {}}
    for ds in ("ctuk", "cicr"):
        Xva, yva, _ = load_known_split(ds, device, "val"); yva = yva.cpu().numpy()
        pts = []
        for k in (2, 5, 10, 20):
            for s in range(5):
                tag = "" if k == 10 else f"_k{k}"
                ck, ev = RESULTS_DIR / f"{ds}_nesy_s{s}{tag}.pt", RESULTS_DIR / f"{ds}_nesy_s{s}{tag}_eval.json"
                if ck.exists() and ev.exists():
                    e = json.loads(ev.read_text())
                    pts.append({"k_T": k, "seed": s, "wcv": wcv(ck, Xva, yva), "auroc": e["ood_auroc_mahal"],
                                "f1": e["test_f1"]})
        per_k = {k: {m: float(np.mean([p[m] for p in pts if p["k_T"] == k])) for m in ("wcv", "auroc", "f1")}
                 for k in (2, 5, 10, 20) if any(p["k_T"] == k for p in pts)}
        out["kt"][ds] = {"points": pts, "per_k": per_k,
                         "r": r([p["wcv"] for p in pts], [p["auroc"] for p in pts]), "n": len(pts)}
        out["ten_model"] += [{"ds": ds, **p} for p in pts if p["k_T"] == 10]
        print(f"[NeSy {ds}] k_T sweep n={len(pts)} r(wcv, AUROC) = {out['kt'][ds]['r']:+.2f}  " +
              "  ".join(f"k{k}: wcv {v['wcv']:.3f} AUROC {v['auroc']:.3f} F1 {v['f1']:.3f}" for k, v in per_k.items()))
    tm = out["ten_model"]
    out["ten_model_r"] = r([p["wcv"] for p in tm], [p["auroc"] for p in tm])
    print(f"[NeSy] ten-model r(wcv, AUROC) = {out['ten_model_r']:+.3f} (n={len(tm)})")
    return out


def confounding_figure(cbm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"ctuk": "#1f77b4", "cicr": "#d62728", "ton": "#2ca02c"}
    markers = dict(zip(CONFIGS, "osD^vP"))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ds, lab in CORPORA.items():
        rows = cbm[ds]["rows"]
        x = np.array([q["within_var"] for q in rows]); y = np.array([q["auroc"] for q in rows])
        ax.scatter(x, y, s=10, color=colors[ds], alpha=0.25)
        for c in CONFIGS:
            xs = [q["within_var"] for q in rows if q["config"] == c]; ys = [q["auroc"] for q in rows if q["config"] == c]
            ax.errorbar(np.mean(xs), np.mean(ys), xerr=np.std(xs), yerr=np.std(ys), fmt=markers[c],
                        color=colors[ds], ms=6, capsize=2)
        b = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 10)
        ax.plot(xx, np.polyval(b, xx), "--", color=colors[ds], lw=1,
                label=f"{lab} (r = {cbm[ds]['corr']['within_var']['r30']:+.2f})")
        ax.scatter(cbm[ds]["mean_within_var"], cbm[ds]["mean_auroc"], marker="*", s=220, color=colors[ds],
                   edgecolor="k", zorder=5)
    ax.set_xlabel("Within-class concept-activation variance"); ax.set_ylabel("OOD AUROC (per-class Mahalanobis)")
    ax.legend(fontsize=8, title="corpus (30 models each)", title_fontsize=8)
    out = ROOT / "results" / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "confounding.pdf", bbox_inches="tight"); fig.savefig(out / "confounding.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-cbm", action="store_true")
    ap.add_argument("--skip-nesy", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    path = RES / "geometry.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    if not args.skip_cbm:
        out["cbm"] = cbm_part(device); confounding_figure(out["cbm"])
    if not args.skip_nesy:
        out["nesy"] = nesy_part(device)
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
