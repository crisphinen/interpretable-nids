# Does a KNOWN-class separability metric govern OOD AUROC without the reversal that
# sinks within-class variance? For every per-seed CBM checkpoint (results/cbm/ms/)
# compute, in the scoring space, from KNOWN classes only (no unknowns, so non-circular):
#   within_var  = mean within-class variance (the confounded metric)
#   fisher_tr   = tr(S_B)/tr(S_W)                 (between/within scatter, trace ratio)
#   fisher_lda  = tr((S_W+eps I)^{-1} S_B)        (ridge LDA separability)
# plus a diagnostic that DOES use unknowns: unk/known mean Mahalanobis distance ratio.
# Pair with validated per-seed AUROC and report the sign of each metric's correlation
# with AUROC by slice (within CTU, within CIC, pooled, between-dataset means).

import sys, json
from pathlib import Path
import numpy as np
import torch
from numpy.linalg import pinv
from statistics import mean

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from cbm.evaluate import load_test_data, rebuild_model, mahalanobis_scores

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["JointCBM_g0", "JointCBM_g0.1", "JointCBM_g0.5", "JointCBM_g1.0",
            "SequentialCBM", "HybridCBM"]  # comparable 8-dim concept space
RNG = np.random.default_rng(0)
base = lambda v: "JointCBM" if v.startswith("JointCBM") else v


def scatters(emb, y):
    classes = np.unique(y)
    mu = emb.mean(0)
    Sw = np.zeros((emb.shape[1],) * 2); Sb = np.zeros_like(Sw)
    for c in classes:
        E = emb[y == c]; muc = E.mean(0)
        Sw += (E - muc).T @ (E - muc)
        Sb += len(E) * np.outer(muc - mu, muc - mu)
    Sw /= len(emb); Sb /= len(emb)
    return Sw, Sb


def pearson(xs, ys):
    mx, my = mean(xs), mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def main():
    rows = []
    for ds in ["ctu", "cic"]:
        (Xk, yk, _), (Xtr, ytr, _), Xun, _, _ = load_test_data(ds)
        if len(Xtr) > 20000:
            i = RNG.choice(len(Xtr), 20000, replace=False); Xtr, ytr = Xtr[i], ytr[i]
        seedj = {(r["variant"], r["seed"]): r for r in
                 json.loads((ROOT / "results" / "cbm" / f"multiseed_{ds}_v2_seeds.json").read_text())}
        for v in VARIANTS:
            for s in SEEDS:
                ck = torch.load(ROOT / "results" / "cbm" / "ms" / f"{ds}_{v}_s{s}.pt",
                                map_location=DEVICE, weights_only=False)
                m = rebuild_model(base(v), ck["n_features"], ck["n_classes"], ck["n_concepts"]).to(DEVICE)
                m.load_state_dict(ck["model_state_dict"]); m.eval()
                E = lambda X: m.get_embedding(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()
                with torch.no_grad():
                    etr, ekn, eun = E(Xtr), E(Xk[:5000]), E(Xun[:5000])
                Sw, Sb = scatters(etr, ytr)
                trSw = np.trace(Sw); d = Sw.shape[0]
                fisher_tr = float(np.trace(Sb) / trSw) if trSw > 0 else 0.0
                fisher_lda = float(np.trace(pinv(Sw + 1e-4 * np.eye(d)) @ Sb))
                sk = mahalanobis_scores(etr, ekn, train_labels=ytr)
                su = mahalanobis_scores(etr, eun, train_labels=ytr)
                unk_known_ratio = float(np.mean(su) / max(np.mean(sk), 1e-9))
                rows.append(dict(dataset=ds, variant=v, seed=s,
                                 within_var=float(np.mean(np.diag(Sw))),
                                 fisher_tr=fisher_tr, fisher_lda=fisher_lda,
                                 unk_known_ratio=unk_known_ratio,
                                 auroc=seedj.get((v, s), {}).get("auroc"),
                                 f1=seedj.get((v, s), {}).get("test_f1")))
        print(f"  {ds} done ({sum(1 for r in rows if r['dataset']==ds)} models)", flush=True)
    (ROOT / "results" / "cbm" / "separability_test.json").write_text(json.dumps(rows, indent=2))

    rows = [r for r in rows if r["auroc"] is not None]
    print("\n=== sign of r(metric, AUROC) by slice (does it reverse?) ===")
    print(f"{'metric':>16} | {'within CTU':>11} {'within CIC':>11} {'pooled':>8} | reversal?")
    for metric in ["within_var", "fisher_tr", "fisher_lda", "unk_known_ratio", "f1"]:
        rc = pearson([r[metric] for r in rows if r["dataset"] == "ctu"], [r["auroc"] for r in rows if r["dataset"] == "ctu"])
        ri = pearson([r[metric] for r in rows if r["dataset"] == "cic"], [r["auroc"] for r in rows if r["dataset"] == "cic"])
        rp = pearson([r[metric] for r in rows], [r["auroc"] for r in rows])
        rev = "YES (flips)" if rc * ri < 0 else "no (consistent)"
        print(f"{metric:>16} | {rc:>+11.2f} {ri:>+11.2f} {rp:>+8.2f} | {rev}")

    print("\n=== per-variant means (fisher_tr, fisher_lda, within_var, AUROC) ===")
    for ds in ["ctu", "cic"]:
        for v in VARIANTS:
            rs = [r for r in rows if r["dataset"] == ds and r["variant"] == v]
            if not rs: continue
            print(f"  {ds} {v:14s} fisher_tr={mean(r['fisher_tr'] for r in rs):.3f} "
                  f"fisher_lda={mean(r['fisher_lda'] for r in rs):.2f} "
                  f"wvar={mean(r['within_var'] for r in rs):.4f} AUROC={mean(r['auroc'] for r in rs):.3f}")


if __name__ == "__main__":
    main()
