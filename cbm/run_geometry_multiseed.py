# Clean, multi-seed geometry x architecture x gamma factorial for the confounding
# result. Uses the SAME per-seed checkpoints (results/cbm/ms/) behind the validated
# multiseed_v2 table, so geometry and AUROC are self-consistent. For each
# (variant, gamma, seed) it computes a scorer-independent geometry metric of the
# scoring space -- mean within-class variance and the degeneracy of the pooled
# within-class covariance (effective rank, log-det) -- and pairs it with that run's
# already-validated Mahalanobis AUROC from multiseed_{ds}_v2_seeds.json.
#
# The point: within-class variance predicts AUROC with OPPOSITE sign between datasets
# vs within an architecture family -- a textbook confound, not a mechanism.
# Writes results/cbm/geometry_multiseed.json.

import sys, json
from pathlib import Path
import numpy as np
import torch
from numpy.linalg import slogdet
from statistics import mean, pstdev

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from cbm.evaluate import load_test_data, rebuild_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["MLPBaseline", "JointCBM_g0", "JointCBM_g0.1", "JointCBM_g0.5",
            "JointCBM_g1.0", "SequentialCBM", "HybridCBM"]
RNG = np.random.default_rng(0)


def geometry(emb, y):
    res = np.concatenate([emb[y == c] - emb[y == c].mean(0) for c in np.unique(y)], 0)
    Sw = np.cov(res.T)
    Sw = Sw.reshape(1, 1) if Sw.ndim == 0 else Sw
    d = Sw.shape[0]; tr = np.trace(Sw); tr2 = np.trace(Sw @ Sw)
    sign, ld = slogdet(Sw + 1e-6 * np.eye(d))
    return dict(within_var=float(np.mean(np.diag(Sw))),
                eff_rank=float(tr * tr / tr2) if tr2 > 0 else 1.0,
                logdet=float(ld) if sign > 0 else float("-inf"), dim=int(d))


def base(v):
    return "JointCBM" if v.startswith("JointCBM") else v


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
            idx = RNG.choice(len(Xtr), 20000, replace=False); Xtr_s, ytr_s = Xtr[idx], ytr[idx]
        else:
            Xtr_s, ytr_s = Xtr, ytr
        seedj = {(r["variant"], r["seed"]): r for r in
                 json.loads((ROOT / "results" / "cbm" / f"multiseed_{ds}_v2_seeds.json").read_text())}
        for v in VARIANTS:
            for s in SEEDS:
                ck = torch.load(ROOT / "results" / "cbm" / "ms" / f"{ds}_{v}_s{s}.pt",
                                map_location=DEVICE, weights_only=False)
                m = rebuild_model(base(v), ck["n_features"], ck["n_classes"], ck["n_concepts"]).to(DEVICE)
                m.load_state_dict(ck["model_state_dict"]); m.eval()
                with torch.no_grad():
                    emb = m.get_embedding(torch.tensor(Xtr_s, dtype=torch.float32, device=DEVICE)).cpu().numpy()
                g = geometry(emb, ytr_s)
                auroc = seedj.get((v, s), {}).get("auroc")
                rows.append(dict(dataset=ds, variant=v, arch=("MLP" if v == "MLPBaseline" else "CBM"),
                                 gamma=ck.get("gamma"), interp=(v != "MLPBaseline"), seed=s,
                                 auroc=auroc, **g))
        print(f"  {ds}: {sum(1 for r in rows if r['dataset']==ds)} models", flush=True)
    (ROOT / "results" / "cbm" / "geometry_multiseed.json").write_text(json.dumps(rows, indent=2))

    # ---- confounding analysis (CBM concept models: comparable 8-dim [0,1] space) ----
    cbm = [r for r in rows if r["interp"] and r["auroc"] is not None]
    print("\n=== per-variant means (CBM concept space) ===")
    print(f"{'dataset':>7} {'variant':>14} {'within_var':>11} {'eff_rank':>9} {'AUROC':>7}")
    cellmeans = {}
    for ds in ["ctu", "cic"]:
        for v in VARIANTS[1:]:
            rs = [r for r in cbm if r["dataset"] == ds and r["variant"] == v]
            if not rs: continue
            wv, er, au = mean(r["within_var"] for r in rs), mean(r["eff_rank"] for r in rs), mean(r["auroc"] for r in rs)
            cellmeans[(ds, v)] = (wv, au)
            print(f"{ds:>7} {v:>14} {wv:>11.4f} {er:>9.2f} {au:>7.3f}")
    print("\n=== confounding: sign of r(within_var, AUROC) by slice ===")
    for ds in ["ctu", "cic"]:
        rs = [r for r in cbm if r["dataset"] == ds]
        print(f"  within {ds.upper()} (across architectures, {len(rs)} runs): r = {pearson([r['within_var'] for r in rs], [r['auroc'] for r in rs]):+.2f}")
    print(f"  pooled across both datasets ({len(cbm)} runs): r = {pearson([r['within_var'] for r in cbm], [r['auroc'] for r in cbm]):+.2f}")
    dsm = {}
    for ds in ["ctu", "cic"]:
        rs = [r for r in cbm if r["dataset"] == ds]
        dsm[ds] = (mean(r["within_var"] for r in rs), mean(r["auroc"] for r in rs))
    print(f"  between datasets (2 dataset means): CTU {dsm['ctu']}  CIC {dsm['cic']}  -> higher-var dataset has higher AUROC: {dsm['ctu'][0]>dsm['cic'][0] and dsm['ctu'][1]>dsm['cic'][1]}")


if __name__ == "__main__":
    main()
