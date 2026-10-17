"""Print every number the manuscript and supplement tables take from results/.

Values are printed table by table as mean and seed std. CTU = ctuk (known-class-only features),
CTU ablation = ctuko (submitted feature list, same rows), CIC = cicr, TON = ton.

usage: python checks/paper_values.py
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CBM, NESY, DATA = ROOT / "results" / "cbm", ROOT / "results" / "nesy", ROOT / "data"
LOAD = lambda p: json.loads(Path(p).read_text())
GAMMAS = ["JointCBM_g0", "JointCBM_g0.1", "JointCBM_g0.5", "JointCBM_g1.0"]
COH3 = [0, 1, 3]  # Benign, C&C-HeartBeat, Okiru (the temporally coherent families; DDoS excluded)


def with_baselines(ds):
    """(F1, AUROC, AUROC std, TPR, TPR std, n) per model of Table III on dataset ds."""
    out = {}
    agg = CBM / f"multiseed_{ds}_v2.json"
    if agg.exists():
        for m, v in LOAD(agg)["results"].items():
            out[m] = (v["test_f1_mean"], v["auroc_mean"], v["auroc_std"],
                      v["tpr5_mean"], v["tpr5_std"], len(v["seeds"]))
    for key, model, tag in (("NeSy", "nesy", ""), ("NeSy_a0.1", "nesy", "_a0.1"),
                            ("RuleOnly", "rule_only", "")):
        rows = [LOAD(f) for f in (NESY / f"{ds}_{model}_s{s}{tag}_eval.json" for s in range(5)) if f.exists()]
        if rows:
            g = lambda k: np.array([r[k] for r in rows])
            out[key] = (g("test_f1").mean(), g("ood_auroc_mahal").mean(), g("ood_auroc_mahal").std(),
                        g("tpr_at_5fpr_mahal").mean(), g("tpr_at_5fpr_mahal").std(), len(rows))
    f = CBM / f"baselines_{ds}.json"
    if f.exists():
        d = LOAD(f)
        for m, v in d["summary"].items():
            out[m] = (v["test_f1"][0], v["auroc"][0], v["auroc"][1], v["tpr5"][0], v["tpr5"][1], len(d["rows"][m]))
    return out


def split_rows(ds):
    """model -> per-seed dicts with f1, per (per-class F1 list) and auroc, for a robustness split."""
    out = {}
    f = CBM / f"multiseed_{ds}_v2_seeds.json"
    if f.exists():
        for r in LOAD(f):
            out.setdefault(r["variant"], []).append(
                {"f1": r["test_f1"], "per": r.get("test_f1_per_class"), "auroc": r["auroc"]})
    for key, tag in (("NeSy", ""), ("NeSy_a0.1", "_a0.1")):
        for s in range(5):
            p = NESY / f"{ds}_nesy_s{s}{tag}_eval.json"
            if p.exists():
                d = LOAD(p)
                out.setdefault(key, []).append({"f1": d["test_f1"], "per": d.get("test_f1_per_class"),
                                                "auroc": d["ood_auroc_mahal"]})
    return out


MAIN_ROWS = ["MLPBaseline", "JointCBM_g0", "JointCBM_g0.1", "JointCBM_g0.5", "JointCBM_g1.0", "SequentialCBM",
             "HybridCBM", "PostHocCBM", "NeSy", "NeSy_a0.1", "RuleOnly", "DT", "RF"]
ARCH = ["MLPBaseline", "JointCBM_g0", "JointCBM_g0.1", "JointCBM_g0.5", "JointCBM_g1.0", "SequentialCBM", "HybridCBM"]
LA = ["0", "0.05", "0.1", "0.2", "0.5", "1.0"]
ms = lambda a: f"{np.mean(a):.3f} ({np.std(a):.3f})"


def head(t):
    print(f"\n== {t}")


def protocol():
    head("tab:protocol (rows per split; unknown pool)")
    for ds in ("ctuk", "cicr", "ton"):
        if not (DATA / ds / "train.parquet").exists():
            print(f"   {ds:5s} splits not generated (see README, Data)"); continue
        n = {s: pq.ParquetFile(DATA / ds / f"{s}.parquet").metadata.num_rows
             for s in ("train", "val", "test_known", "test_unknown")}
        print(f"   {ds:5s} " + "  ".join(f"{k} {v:,}" for k, v in n.items()))


def table_main():
    head("tab:main: F1 | AUROC mean±std | TPR@5% mean±std   (CTU ctuk, CIC cicr)")
    res = {ds: with_baselines(ds) for ds in ("ctuk", "cicr")}
    for m in MAIN_ROWS:
        cells = []
        for ds in ("ctuk", "cicr"):
            r = res[ds].get(m)
            cells.append("-" if r is None else f"{r[0]:.4f}  {r[1]:.3f}±{r[2]:.3f}  {r[3]:.3f}±{r[4]:.3f}")
        print(f"   {m:14s} " + "   |   ".join(cells))
    head("tab:kfs: F1 | AUROC mean±std, ctuko then ctuk")
    res["ctuko"] = with_baselines("ctuko")
    for m in MAIN_ROWS:
        if m in res["ctuko"]:
            print(f"   {m:14s} " + "   |   ".join(f"{res[d][m][0]:.3f}  {res[d][m][1]:.3f}±{res[d][m][2]:.3f}"
                                                  for d in ("ctuko", "ctuk")))


def splits():
    head("tab:gamma_splits: JointCBM AUROC per gamma, delta = g1.0 - g0")
    for ds in ("ctuk", "ctualt", "ctualt_o", "ctut", "ctut_o", "cicr", "cicr_alt"):
        r = split_rows(ds)
        m = [np.mean([x["auroc"] for x in r[g]]) for g in GAMMAS]
        print(f"   {ds:9s} " + "  ".join(f"{v:.3f}" for v in m) + f"  delta {m[3] - m[0]:+.3f}")
    for ds in ("ctualt", "cicr_alt"):
        r = split_rows(ds)
        print(f"   known F1 {ds}: " + "  ".join(f"{g} {np.mean([x['f1'] for x in r[g]]):.4f} "
                                               f"(min seed {min(x['f1'] for x in r[g]):.3f})"
                                               for g in ["MLPBaseline"] + GAMMAS))
    head("tab:temporal (ctut): Benign C&C-HB DDoS Okiru | F1_4 (std) | Coh-3 (std) | AUROC")
    r = split_rows("ctut")
    for m in ["MLPBaseline"] + GAMMAS + ["SequentialCBM", "HybridCBM", "NeSy", "NeSy_a0.1"]:
        rs = r[m]
        au = [x["auroc"] for x in rs]
        if all(x["per"] for x in rs):
            per = np.mean([x["per"] for x in rs], axis=0)
            f4 = [x["f1"] for x in rs]; coh = [np.mean([x["per"][i] for i in COH3]) for x in rs]
            print(f"   {m:14s} " + " ".join(f"{v:.3f}" for v in per) + f" | {ms(f4)} | {ms(coh)} | "
                  f"{np.mean(au):.3f}±{np.std(au):.3f}  DDoS F1 per seed "
                  + " ".join(f"{x['per'][2]:.2f}" for x in rs))
        else:
            print(f"   {m:14s} AUROC {np.mean(au):.3f}±{np.std(au):.3f}")


def scorers():
    head("tab:scorer: NeSy AUROC by scorer, mean (std)")
    for ds, f in (("ctuk", "scorer_ablation_ctuk_ctuko.json"), ("cicr", "scorer_ablation_cicr.json")):
        p = NESY / f
        if not p.exists():
            print(f"   {ds}: {f} missing"); continue
        d = LOAD(p)[ds]
        print(f"   {ds:5s} " + "  ".join(f"{k} {v['mean']:.3f} ({v['std']:.3f})" for k, v in d.items()))
    head("tab:scorer_arch / tab:kfs_scorer: Mahal | Energy | Comb, mean (std)")
    for ds, f in (("ctuk", "scorer_factorial_ctuk_ctuko.json"), ("ctuko", "scorer_factorial_ctuk_ctuko.json"),
                  ("cicr", "scorer_factorial_cicr.json")):
        rows = [r for r in LOAD(CBM / f) if r["dataset"] == ds]
        for v in ARCH:
            rs = [r for r in rows if r["variant"] == v]
            print(f"   {ds:5s} {v:14s} " + "  ".join(ms([r[k] for r in rs]) for k in ("auroc_mahal", "auroc_energy", "auroc_comb")))
    head("tab:ood_baselines (seed-0 MLP)")
    for f, ds in (("ood_baselines_ctuk_ctuko.json", "ctuk"), ("ood_baselines_cicr.json", "cicr")):
        p = CBM / f
        print(f"   {ds:5s} " + ("missing" if not p.exists() else "  ".join(f"{k} {v:.3f}" for k, v in LOAD(p)[ds].items())))
    head("tab:lofo")
    for k, v in LOAD(CBM / "lofo.json").items():
        print(f"   {k:22s} {v['held_out']:20s} Mahal {v['mahal']:.2f}  Energy {v['energy']:.2f}  M/E {v['winner_M']}/{v['winner_E']}")


def openset():
    for ds in ("ctuk", "cicr"):
        d = LOAD(CBM / f"openset_{ds}.json")
        head(f"tab:openset_agg {ds}: known rej % (std) | macro-F1 (std) | eq-fam AUROC | eq-fam recall | tau CV")
        for m, r in d["models"].items():
            print(f"   {m:14s} {100 * r['known_rej'][0]:.1f} ({100 * r['known_rej'][1]:.1f}) | {r['macro_f1'][0]:.3f} "
                  f"({r['macro_f1'][1]:.3f}) | {r['eqfam_auroc'][0]:.3f} | {r['eqfam_recall'][0]:.2f} | {r['tau_cv']:.2f}")
        head(f"tab:openset_perfam {ds}: JointCBM AUROC rec | NeSy AUROC rec")
        j, n = d["models"]["JointCBM_g0"]["per_family"], d["models"]["NeSy"]["per_family"]
        for fam in j:
            print(f"   {fam:28s} {j[fam]['auroc']:.3f} {j[fam]['recall']:.3f} | {n[fam]['auroc']:.3f} {n[fam]['recall']:.3f}")
        if ds == "cicr":
            c = np.array(d["models"]["NeSy"]["rows"][0]["confusion"])
            head("tab:confusion (NeSy cicr seed 0), rows true; last row unknown")
            for name, row in zip(d["classes"], c):
                print(f"   {name[:16]:16s} " + " ".join(f"{x:6d}" for x in row))
            u = c[-1]
            print(f"   unknown total {u.sum():,}; rejected {u[-1]:,} ({u[-1] / u.sum():.3f}); "
                  f"unknown:known {u.sum() / c[:-1].sum():.1f}")
            tp = np.diag(c).astype(float); prec = tp / c.sum(0); rec = tp / c.sum(1)
            f1 = 2 * prec * rec / (prec + rec)
            print("   per-class precision " + " ".join(f"{x:.2f}" for x in prec) + " | F1 " + " ".join(f"{x:.2f}" for x in f1)
                  + f" | macro {f1.mean():.3f}")
            w = c.astype(float).copy(); w[-1] *= c[:-1].sum() / u.sum()
            tp = np.diag(w); prec = tp / w.sum(0); rec = tp / w.sum(1)
            print(f"   reweighted to equal known/unknown totals: macro-F1 {np.mean(2 * prec * rec / (prec + rec)):.3f}")


def fidelity():
    for ds in ("ctuk", "cicr"):
        d = LOAD(CBM / f"fidelity_{ds}.json")
        head(f"tab:cintervene / tab:fidelity CBM {ds}: F1_pred (std) | F1_gt (std) | gap (std) | probe gap (std); LR {d['f1_lr']:.3f}")
        for g, r in d["gamma"].items():
            gap = [x["f1_pred"] - x["f1_gt"] for x in r["rows"]]
            print(f"   {g:14s} {r['f1_pred'][0]:.3f} ({r['f1_pred'][1]:.3f}) | {r['f1_gt'][0]:.3f} ({r['f1_gt'][1]:.3f}) | "
                  f"{ms(gap)} | {r['probe_gap'][0]:.3f} ({r['probe_gap'][1]:.3f})")
        n = LOAD(NESY / f"fidelity_{ds}.json")
        head(f"tab:alpha / tab:headagree {ds}: lambda_a: alpha (std) | dep_neural | dep_rule (std) | hard_soft | crisp | AUROC (std)")
        for la in LA:
            s = n["sweep"][la]; rs = s["rows"]
            print(f"   {la:5s} {s['alpha']:.2f} ({np.std([r['alpha'] for r in rs]):.2f}) | {s['dep_neural']:.3f} | "
                  f"{s['dep_rule']:.2f} ({np.std([r['dep_rule'] for r in rs]):.2f}) | {s['hard_soft']:.4f} | "
                  f"{s['crisp']:.3f} | {ms([r['auroc'] for r in rs])}")
        head(f"tab:rulequality {ds}: cov | dominant | prec | target | target prec (std) | thr rel std")
        for name, r in n["rule_quality"].items():
            print(f"   {name:24s} {r['coverage']:.2f} | {r['dominant'][:14]:14s} | {r['precision']:.2f} | {r['target'][:14]:14s} | "
                  f"{r['target_precision']:.2f} ({r['target_precision_std']:.2f}) | {r['threshold_rel_std']:.2f}")
        td = n["threshold_drift"]
        drift0 = np.array(td["drift_per_seed"][0])
        i = int(np.argmin(drift0))
        print(f"   threshold drift mean pairwise r {td['mean_pairwise_r']:.2f}; seed-0 largest negative drift "
              f"{drift0[i]:+.2f} ({td['labels'][i]})")


def selectivity():
    for ds in ("ctuk", "cicr"):
        d = LOAD(NESY / f"{ds}_nesy_s0_selectivity.json")
        classes = list(next(iter(d.values())))
        head(f"rule selectivity {ds} seed 0 (mean binarised activation per known class, validation split)")
        print("   " + " " * 24 + " ".join(f"{c[:12]:>12s}" for c in classes))
        for rule, row in d.items():
            print(f"   {rule:24s}" + " ".join(f"{row[c]:12.2f}" for c in classes))


def geometry():
    g = LOAD(CBM / "geometry.json")
    head("tab:signflip / tab:loco: r30 | r6 | LOCO | r without SequentialCBM")
    for ds in ("ctuk", "cicr", "ton"):
        print(f"   [{ds}] mean within-var {g['cbm'][ds]['mean_within_var']:.3f}, mean AUROC {g['cbm'][ds]['mean_auroc']:.3f}")
        for k, c in g["cbm"][ds]["corr"].items():
            print(f"   {ds:5s} {k:12s} {c['r30']:+.2f} | {c['r6']:+.2f} | [{c['loco'][0]:+.2f},{c['loco'][1]:+.2f}] | "
                  f"{c['r_no_seq']:+.2f}{'  (sign flips w/o Seq)' if c['sign_flips_without_seq'] else ''}")
        rows = g["cbm"][ds]["rows"]
        for v in ARCH:
            rs = [r for r in rows if r["config"] == v]
            if rs:
                print(f"      {v:14s} within_var {np.mean([r['within_var'] for r in rs]):.4f}  AUROC {np.mean([r['auroc'] for r in rs]):.3f}")
    print(f"   dataset-means r(within_var, AUROC) {g['cbm']['dataset_means_r']:+.2f}; NeSy ten-model r {g['nesy']['ten_model_r']:.3f}")
    for ds, k in g["nesy"]["kt"].items():
        print(f"   k_T sweep {ds}: {json.dumps(k)[:400]}")


if __name__ == "__main__":
    for f in (protocol, table_main, splits, scorers, openset, fidelity, selectivity, geometry):
        f()
