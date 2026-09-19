# Figure for the training-time K_FINAL sweep: shows that activation-space OOD
# detectability is governed by the interpretable representation's within-class
# variance, demonstrated causally within a dataset (not just between the two).
#
# (a) OOD AUROC (hard scorer) vs annealing steepness k, per dataset, error bars
#     over 5 seeds -- CIC rises with k, CTU is flat (F1 constant throughout).
# (b) OOD AUROC vs within-class rule-activation variance, one point per run --
#     the mechanism: variance governs detectability; k moves CIC along it, CTU
#     sits in a high-variance regime k cannot leave.

import json, sys, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
d = json.loads((ROOT / "results" / "nesy" / "ksweep_train.json").read_text())
OUT = ROOT / "paper" / "figures" / "ksweep.pdf"

DS = {"ctu": ("CTU-IoT-23", "#1565C0", "o"), "cic": ("CIC-IoT-2023", "#C62828", "s")}


def cell(ds, kf):
    return [d[k] for k in d if k.startswith(f"{ds}|k{kf}|")]


def curve(ds):
    ks, mu, sd, f1 = [], [], [], []
    for kf in [2.0, 5.0, 10.0, 20.0]:
        r = cell(ds, kf)
        if not r:
            continue
        ks.append(kf)
        mu.append(st.mean(v["auroc_hard"] for v in r))
        sd.append(st.pstdev([v["auroc_hard"] for v in r]) if len(r) > 1 else 0)
        f1.append(st.mean(v["f1"] for v in r))
    return ks, mu, sd, f1


fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

# (a) AUROC vs k
for ds, (label, c, mk) in DS.items():
    ks, mu, sd, f1 = curve(ds)
    if not ks:
        continue
    axA.errorbar(ks, mu, yerr=sd, marker=mk, color=c, capsize=3, lw=2, ms=7,
                 label=f"{label}  (F1$\\approx${st.mean(f1):.3f})")
axA.axhline(0.5, ls=":", c="gray", lw=1)
axA.set_xscale("log"); axA.set_xticks([2, 5, 10, 20]); axA.set_xticklabels(["2", "5", "10", "20"])
axA.set_xlabel("Final annealing steepness $k$", fontsize=11)
axA.set_ylabel("OOD AUROC (per-class Mahalanobis)", fontsize=11)
axA.set_title("(a) Detectability vs annealing hardness", fontsize=11)
axA.legend(fontsize=9, loc="center right"); axA.grid(alpha=0.3)

# (b) AUROC vs within-class variance -- one point per run
for ds, (label, c, mk) in DS.items():
    xs = [d[k]["wvar_hard"] for k in d if k.startswith(f"{ds}|k")]
    ys = [d[k]["auroc_hard"] for k in d if k.startswith(f"{ds}|k")]
    if not xs:
        continue
    axB.scatter(xs, ys, c=c, marker=mk, s=42, alpha=0.75, edgecolor="white", label=label)
axB.set_xlabel("Within-class rule-activation variance", fontsize=11)
axB.set_ylabel("OOD AUROC (per-class Mahalanobis)", fontsize=11)
axB.set_title("(b) Variance governs detectability", fontsize=11)
axB.legend(fontsize=9, loc="lower right"); axB.grid(alpha=0.3)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, bbox_inches="tight", dpi=150)
plt.savefig(str(OUT).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print("wrote", OUT)
