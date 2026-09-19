# Confounding figure: within-class variance does NOT govern OOD AUROC. The
# between-dataset association (higher-variance dataset -> higher AUROC) reverses
# within a dataset across architectures (Simpson's paradox), so the geometry
# statistic is a confound, not a mechanism. One point per CBM model (7 variants x
# 5 seeds x 2 datasets); dashed lines are per-dataset least-squares fits.

import json, sys
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
rows = [r for r in json.loads((ROOT / "results" / "cbm" / "geometry_multiseed.json").read_text())
        if r["interp"] and r["auroc"] is not None]
OUT = ROOT / "paper" / "figures" / "confounding.pdf"
DS = {"ctu": ("CTU-IoT-23", "#1565C0", "o"), "cic": ("CIC-IoT-2023", "#C62828", "s")}


def fit(xs, ys):
    mx, my = mean(xs), mean(ys)
    b = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)
    return b, my - b * mx


fig, ax = plt.subplots(figsize=(6.4, 4.6))
allx, ally = [], []
for ds, (label, c, mk) in DS.items():
    rs = [r for r in rows if r["dataset"] == ds]
    xs = [r["within_var"] for r in rs]; ys = [r["auroc"] for r in rs]
    allx += xs; ally += ys
    ax.scatter(xs, ys, c=c, marker=mk, s=40, alpha=0.7, edgecolor="white", label=label)
    b, a = fit(xs, ys)
    xr = [min(xs), max(xs)]
    ax.plot(xr, [a + b * x for x in xr], "--", c=c, lw=1.6)
# between-dataset means arrow
mctu = (mean(r["within_var"] for r in rows if r["dataset"] == "ctu"), mean(r["auroc"] for r in rows if r["dataset"] == "ctu"))
mcic = (mean(r["within_var"] for r in rows if r["dataset"] == "cic"), mean(r["auroc"] for r in rows if r["dataset"] == "cic"))
ax.plot([mcic[0], mctu[0]], [mcic[1], mctu[1]], "-", c="0.4", lw=2.2, zorder=1)
ax.scatter([mctu[0], mcic[0]], [mctu[1], mcic[1]], c="0.3", marker="*", s=210, zorder=5, label="dataset means (between: +)")
ax.set_xlabel("Within-class concept-activation variance", fontsize=11)
ax.set_ylabel("OOD AUROC (per-class Mahalanobis)", fontsize=11)
ax.set_title("Within-class variance does not govern detectability", fontsize=11)
ax.legend(fontsize=8.5, loc="lower left"); ax.grid(alpha=0.3)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, bbox_inches="tight", dpi=150)
plt.savefig(str(OUT).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print("wrote", OUT)
