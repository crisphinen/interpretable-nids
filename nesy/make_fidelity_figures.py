#!/usr/bin/env python3
# Fidelity figures from the NeSy-NIDS models:
#   threshold_drift5_{ctu,cic}   per-condition drift (learned - initial), bars = five-seed mean,
#                                dots = seeds (fig:threshdrift); reads results/nesy/fidelity_{ds}.json
#   rule_selectivity1_{ctu,cic}  mean hard activation per rule and known class, seed 0
#                                (fig:ruleactiv); reads results/nesy/{ds}_nesy_s0_selectivity.json
# The CTU figures use ctuk and the CIC figures cicr. Output: results/figures/ (PDF + PNG).
#
# usage: python -m nesy.make_fidelity_figures
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "nesy"
OUT = ROOT / "results" / "figures"
PANELS = [("ctuk", "ctu", "CTU-IoT-23"), ("cicr", "cic", "CIC-IoT-2023")]


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {OUT / stem}.pdf")


def drift_figure(ds, short, title):
    d = json.loads((RES / f"fidelity_{ds}.json").read_text())["threshold_drift"]
    V = np.array(d["drift_per_seed"]); mean = V.mean(0); xs = np.arange(len(mean))
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(mean)), 3.6))
    ax.bar(xs, mean, color=np.where(mean > 0, "#d62728", "#1f77b4"), alpha=0.8, width=0.65)
    for v in V:
        ax.scatter(xs, v, s=9, color="#222", zorder=5)
    ax.axhline(0, color="#333", lw=0.9, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(d["labels"], fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel("Learned - initial threshold", fontsize=8.5)
    ax.set_title(f"Threshold drift: {title} (5 seeds; mean pairwise r = {d['mean_pairwise_r']:.2f})",
                 fontsize=8.5, fontweight="bold")
    save(fig, f"threshold_drift5_{short}")


def selectivity_figure(ds, short, title):
    sel = json.loads((RES / f"{ds}_nesy_s0_selectivity.json").read_text())
    rules = list(sel); classes = list(sel[rules[0]])
    M = np.array([[sel[r][c] for c in classes] for r in rules])
    fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(classes)), max(3.8, 0.5 * len(rules))))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=8, rotation=22, ha="right")
    ax.set_yticks(range(len(rules))); ax.set_yticklabels(rules, fontsize=7.5)
    for i in range(len(rules)):
        for j in range(len(classes)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > 0.55 else "#333")
    ax.set_title(f"Rule class selectivity: {title} (seed 0, k=10)", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.75, label="Mean activation")
    save(fig, f"rule_selectivity1_{short}")


if __name__ == "__main__":
    for ds, short, title in PANELS:
        drift_figure(ds, short, title)
        selectivity_figure(ds, short, title)
