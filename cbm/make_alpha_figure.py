#!/usr/bin/env python3
# alpha_sweep figure: learned gate alpha, OOD AUROC and known-class F1 vs the
# alpha-regulariser weight lambda_alpha, for CTU-IoT-23 and CIC-IoT-2023.
# Renders from results/nesy/r2_alpha_sweep.json (mean over five seeds; built by
# nesy/aggregate_alpha_sweep.py from the nesy/run_alpha_sweep.sh per-seed evals).
#
# usage: python -m cbm.make_alpha_figure

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
NESY_RES = PROJECT_ROOT / "results" / "nesy"

OUTDIRS = [
    PROJECT_ROOT / "paper" / "figures",
    Path("/home/Ngari/Research/writing/latex/figures"),
    Path("/home/Ngari/Research/writing/latex_conf/figures"),
]
for d in OUTDIRS:
    d.mkdir(parents=True, exist_ok=True)

STYLE = {
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
}
plt.rcParams.update(STYLE)

C_ALPHA = "#59a14f"   # learned gate (rule reliance)
C_AUROC = "#4e79a7"   # OOD AUROC
C_F1    = "#f28e2b"   # known-class F1


def _save(fig, stem):
    for d in OUTDIRS:
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {stem} done")


def _panel(ax, d, title):
    # order lambda_alpha points numerically
    keys = sorted(d.keys(), key=float)
    x = np.arange(len(keys))
    lam = [float(k) for k in keys]

    alpha   = np.array([d[k]["alpha"] for k in keys])
    a_std   = np.array([d[k].get("alpha_std", 0.0) for k in keys])
    auroc   = np.array([d[k]["auroc"] for k in keys])
    au_std  = np.array([d[k].get("auroc_std", 0.0) for k in keys])
    f1      = np.array([d[k]["f1"] for k in keys])

    ax.errorbar(x, alpha, yerr=a_std, marker="o", color=C_ALPHA, lw=1.6,
                capsize=2.5, label=r"$\alpha$ (rule reliance)")
    ax.errorbar(x, auroc, yerr=au_std, marker="s", color=C_AUROC, lw=1.6,
                capsize=2.5, label="OOD AUROC")
    ax.plot(x, f1, marker="^", color=C_F1, lw=1.6, ls="--",
            label="known-class F1")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:g}" for v in lam])
    ax.set_xlabel(r"$\lambda_\alpha$")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)


def main():
    d = json.load(open(NESY_RES / "r2_alpha_sweep.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    _panel(axes[0], d["ctu"], "(a) CTU-IoT-23")
    _panel(axes[1], d["cic"], "(b) CIC-IoT-2023")
    axes[0].set_ylabel("value")
    axes[1].legend(loc="center right", frameon=False)
    fig.tight_layout()
    _save(fig, "alpha_sweep")


if __name__ == "__main__":
    main()
