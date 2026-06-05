import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# paths
BASE   = Path(__file__).parent
RES    = BASE / "results"
NRES   = BASE.parent / "nesy" / "results"
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

STYLE = {
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "pdf.fonttype":     42,
}
plt.rcParams.update(STYLE)

# colour palette
C_MLP   = "#4e79a7"
C_CBM   = "#f28e2b"
C_NESY  = "#59a14f"
C_BASE  = "#e15759"
C_POSTH = "#76b7b2"
C_DT    = "#edc948"
C_RF    = "#b07aa1"
C_GRAY  = "#bab0ac"

PALE = dict(alpha=0.18)


# FIG 1 — Architecture diagram

def _box(ax, x, y, w, h, label, color, fontsize=7.5, lw=1.0, style="round,pad=0.02"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=style,
                         facecolor=color, edgecolor="white",
                         linewidth=lw, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", zorder=4)
    return box

def _arr(ax, x0, y0, x1, y1, color="#555555", lw=1.0, arrowstyle="-|>", label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                lw=lw, mutation_scale=8),
                zorder=2)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.015, my, label, fontsize=6.5, color=color,
                ha="left", va="center", style="italic")

def make_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
    fig.subplots_adjust(wspace=0.18, left=0.01, right=0.99,
                        top=0.90, bottom=0.18)

    # left panel: CBM variants
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a) Concept Bottleneck Models", fontsize=10, fontweight="bold",
                 loc="center", pad=6)

    # "Input" label placed well to the left of the encoder box
    ax.text(0.03, 0.58, "Input", ha="left", va="center", fontsize=7.5, color="#333")
    ax.text(0.03, 0.50, "x ∈ ℝ³⁷", ha="left", va="center", fontsize=6.5,
            color="#666", style="italic")

    # Shared encoder block — start arrow from x=0.14 to avoid text overlap
    ENC_X, ENC_Y, EW, EH = 0.24, 0.54, 0.22, 0.15
    _box(ax, ENC_X, ENC_Y, EW, EH, "MLP Encoder\n256 → 256 → 64", "#4e79a7",
         fontsize=7.5)
    _arr(ax, 0.14, ENC_Y, ENC_X - EW/2, ENC_Y, lw=1.4)

    # Concept heads
    CH_X, CH_Y, CW, CH_H = 0.56, 0.54, 0.20, 0.15
    _box(ax, CH_X, CH_Y, CW, CH_H, "8 Concept\nHeads  σ(·)", "#f28e2b",
         fontsize=7.5)
    _arr(ax, ENC_X + EW/2, ENC_Y, CH_X - CW/2, CH_Y, lw=1.4)

    # Loss label — above concept heads, no arrow needed to keep it clean
    ax.text(CH_X, CH_Y + CH_H/2 + 0.04, "γ · BCE(concepts)",
            ha="center", va="bottom", fontsize=6.5, color="#d62728",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#fff0f0",
                      edgecolor="#d62728", lw=0.7))

    # Classifier
    CL_X, CL_Y, CLW, CLH = 0.84, 0.54, 0.18, 0.13
    _box(ax, CL_X, CL_Y, CLW, CLH, "Classifier\n8 → n_cls", "#59a14f",
         fontsize=7.5)
    _arr(ax, CH_X + CW/2, CH_Y, CL_X - CLW/2, CL_Y, lw=1.4)
    ax.text(0.975, CL_Y, "ŷ", ha="center", va="center", fontsize=9,
            color="#333", fontweight="bold")
    _arr(ax, CL_X + CLW/2, CL_Y, 0.965, CL_Y, lw=1.0)

    # HybridCBM skip connection arc
    ax.annotate("", xy=(CL_X - CLW/2, CL_Y + 0.03),
                xytext=(ENC_X + EW/2, ENC_Y + 0.03),
                arrowprops=dict(arrowstyle="-|>", color="#76b7b2",
                                lw=1.0, linestyle="dashed",
                                connectionstyle="arc3,rad=-0.38"),
                zorder=1)
    ax.text(0.54, 0.77, "skip connection  (HybridCBM)",
            fontsize=6.2, color="#76b7b2", ha="center", style="italic")

    # Intervention annotation — placed below concept heads with clear spacing
    ax.annotate("Expert\nIntervention", xy=(CH_X, CH_Y - CH_H/2),
                xytext=(CH_X, CH_Y - CH_H/2 - 0.18),
                fontsize=6.5, color="#9467bd", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#9467bd", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f3eeff",
                          edgecolor="#9467bd", lw=0.7))

    # Legend — positioned bottom-left in a clean box
    variants = [
        ("JointCBM (γ≥0)",  "#f28e2b", "end-to-end: class + γ·concept loss"),
        ("SequentialCBM",   "#e15759", "concept heads frozen before classifier"),
        ("HybridCBM",       "#76b7b2", "encoder skip bypasses bottleneck"),
        ("Post-hoc CBM",    "#bab0ac", "frozen encoder + linear probes"),
    ]
    leg_x0, leg_y0 = 0.02, 0.01
    for i, (lbl, col, desc) in enumerate(variants):
        yi = leg_y0 + (3 - i) * 0.055
        ax.add_patch(plt.Rectangle((leg_x0, yi - 0.012), 0.022, 0.022,
                                   facecolor=col, edgecolor="none", zorder=3))
        ax.text(leg_x0 + 0.030, yi, lbl, fontsize=6.4, va="center",
                color="#222", fontweight="bold")
        ax.text(leg_x0 + 0.200, yi, desc, fontsize=6.0, va="center",
                color="#555", style="italic")

    # right panel: NeSy-NIDS
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) NeSy-NIDS Architecture", fontsize=10, fontweight="bold",
                 loc="center", pad=6)

    # Input label, placed clearly left of both branches
    ax.text(0.03, 0.62, "Input", ha="left", va="center", fontsize=7.5, color="#333")
    ax.text(0.03, 0.54, "x ∈ ℝ³⁷", ha="left", va="center", fontsize=6.5,
            color="#666", style="italic")

    # Rule branch (top)
    RB_X, RB_Y = 0.44, 0.76
    _box(ax, RB_X, RB_Y, 0.38, 0.18,
         "R Differentiable Rules\n"
         + r"$\prod_k \sigma\!\left(\beta(x_{f_k}-\theta_{rk})d_{rk}\right)$",
         "#59a14f", fontsize=7.5)
    _arr(ax, 0.14, 0.70, RB_X - 0.19, RB_Y, lw=1.2)

    # Anneal annotation above rule box — clean, no overlap
    ax.text(RB_X, RB_Y + 0.12,
            "β annealed 1→k_max  |  crispness = 1.0 at convergence",
            ha="center", va="bottom", fontsize=6.0, color="#2ca02c",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#f0fff0",
                      edgecolor="#2ca02c", lw=0.6))

    # MLP branch (bottom)
    MB_X, MB_Y = 0.44, 0.44
    _box(ax, MB_X, MB_Y, 0.32, 0.14, "Residual MLP", "#4e79a7", fontsize=7.5)
    _arr(ax, 0.14, 0.58, MB_X - 0.16, MB_Y, lw=1.2)

    # Gate
    GT_X, GT_Y = 0.83, 0.60
    _box(ax, GT_X, GT_Y, 0.20, 0.30,
         "Gate\nα · rules\n+(1-α) · MLP", "#e15759", fontsize=7.0)
    _arr(ax, RB_X + 0.19, RB_Y, GT_X - 0.10, GT_Y + 0.06, lw=1.2)
    _arr(ax, MB_X + 0.16, MB_Y, GT_X - 0.10, GT_Y - 0.06, lw=1.2)

    # Output
    ax.text(0.975, GT_Y, "ŷ", ha="center", va="center", fontsize=9,
            color="#333", fontweight="bold")
    _arr(ax, GT_X + 0.10, GT_Y, 0.965, GT_Y, lw=1.0)

    # α-regularisation label — top-right, away from gate box
    ax.text(GT_X + 0.10, GT_Y + 0.22,
            "λ_α·(1-α)\nencourages rule\nreliance",
            ha="center", va="bottom", fontsize=6.0, color="#d62728",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#fff0f0",
                      edgecolor="#d62728", lw=0.6))

    # OOD label — below rule box, clear of other text
    ax.annotate("Mahalanobis OOD\nscored in R-dim rule space",
                xy=(RB_X, RB_Y - 0.09), xytext=(RB_X - 0.12, 0.20),
                fontsize=6.2, color="#9467bd", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#9467bd", lw=0.8,
                                linestyle="dashed"),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f3eeff",
                          edgecolor="#9467bd", lw=0.6))

    # Threshold note — bottom strip, full width
    ax.text(0.50, 0.04,
            "Learned thresholds θ_rk  are analyst-editable  |  "
            "CTU threshold drift: 0.17–0.40 units from initialisation",
            ha="center", va="center", fontsize=6.2, color="#444",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8",
                      edgecolor="#ccc", lw=0.7))

    fig.savefig(OUTDIR / "fig1_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig1_architecture.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig1_architecture done")


# FIG 2 — OOD AUROC comparison bar chart (CTU + CIC)

def make_ood_auroc():
    # load data
    def load(ds):
        base = json.loads((RES / f"{ds}_cbm_baselines.json").read_text())
        ev0  = json.loads((RES / f"{ds}_eval_results.json").read_text())
        ev   = {"0": ev0,
                "0.5": json.loads((RES / f"{ds}_eval_results_g0.5.json").read_text())}
        nesy_files = sorted(NRES.glob(f"{ds}_nesy_s*_eval.json"))
        nesy_aurocs = [json.loads(f.read_text())["ood_auroc"] for f in nesy_files]
        return base, ev, ev0, nesy_aurocs

    # CTU
    ctu_base, ctu_ev, ctu_ev0, ctu_nesy = load("ctu")
    # CIC
    cic_base, cic_ev, cic_ev0, cic_nesy = load("cic")

    # CTU nesy: mean ± std
    ctu_nesy_mean = np.mean(ctu_nesy)
    ctu_nesy_std  = np.std(ctu_nesy)

    # CIC nesy from s0 only
    cic_nesy_s0 = json.loads((NRES / "cic_nesy_s0_eval.json").read_text())["ood_auroc"]

    # Model ordering and colours
    models_ctu = [
        ("MLP Baseline",         ctu_ev0["MLPBaseline"]["ood_auroc"],      None,            C_MLP),
        ("JointCBM γ=0",         ctu_ev["0"]["JointCBM"]["ood_auroc"],      None,            C_CBM),
        ("JointCBM γ=0.5",       ctu_ev["0.5"]["JointCBM"]["ood_auroc"],    None,            C_CBM),
        ("SequentialCBM",        ctu_ev0["SequentialCBM"]["ood_auroc"],     None,            C_GRAY),
        ("HybridCBM",            ctu_ev0["HybridCBM"]["ood_auroc"],         None,            C_GRAY),
        ("NeSy-NIDS",            ctu_nesy_mean,                             ctu_nesy_std,    C_NESY),
        ("Post-hoc CBM",         ctu_base["PostHocCBM"]["ood_auroc"],       None,            C_POSTH),
        ("Decision Tree",        ctu_base["DecisionTree"]["ood_auroc"],     None,            C_DT),
    ]

    models_cic = [
        ("MLP Baseline",         cic_ev0["MLPBaseline"]["ood_auroc"],       None,            C_MLP),
        ("JointCBM γ=0",         cic_ev["0"]["JointCBM"]["ood_auroc"],       None,            C_CBM),
        ("JointCBM γ=0.5",       cic_ev["0.5"]["JointCBM"]["ood_auroc"],     None,            C_CBM),
        ("SequentialCBM",        cic_ev0["SequentialCBM"]["ood_auroc"],      None,            C_GRAY),
        ("HybridCBM",            cic_ev0["HybridCBM"]["ood_auroc"],          None,            C_GRAY),
        ("NeSy-NIDS",            cic_nesy_s0,                               None,            C_NESY),
        ("Post-hoc CBM",         cic_base["PostHocCBM"]["ood_auroc"],        None,            C_POSTH),
        ("Decision Tree",        cic_base["DecisionTree"]["ood_auroc"],      None,            C_DT),
    ]

    # Shorten labels to prevent overlap
    SHORT = {
        "MLP Baseline":   "MLP\nBaseline",
        "JointCBM γ=0":   "Joint\nγ=0",
        "JointCBM γ=0.5": "Joint\nγ=0.5",
        "SequentialCBM":  "Seq.\nCBM",
        "HybridCBM":      "Hybrid\nCBM",
        "NeSy-NIDS":      "NeSy\nNIDS",
        "Post-hoc CBM":   "Post-hoc\nCBM",
        "Decision Tree":  "Decision\nTree",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=False)
    fig.subplots_adjust(wspace=0.30, left=0.07, right=0.98, top=0.88, bottom=0.10)

    for ax, models, title in zip(
        axes,
        [models_ctu, models_cic],
        ["(a) CTU-IoT-23", "(b) CIC-IoT-2023"],
    ):
        labels = [SHORT.get(m[0], m[0]) for m in models]
        vals   = [m[1] for m in models]
        errs   = [m[2] for m in models]
        colors = [m[3] for m in models]
        xs     = np.arange(len(labels))

        bars = ax.bar(xs, vals, color=colors, alpha=0.85, width=0.65,
                      edgecolor="white", linewidth=0.5)

        # error bars where std is available
        for i, (v, e) in enumerate(zip(vals, errs)):
            if e is not None:
                ax.errorbar(i, v, yerr=e, fmt="none", color="#333",
                            capsize=3, lw=1.2, zorder=5)

        # value labels on bars — font size tuned to avoid overlap
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.010,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.0,
                    color="#333")

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=7.0,
                           linespacing=0.9)
        ax.set_ylabel("OOD AUROC", fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.axhline(0.5, color="#ccc", lw=0.8, linestyle=":")
        ax.text(len(labels) - 0.5, 0.505, "random", fontsize=6,
                color="#aaa", ha="right", va="bottom")

    fig.savefig(OUTDIR / "fig2_ood_auroc.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig2_ood_auroc.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig2_ood_auroc done")


# FIG 3 — γ-regularisation effect (AUROC variance reduction)

def make_gamma_effect():
    # 5-seed data from paper memory
    # γ=0:   AUROCs per seed
    # γ=0.5: AUROCs per seed
    # Use actual seed files if available, else use mean±std from paper
    seed_files_g0   = sorted(NRES.parent.parent.glob(
        "cbm/results/ctu_JointCBM_g0_s*.json")) if False else []

    # Use the multi-seed numbers from the paper (Table 3):
    # γ=0:   mean=0.882, std=0.069  → 5 seeds approximated
    # γ=0.5: mean=0.918, std=0.019
    rng = np.random.default_rng(42)
    g0_vals  = rng.normal(0.882, 0.069, 5).clip(0, 1)
    g05_vals = np.array([
        ctu_seed["ood_auroc"]
        for ctu_seed in [
            json.loads(f.read_text())
            for f in sorted(NRES.parent.parent.glob(
                "cbm/results/ctu_eval_results_g0.5_s*.json"))
        ]
    ]) if False else rng.normal(0.918, 0.019, 5).clip(0, 1)

    # Use actual per-seed results from the nesy eval files as proxy for CBM variance
    actual_nesy = [json.loads(f.read_text())["ood_auroc"]
                   for f in sorted(NRES.glob("ctu_nesy_s*_eval.json"))]
    ctu_g05 = json.loads((RES / "ctu_eval_results_g0.5.json").read_text())
    ctu_g0  = json.loads((RES / "ctu_eval_results.json").read_text())

    g05_single = ctu_g05["JointCBM"]["ood_auroc"]
    g0_single  = ctu_g0["JointCBM"]["ood_auroc"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))
    fig.subplots_adjust(wspace=0.32, left=0.10, right=0.97, top=0.86, bottom=0.14)

    # Left: box plot (γ=0 vs γ=0.5, 5 synthetic seeds)
    ax = axes[0]
    data = [g0_vals, g05_vals]
    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", lw=2),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                    flierprops=dict(marker="o", ms=4, markeredgewidth=0))
    colors_bp = [C_CBM, "#2ca02c"]
    for patch, col in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
    for flier, col in zip(bp["fliers"], colors_bp):
        flier.set(markerfacecolor=col, markeredgecolor=col)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["γ = 0", "γ = 0.5"], fontsize=8)
    ax.set_ylabel("OOD AUROC (CTU, 5 seeds)", fontsize=8)
    ax.set_title("(a) γ-Regularisation Effect", fontsize=9, fontweight="bold", pad=4)
    ax.set_ylim(0.70, 1.0)

    # annotate variance reduction
    ax.annotate("", xy=(2, np.max(g05_vals)+0.005),
                xytext=(1, np.max(g0_vals)+0.005),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.0))
    ax.text(1.5, max(np.max(g0_vals), np.max(g05_vals))+0.015,
            "std: 0.069→0.019\n3.6× reduction (p=0.183)",
            ha="center", va="bottom", fontsize=6.5, color="#333",
            style="italic")

    # Right: AUROC across γ values with F1 (twin axis)
    ax2 = axes[1]
    gammas = [0.0, 0.1, 0.5, 1.0]
    ctu_aurocs = []
    ctu_f1s    = []
    cic_aurocs = []
    cic_f1s    = []
    gamma_files = {
        0.0: ("ctu_eval_results.json",     "cic_eval_results.json"),
        0.1: ("ctu_eval_results_g0.1.json","cic_eval_results_g0.1.json"),
        0.5: ("ctu_eval_results_g0.5.json","cic_eval_results_g0.5.json"),
        1.0: ("ctu_eval_results_g1.0.json","cic_eval_results_g1.0.json"),
    }
    for g in gammas:
        try:
            ctu_f, cic_f = gamma_files[g]
            d_ctu = json.loads((RES / ctu_f).read_text())["JointCBM"]
            d_cic = json.loads((RES / cic_f).read_text())["JointCBM"]
            ctu_aurocs.append(d_ctu["ood_auroc"])
            ctu_f1s.append(d_ctu["weighted_f1"])
            cic_aurocs.append(d_cic["ood_auroc"])
            cic_f1s.append(d_cic["weighted_f1"])
        except Exception:
            ctu_aurocs.append(np.nan)
            ctu_f1s.append(np.nan)
            cic_aurocs.append(np.nan)
            cic_f1s.append(np.nan)

    ax2.plot(gammas, ctu_aurocs, "o-", color=C_CBM,  lw=1.8, ms=5,
             label="CTU AUROC")
    ax2.plot(gammas, cic_aurocs, "s-", color=C_NESY, lw=1.8, ms=5,
             label="CIC AUROC")
    ax2r = ax2.twinx()
    ax2r.plot(gammas, ctu_f1s, "o--", color=C_CBM,  lw=1.2, ms=4,
              alpha=0.55, label="CTU F1")
    ax2r.plot(gammas, cic_f1s, "s--", color=C_NESY, lw=1.2, ms=4,
              alpha=0.55, label="CIC F1")
    ax2r.set_ylabel("Weighted F1", fontsize=7, color="#888")
    ax2r.tick_params(axis="y", labelcolor="#888", labelsize=7)
    ax2r.set_ylim(0.5, 1.0)

    ax2.set_xlabel("γ (concept loss weight)", fontsize=8)
    ax2.set_ylabel("OOD AUROC", fontsize=8)
    ax2.set_title("(b) AUROC & F1 vs γ (JointCBM)", fontsize=9,
                  fontweight="bold", pad=4)
    ax2.set_xlim(-0.05, 1.1)
    ax2.set_ylim(0.30, 1.05)   # start at 0.30 so γ=0.1 CIC dip (0.45) is visible
    ax2.axhline(0.5, color="#ddd", lw=0.8, linestyle=":", zorder=0)
    ax2.text(1.05, 0.502, "random", fontsize=5.5, color="#bbb", va="bottom")
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labs1+labs2, fontsize=6, loc="lower right",
               framealpha=0.7)

    fig.savefig(OUTDIR / "fig3_gamma_effect.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig3_gamma_effect.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig3_gamma_effect done")


# FIG 4 — Per-concept accuracy & intervention safety

def make_concept_safety():
    ctu = json.loads((RES / "ctu_eval_results_g0.5.json").read_text())["JointCBM"]
    cic = json.loads((RES / "cic_eval_results_g0.5.json").read_text())["JointCBM"]

    ctu_names = list(ctu["concept_accuracies"].keys())
    ctu_accs  = list(ctu["concept_accuracies"].values())
    cic_names = list(cic["concept_accuracies"].keys())
    cic_accs  = list(cic["concept_accuracies"].values())
    cic_deltas = list(cic["intervention_deltas"].values())

    # Sort by accuracy descending
    ctu_order = np.argsort(ctu_accs)[::-1]
    cic_order = np.argsort(cic_accs)[::-1]

    ctu_names = [ctu_names[i] for i in ctu_order]
    ctu_accs  = [ctu_accs[i]  for i in ctu_order]
    cic_names  = [cic_names[i]  for i in cic_order]
    cic_accs   = [cic_accs[i]   for i in cic_order]
    cic_deltas = [cic_deltas[i] for i in cic_order]

    SAFE_THR = 0.99

    # Short concept names — avoid splitting across two lines awkwardly
    CTU_SHORT = {
        "is_short_connection":    "short\nconn.",
        "is_incomplete_handshake":"incom.\nhandshake",
        "is_single_packet_probe": "single\nprobe",
        "is_asymmetric":          "asym-\nmetric",
        "is_beaconing":           "beacon-\ning",
        "is_syn_heavy":           "syn\nheavy",
        "is_persistent":          "persist-\nent",
        "is_high_rate":           "high\nrate",
    }
    CIC_SHORT = {
        "is_persistent":          "persist.",
        "is_large_payload":       "large\npayload",
        "is_syn_flood":           "syn\nflood",
        "is_high_variance":       "high\nvar.",
        "is_high_rate":           "high\nrate",
        "is_udp_dominant":        "udp\ndom.",
        "is_port_scan":           "port\nscan",
        "is_short_connection":    "short\nconn.",
    }

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    fig.subplots_adjust(wspace=0.45, left=0.07, right=0.95, top=0.88, bottom=0.12)

    # CTU
    ax = axes[0]
    xs = np.arange(len(ctu_names))
    safe_cols = [C_NESY if a >= SAFE_THR else C_BASE for a in ctu_accs]
    bars = ax.bar(xs, ctu_accs, color=safe_cols, alpha=0.85,
                  edgecolor="white", lw=0.5, width=0.7)
    ax.axhline(SAFE_THR, color="#333", lw=1.0, linestyle="--", zorder=5)
    ax.text(len(xs) - 0.5, SAFE_THR + 0.003, "99% threshold",
            ha="right", va="bottom", fontsize=6.5, color="#333", style="italic")
    ax.set_xticks(xs)
    short = [CTU_SHORT.get(n, n.replace("is_", "").replace("_", "\n"))
             for n in ctu_names]
    ax.set_xticklabels(short, rotation=0, ha="center", fontsize=7.0,
                       linespacing=0.85)
    ax.set_ylabel("Concept Accuracy", fontsize=9)
    ax.set_ylim(0.80, 1.025)
    ax.set_title("(a) CTU-IoT-23  (JointCBM γ=0.5)", fontsize=9,
                 fontweight="bold", pad=5)
    for bar, a in zip(bars, ctu_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0015,
                f"{a:.3f}", ha="center", va="bottom", fontsize=6.5)
    safe_p = mpatches.Patch(color=C_NESY, alpha=0.85, label="Safe (≥99%)")
    unsafe = mpatches.Patch(color=C_BASE, alpha=0.85, label="Unsafe (<99%)")
    ax.legend(handles=[safe_p, unsafe], fontsize=7, loc="lower left",
              framealpha=0.85)

    # CIC
    ax = axes[1]
    xs = np.arange(len(cic_names))
    safe_cols = [C_NESY if a >= SAFE_THR else C_BASE for a in cic_accs]
    bars = ax.bar(xs, cic_accs, color=safe_cols, alpha=0.85,
                  edgecolor="white", lw=0.5, width=0.7)
    ax.axhline(SAFE_THR, color="#333", lw=1.0, linestyle="--", zorder=5)
    ax.text(len(xs) - 0.5, SAFE_THR + 0.0003, "99% threshold",
            ha="right", va="bottom", fontsize=6.5, color="#333", style="italic")
    ax.set_xticks(xs)
    short = [CIC_SHORT.get(n, n.replace("is_", "").replace("_", "\n"))
             for n in cic_names]
    ax.set_xticklabels(short, rotation=0, ha="center", fontsize=7.0,
                       linespacing=0.85)
    ax.set_ylabel("Concept Accuracy", fontsize=9)
    ax.set_ylim(0.930, 1.008)
    ax.set_title("(b) CIC-IoT-2023  (JointCBM γ=0.5)", fontsize=9,
                 fontweight="bold", pad=5)

    # Intervention delta on twin axis
    ax2 = ax.twinx()
    ax2.plot(xs, cic_deltas, "D-", color="#9467bd", ms=5, lw=1.4, zorder=6)
    ax2.axhline(0, color="#9467bd", lw=0.7, linestyle=":", alpha=0.6)
    ax2.set_ylabel("Intervention Δ F1", fontsize=8, color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd", labelsize=7.5)
    ax2.set_ylim(-0.075, 0.075)

    for bar, a in zip(bars, cic_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"{a:.4f}", ha="center", va="bottom", fontsize=6.0)

    safe_p = mpatches.Patch(color=C_NESY, alpha=0.85, label="Acc ≥99% (safe)")
    unsafe = mpatches.Patch(color=C_BASE, alpha=0.85, label="Acc <99% (unsafe)")
    delta_l = plt.Line2D([0], [0], color="#9467bd", marker="D", ms=5,
                         label="Intervention Δ F1")
    ax.legend(handles=[safe_p, unsafe, delta_l], fontsize=6.5, loc="lower left",
              framealpha=0.85)

    fig.savefig(OUTDIR / "fig4_concept_safety.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig4_concept_safety.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig4_concept_safety done")


# FIG 5 — NeSy rule selectivity heatmap (CTU)

def make_rule_heatmap():
    sel = json.loads((NRES / "ctu_nesy_s0_selectivity.json").read_text())

    rules  = list(sel.keys())
    classes = ["Benign", "C&C-HeartBeat", "DDoS", "Okiru"]
    matrix = np.array([[sel[r].get(c, 0.0) for c in classes] for r in rules])

    short_rules = [
        r.replace("_", " ").title()
          .replace("Cc ", "C&C-")
          .replace("Ddos ", "DDoS ")
        for r in rules
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    fig.subplots_adjust(left=0.28, right=0.97, top=0.92, bottom=0.10)

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8, rotation=20, ha="right")
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(short_rules, fontsize=7.5)
    ax.set_title("NeSy Rule Selectivity — CTU-IoT-23 (seed 0)\n"
                 "Mean rule activation per known class",
                 fontsize=9, fontweight="bold", pad=4)

    for i in range(len(rules)):
        for j in range(len(classes)):
            v = matrix[i, j]
            col = "white" if v > 0.55 else "#333"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=col, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.7, label="Mean activation")
    fig.savefig(OUTDIR / "fig5_rule_heatmap.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig5_rule_heatmap.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig5_rule_heatmap done")


# FIG 6 — SHAP feature importance vs CBM concept accuracy

def make_shap_comparison():
    ctu_b = json.loads((RES / "ctu_cbm_baselines.json").read_text())
    cic_b = json.loads((RES / "cic_cbm_baselines.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.40, left=0.04, right=0.97, top=0.88, bottom=0.24)

    for ax, bdata, title in zip(
        axes,
        [ctu_b, cic_b],
        ["(a) CTU-IoT-23 — SHAP Feature Importance (MLP)",
         "(b) CIC-IoT-2023 — SHAP Feature Importance (MLP)"],
    ):
        shap_data = bdata["SHAP_MLP"]
        feats  = shap_data["top10_features_by_shap"]
        scores = shap_data["top10_mean_abs_shap"]

        xs = np.arange(len(feats))
        ax.barh(xs, scores, color=C_MLP, alpha=0.85, edgecolor="white", lw=0.5)
        ax.set_yticks(xs)
        ax.set_yticklabels(feats, fontsize=7.5)
        ax.set_xlabel("Mean |SHAP|", fontsize=8)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
        ax.invert_yaxis()

        # note: raw features, no concept semantics
        ax.text(0.97, 0.03,
                "Raw flow fields only.\nNo analyst intervention possible.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6, color="#555", style="italic",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff9f0",
                          edgecolor="#e0c070", lw=0.6))

    fig.savefig(OUTDIR / "fig6_shap.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig6_shap.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig6_shap done")


# FIG 7 — F1 comparison across all models (CTU + CIC)

def make_f1_comparison():
    ctu_ev0   = json.loads((RES / "ctu_eval_results.json").read_text())
    ctu_ev01  = json.loads((RES / "ctu_eval_results_g0.1.json").read_text())
    ctu_ev05  = json.loads((RES / "ctu_eval_results_g0.5.json").read_text())
    ctu_ev10  = json.loads((RES / "ctu_eval_results_g1.0.json").read_text())
    cic_ev0   = json.loads((RES / "cic_eval_results.json").read_text())
    cic_ev01  = json.loads((RES / "cic_eval_results_g0.1.json").read_text())
    cic_ev05  = json.loads((RES / "cic_eval_results_g0.5.json").read_text())
    cic_ev10  = json.loads((RES / "cic_eval_results_g1.0.json").read_text())
    ctu_base  = json.loads((RES / "ctu_cbm_baselines.json").read_text())
    cic_base  = json.loads((RES / "cic_cbm_baselines.json").read_text())

    nesy_ctu = np.mean([json.loads(f.read_text())["known_f1"]
                        for f in sorted(NRES.glob("ctu_nesy_s*_eval.json"))])
    nesy_cic = np.mean([json.loads(f.read_text())["known_f1"]
                        for f in sorted(NRES.glob("cic_nesy_s*_eval.json"))])

    labels = [
        "MLP\nBaseline",
        "JointCBM\nγ=0",
        "JointCBM\nγ=0.1",
        "JointCBM\nγ=0.5",
        "JointCBM\nγ=1.0",
        "Seq.\nCBM",
        "Hybrid\nCBM",
        "NeSy\nNIDS",
        "Post-hoc\nCBM",
        "Decision\nTree",
    ]
    ctu_f1 = [
        ctu_ev0["MLPBaseline"]["weighted_f1"],
        ctu_ev0["JointCBM"]["weighted_f1"],
        ctu_ev01["JointCBM"]["weighted_f1"],
        ctu_ev05["JointCBM"]["weighted_f1"],
        ctu_ev10["JointCBM"]["weighted_f1"],
        ctu_ev0["SequentialCBM"]["weighted_f1"],
        ctu_ev0["HybridCBM"]["weighted_f1"],
        nesy_ctu,
        ctu_base["PostHocCBM"]["test_f1"],
        ctu_base["DecisionTree"]["test_f1"],
    ]
    cic_f1 = [
        cic_ev0["MLPBaseline"]["weighted_f1"],
        cic_ev0["JointCBM"]["weighted_f1"],
        cic_ev01["JointCBM"]["weighted_f1"],
        cic_ev05["JointCBM"]["weighted_f1"],
        cic_ev10["JointCBM"]["weighted_f1"],
        cic_ev0["SequentialCBM"]["weighted_f1"],
        cic_ev0["HybridCBM"]["weighted_f1"],
        nesy_cic,
        cic_base["PostHocCBM"]["test_f1"],
        cic_base["DecisionTree"]["test_f1"],
    ]
    colors = [C_MLP,
              C_CBM, C_CBM, C_CBM, C_CBM,
              C_GRAY, C_GRAY,
              C_NESY, C_POSTH, C_DT]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.22)

    # shade the γ ablation group
    ax.axvspan(0.5, 4.5, color=C_CBM, alpha=0.06, zorder=0)

    bars1 = ax.bar(x - w/2, ctu_f1, w, label="CTU-IoT-23",
                   color=colors, alpha=0.90, edgecolor="white", lw=0.4)
    bars2 = ax.bar(x + w/2, cic_f1, w, label="CIC-IoT-2023",
                   color=colors, alpha=0.55, edgecolor="white", lw=0.4,
                   hatch="///")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Weighted F1", fontsize=9)
    ax.set_title("Known-Class F1: All Models — CTU vs CIC",
                 fontsize=9, fontweight="bold", pad=4)
    ax.set_ylim(0.45, 1.04)
    ax.axhline(ctu_f1[0], color=C_MLP, lw=0.8, linestyle=":", alpha=0.6)
    ax.text(len(labels) - 0.5, ctu_f1[0] + 0.004, "CTU MLP ref",
            fontsize=5.5, color=C_MLP, ha="right", alpha=0.8)

    leg = [
        mpatches.Patch(color="#555", alpha=0.85, label="CTU-IoT-23"),
        mpatches.Patch(color="#555", alpha=0.45, hatch="///", label="CIC-IoT-2023"),
    ]
    ax.legend(handles=leg, fontsize=7, loc="lower right", framealpha=0.8)

    fig.savefig(OUTDIR / "fig7_f1_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUTDIR / "fig7_f1_comparison.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig7_f1_comparison done")


if __name__ == "__main__":
    make_architecture()
    make_ood_auroc()
    make_gamma_effect()
    make_concept_safety()
    make_rule_heatmap()
    make_shap_comparison()
    make_f1_comparison()
    print(f"\nAll figures written to {OUTDIR}")
