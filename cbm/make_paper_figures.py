"""
Generate all figures used in the paper (journal + conference versions).

Figures produced:
  Architecture12       - dual-panel architecture diagram
  auroc_vs_f1          - AUROC vs F1 scatter across all models
  fig6_shap            - SHAP feature importance (MLP baseline)
  fig7_f1_comparison   - known-class F1 across all models
  rule_selectivity1_ctu/cic  - rule class-selectivity heatmaps
  threshold_drift_ctu/cic    - learned vs initial threshold drift
  fig4_concept_safety  - per-concept accuracy + intervention deltas
  cic_concept_space_pca      - PCA of CIC concept space

Run from repo root:
  python -m cbm.make_paper_figures
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CBM_RES  = PROJECT_ROOT / "results" / "cbm"
NESY_RES = PROJECT_ROOT / "results" / "nesy"

# All output directories — figures go to every one
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

C_MLP   = "#4e79a7"
C_CBM   = "#f28e2b"
C_NESY  = "#59a14f"
C_BASE  = "#e15759"
C_POSTH = "#76b7b2"
C_DT    = "#edc948"
C_RF    = "#b07aa1"
C_GRAY  = "#bab0ac"


def _save(fig, stem):
    """Save figure as PDF + PNG to every output directory."""
    for d in OUTDIRS:
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {stem} done")


# ── Architecture12 ────────────────────────────────────────────────────────────

def make_architecture():
    def _box(ax, x, y, w, h, label, color, fs=7.5):
        ax.add_patch(FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=1.0, zorder=3))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white", zorder=4)

    def _arr(ax, x0, y0, x1, y1, color="#555", lw=1.0, conn=None):
        kw = dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=8)
        if conn:
            kw["connectionstyle"] = conn
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=kw, zorder=2)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
    fig.subplots_adjust(wspace=0.12, left=0.01, right=0.99, top=0.91, bottom=0.08)

    # ── left: CBM ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(a) Concept Bottleneck Models", fontsize=10,
                 fontweight="bold", pad=6)

    ax.text(0.03, 0.60, "x",    ha="left", va="center", fontsize=10,
            color="#333", fontstyle="italic")
    ax.text(0.03, 0.52, "ℝ³⁹", ha="left", va="center", fontsize=7,
            color="#666")

    _box(ax, 0.24, 0.56, 0.22, 0.15, "MLP Encoder\n256→256→64", C_MLP)
    _arr(ax, 0.13, 0.56, 0.13 + 0.01, 0.56, lw=1.4)

    _box(ax, 0.56, 0.56, 0.20, 0.15, "K=8 Concept\nHeads  σ(·)", C_CBM)
    _arr(ax, 0.35, 0.56, 0.46, 0.56, lw=1.4)

    ax.text(0.56, 0.74, "γ · BCE", ha="center", va="bottom", fontsize=6.5,
            color="#d62728", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#fff0f0",
                      edgecolor="#d62728", lw=0.6))
    _arr(ax, 0.56, 0.74, 0.56, 0.64, color="#d62728", lw=0.8)

    _box(ax, 0.83, 0.56, 0.17, 0.13, "Classifier\n8→C", C_NESY)
    _arr(ax, 0.66, 0.56, 0.745, 0.56, lw=1.4)
    ax.text(0.975, 0.56, "ŷ", ha="center", va="center", fontsize=9,
            color="#333", fontweight="bold")
    _arr(ax, 0.915, 0.56, 0.965, 0.56, lw=1.0)

    # HybridCBM skip
    _arr(ax, 0.35, 0.60, 0.745, 0.60, color="#76b7b2", lw=1.0,
         conn="arc3,rad=-0.30")
    ax.text(0.54, 0.78, "skip (HybridCBM)", fontsize=6, color="#76b7b2",
            ha="center", style="italic")

    # Intervention callout
    ax.annotate("Expert\nintervention", xy=(0.56, 0.485),
                xytext=(0.56, 0.30), fontsize=6.5, color="#9467bd",
                ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#9467bd", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f3eeff",
                          edgecolor="#9467bd", lw=0.7))

    # OOD note
    ax.text(0.56, 0.15, "OOD: Mahalanobis in 8-dim concept space\n"
            "Test-time intervention on ĉ",
            ha="center", va="center", fontsize=6.2, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8",
                      edgecolor="#ccc", lw=0.7))

    # Legend
    for i, (lbl, col, desc) in enumerate([
        ("JointCBM",      C_CBM,    "encoder + concept + classifier end-to-end"),
        ("SequentialCBM", C_BASE,   "concept heads frozen; classifier trained after"),
        ("HybridCBM",     "#76b7b2","encoder skip appended to concept predictions"),
        ("Post-hoc CBM",  C_GRAY,   "frozen encoder + linear concept probes"),
    ]):
        yi = 0.03 + (3 - i) * 0.058
        ax.add_patch(plt.Rectangle((0.02, yi - 0.013), 0.022, 0.024,
                                   facecolor=col, edgecolor="none", zorder=3))
        ax.text(0.048, yi, lbl, fontsize=6.3, va="center",
                color="#222", fontweight="bold")
        ax.text(0.220, yi, desc, fontsize=5.8, va="center",
                color="#555", style="italic")

    # ── right: NeSy-NIDS ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(b) NeSy-NIDS", fontsize=10, fontweight="bold", pad=6)

    ax.text(0.03, 0.68, "x",      ha="left", va="center", fontsize=10,
            color="#333", fontstyle="italic")
    ax.text(0.03, 0.60, "ℝ³⁷/³⁹", ha="left", va="center", fontsize=7,
            color="#666")

    _box(ax, 0.43, 0.78, 0.40, 0.17,
         "R Differentiable Rules\n"
         r"$\prod_k\sigma(\beta(x_{f_k}-\theta_{rk})d_{rk})$",
         C_NESY, fs=7.2)
    _arr(ax, 0.12, 0.72, 0.23, 0.78, lw=1.2)

    ax.text(0.43, 0.90,
            "β: 1 → k_max over T_warm epochs (k-annealing)",
            ha="center", va="bottom", fontsize=6.0, color="#2ca02c",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#f0fff0",
                      edgecolor="#2ca02c", lw=0.5))

    _box(ax, 0.43, 0.46, 0.32, 0.14,
         "Residual MLP\n(neural fallback)", C_MLP, fs=7.2)
    _arr(ax, 0.12, 0.62, 0.27, 0.46, lw=1.2)

    _box(ax, 0.83, 0.62, 0.20, 0.28,
         "α-Gate\nα·rules\n+(1−α)·MLP", C_BASE, fs=6.8)
    _arr(ax, 0.63, 0.78, 0.73, 0.68, lw=1.2)
    _arr(ax, 0.59, 0.46, 0.73, 0.58, lw=1.2)

    ax.text(0.83, 0.88,
            "λ_α·(1−α)²\npenalises\nneural reliance",
            ha="center", va="bottom", fontsize=5.8, color="#d62728",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#fff0f0",
                      edgecolor="#d62728", lw=0.5))

    ax.text(0.975, 0.62, "ŷ", ha="center", va="center", fontsize=9,
            color="#333", fontweight="bold")
    _arr(ax, 0.93, 0.62, 0.965, 0.62, lw=1.0)

    ax.annotate("Mahalanobis OOD\nin R-dim rule space",
                xy=(0.43, 0.695), xytext=(0.20, 0.28),
                fontsize=6.2, color="#9467bd", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#9467bd", lw=0.8,
                                linestyle="dashed"),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f3eeff",
                          edgecolor="#9467bd", lw=0.6))

    ax.text(0.50, 0.04,
            "Thresholds θ_rk learned jointly  |  "
            "STE binarisation: 100% crisp rule activations at k=k_max",
            ha="center", va="center", fontsize=6.0, color="#444",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8",
                      edgecolor="#ccc", lw=0.7))

    _save(fig, "Architecture12")


# ── auroc_vs_f1 ───────────────────────────────────────────────────────────────

def make_auroc_vs_f1():
    def _gather(ds):
        pts = []
        for gfile in sorted(CBM_RES.glob(f"{ds}_eval_results*.json")):
            data = json.loads(gfile.read_text())
            for mname, v in data.items():
                if isinstance(v, dict) and v.get("ood_auroc") is not None:
                    pts.append((mname, v["weighted_f1"], v["ood_auroc"]))
        base = json.loads((CBM_RES / f"{ds}_cbm_baselines.json").read_text())
        for mname, v in base.items():
            f1 = v.get("test_f1") or v.get("weighted_f1") if isinstance(v, dict) else None
            auroc = v.get("ood_auroc") if isinstance(v, dict) else None
            if f1 is not None and auroc is not None:
                pts.append((mname, f1, auroc))
        for sf in sorted(NESY_RES.glob(f"{ds}_nesy_s*_eval.json")):
            d = json.loads(sf.read_text())
            pts.append((f"NeSy s{d['seed']}", d["known_f1"], d["ood_auroc"]))
        nb = json.loads((NESY_RES / f"{ds}_baselines.json").read_text())
        for mname, v in nb.items():
            f1 = v.get("f1")
            auroc = v.get("auroc_maha") or v.get("auroc")
            if f1 and auroc:
                pts.append((mname, f1, auroc))
        return pts

    def _col(label):
        l = label.lower()
        if "nesy" in l:         return C_NESY
        if "joint" in l:        return C_CBM
        if "sequential" in l:   return C_GRAY
        if "hybrid" in l:       return C_POSTH
        if "mlp" in l:          return C_MLP
        if "randomforest" in l: return C_RF
        if "decision" in l:     return C_DT
        if "posthoc" in l or "post" in l: return "#76b7b2"
        return "#999999"

    def _mk(label):
        l = label.lower()
        if "nesy" in l:       return "D"
        if "joint" in l:      return "o"
        if "sequential" in l: return "s"
        if "hybrid" in l:     return "^"
        if "mlp" in l:        return "P"
        return "X"

    ANNOTATE = {"MLPBaseline", "NeSy s0", "SequentialCBM", "DecisionTree",
                "RandomForest", "PostHocCBM"}
    SHORT = {
        "MLPBaseline": "MLP", "SequentialCBM": "SeqCBM",
        "NeSy s0": "NeSy(s0)", "DecisionTree": "DT",
        "RandomForest": "RF",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    fig.subplots_adjust(wspace=0.30, left=0.07, right=0.98,
                        top=0.88, bottom=0.18)

    for ax, ds, title in zip(axes, ["ctu", "cic"],
                              ["(a) CTU-IoT-23", "(b) CIC-IoT-2023"]):
        for label, f1, auroc in _gather(ds):
            ax.scatter(f1, auroc, c=_col(label), marker=_mk(label),
                       s=55, alpha=0.88, edgecolors="white",
                       linewidths=0.5, zorder=4)
            if label in ANNOTATE:
                ax.annotate(SHORT.get(label, label), (f1, auroc),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=6.2, color="#333")

        ax.axhline(0.5, color="#ddd", lw=0.8, linestyle=":", zorder=0)
        ax.set_xlabel("Weighted F1 (known classes)", fontsize=9)
        ax.set_ylabel("OOD AUROC (Mahalanobis)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.20, 1.05)

    legend_entries = [
        mpatches.Patch(color=C_MLP,    label="MLP Baseline"),
        mpatches.Patch(color=C_CBM,    label="JointCBM (all γ)"),
        mpatches.Patch(color=C_GRAY,   label="SequentialCBM"),
        mpatches.Patch(color=C_POSTH,  label="HybridCBM / PostHocCBM"),
        mpatches.Patch(color=C_NESY,   label="NeSy-NIDS"),
        mpatches.Patch(color=C_DT,     label="Decision Tree"),
        mpatches.Patch(color=C_RF,     label="Random Forest"),
    ]
    fig.legend(handles=legend_entries, fontsize=7, loc="lower center",
               ncol=4, framealpha=0.85, bbox_to_anchor=(0.5, -0.01))

    _save(fig, "auroc_vs_f1")


# ── fig6_shap ─────────────────────────────────────────────────────────────────

def make_shap():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.40, left=0.04, right=0.97,
                        top=0.88, bottom=0.24)

    for ax, ds, title in zip(
        axes,
        ["ctu", "cic"],
        ["(a) CTU-IoT-23 — SHAP Feature Importance (MLP)",
         "(b) CIC-IoT-2023 — SHAP Feature Importance (MLP)"],
    ):
        bdata = json.loads((CBM_RES / f"{ds}_cbm_baselines.json").read_text())
        shap  = bdata["SHAP_MLP"]
        feats  = shap["top10_features_by_shap"]
        scores = shap["top10_mean_abs_shap"]

        xs = np.arange(len(feats))
        ax.barh(xs, scores, color=C_MLP, alpha=0.85,
                edgecolor="white", lw=0.5)
        ax.set_yticks(xs)
        ax.set_yticklabels(feats, fontsize=7.5)
        ax.set_xlabel("Mean |SHAP|", fontsize=8)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
        ax.invert_yaxis()

        ax.text(0.97, 0.03,
                "Raw flow fields only.\nNo analyst intervention possible.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6, color="#555", style="italic",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff9f0",
                          edgecolor="#e0c070", lw=0.6))

    _save(fig, "fig6_shap")


# ── fig7_f1_comparison ───────────────────────────────────────────────────────

def make_f1_comparison():
    def _get(fname, model):
        path = CBM_RES / fname
        if not path.exists():
            return float("nan")
        return json.loads(path.read_text()).get(model, {}).get("weighted_f1", float("nan"))

    nesy_ctu = np.mean([json.loads(f.read_text())["known_f1"]
                        for f in sorted(NESY_RES.glob("ctu_nesy_s*_eval.json"))])
    nesy_cic_files = sorted(NESY_RES.glob("cic_nesy_s*_eval.json"))
    nesy_cic = np.mean([json.loads(f.read_text())["known_f1"]
                        for f in nesy_cic_files]) if nesy_cic_files else float("nan")

    ctu_b = json.loads((CBM_RES / "ctu_cbm_baselines.json").read_text())
    cic_b = json.loads((CBM_RES / "cic_cbm_baselines.json").read_text())

    labels = ["MLP\nBaseline", "Joint\nγ=0", "Joint\nγ=0.1",
              "Joint\nγ=0.5", "Joint\nγ=1.0", "Seq.\nCBM",
              "Hybrid\nCBM", "NeSy\nNIDS", "Post-hoc\nCBM", "Decision\nTree"]

    ctu_f1 = [
        _get("ctu_eval_results.json",     "MLPBaseline"),
        _get("ctu_eval_results.json",     "JointCBM"),
        _get("ctu_eval_results_g0.1.json","JointCBM"),
        _get("ctu_eval_results_g0.5.json","JointCBM"),
        _get("ctu_eval_results_g1.0.json","JointCBM"),
        _get("ctu_eval_results.json",     "SequentialCBM"),
        _get("ctu_eval_results.json",     "HybridCBM"),
        nesy_ctu,
        ctu_b["PostHocCBM"]["test_f1"],
        ctu_b["DecisionTree"]["test_f1"],
    ]
    cic_f1 = [
        _get("cic_eval_results.json",     "MLPBaseline"),
        _get("cic_eval_results.json",     "JointCBM"),
        _get("cic_eval_results_g0.1.json","JointCBM"),
        _get("cic_eval_results_g0.5.json","JointCBM"),
        _get("cic_eval_results_g1.0.json","JointCBM"),
        _get("cic_eval_results.json",     "SequentialCBM"),
        _get("cic_eval_results.json",     "HybridCBM"),
        nesy_cic,
        cic_b["PostHocCBM"]["test_f1"],
        cic_b["DecisionTree"]["test_f1"],
    ]
    colors = [C_MLP, C_CBM, C_CBM, C_CBM, C_CBM,
              C_GRAY, C_GRAY, C_NESY, C_POSTH, C_DT]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.22)

    ax.axvspan(0.5, 4.5, color=C_CBM, alpha=0.06, zorder=0)
    ax.bar(x - w/2, ctu_f1, w, color=colors, alpha=0.90,
           edgecolor="white", lw=0.4, label="CTU-IoT-23")
    ax.bar(x + w/2, cic_f1, w, color=colors, alpha=0.55,
           edgecolor="white", lw=0.4, hatch="///", label="CIC-IoT-2023")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Weighted F1", fontsize=9)
    ax.set_title("Known-Class F1: All Models — CTU vs CIC",
                 fontsize=9, fontweight="bold", pad=4)
    ax.set_ylim(0.45, 1.04)
    ref = ctu_f1[0]
    ax.axhline(ref, color=C_MLP, lw=0.8, linestyle=":", alpha=0.6)
    ax.text(len(labels) - 0.5, ref + 0.004, "CTU MLP ref",
            fontsize=5.5, color=C_MLP, ha="right", alpha=0.8)

    leg = [
        mpatches.Patch(color="#555", alpha=0.85, label="CTU-IoT-23"),
        mpatches.Patch(color="#555", alpha=0.45, hatch="///",
                       label="CIC-IoT-2023"),
    ]
    ax.legend(handles=leg, fontsize=7, loc="lower right", framealpha=0.8)

    _save(fig, "fig7_f1_comparison")


# ── rule_selectivity1_ctu / rule_selectivity1_cic ────────────────────────────

def make_rule_selectivity():
    for ds, title in [("ctu", "CTU-IoT-23"), ("cic", "CIC-IoT-2023")]:
        sel = json.loads(
            (NESY_RES / f"{ds}_nesy_s0_selectivity.json").read_text())

        rules   = list(sel.keys())
        classes = list(list(sel.values())[0].keys())
        matrix  = np.array([[sel[r].get(c, 0.0) for c in classes]
                             for r in rules])

        short_rules = [
            r.replace("_", " ").title()
             .replace("Cc ", "C&C-")
             .replace("Ddos ", "DDoS ")
             .replace("Udp ", "UDP ")
            for r in rules]
        short_cls = [
            c.replace("_Final", "")
             .replace("Mirai-greeth_flood", "Mirai-Flood")
             .replace("DDoS-SYN_Flood", "DDoS-SYN")
             .replace("Recon-PortScan", "PortScan")
             .replace("VulnerabilityScan", "VulnScan")
            for c in classes]

        fig_h = max(3.8, 0.5 * len(rules))
        fig_w = max(4.5, 1.1 * len(classes))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(left=0.30, right=0.95, top=0.90, bottom=0.14)

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(short_cls, fontsize=8, rotation=22, ha="right")
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels(short_rules, fontsize=7.5)
        ax.set_title(
            f"Rule Class Selectivity — {title}  (seed 0, k=10)\n"
            "Mean rule activation per known class after STE binarisation",
            fontsize=9, fontweight="bold", pad=4)

        for i in range(len(rules)):
            for j in range(len(classes)):
                v = matrix[i, j]
                col = "white" if v > 0.55 else "#333"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=col, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.75, label="Mean activation")
        _save(fig, f"rule_selectivity1_{ds}")


# ── threshold_drift_ctu / threshold_drift_cic ────────────────────────────────

def make_threshold_drift():
    from nesy.model import NeSyNIDS, CTU_RULES, CIC_RULES

    for ds, rule_templates, title in [
        ("ctu", CTU_RULES,  "CTU-IoT-23"),
        ("cic", CIC_RULES,  "CIC-IoT-2023"),
    ]:
        ckpt = torch.load(NESY_RES / f"{ds}_nesy_s0.pt",
                          map_location="cpu", weights_only=False)
        model = NeSyNIDS(ckpt["n_features"], ckpt["n_classes"], rule_templates)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        labels, drifts, colors = [], [], []
        for rule, tmpl in zip(model.rule_bank.rules, rule_templates):
            learned = rule.thresholds.detach().tolist()
            for j, (l, i, ct) in enumerate(
                    zip(learned, tmpl.init_thresholds, tmpl.condition_types)):
                drift = l - i
                sym = ">" if ct == "gt" else "<"
                labels.append(f"{rule.name}\n[cond {j+1}: {sym}]")
                drifts.append(drift)
                colors.append("#d62728" if drift > 0 else C_MLP)

        fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.42 * len(labels))))
        fig.subplots_adjust(left=0.38, right=0.94, top=0.92, bottom=0.08)

        xs   = np.arange(len(labels))
        bars = ax.barh(xs, drifts, color=colors, alpha=0.85,
                       edgecolor="white", lw=0.5)
        ax.axvline(0, color="#333", lw=0.9, linestyle="--", zorder=5)

        for bar, d in zip(bars, drifts):
            offset = 0.005 if d >= 0 else -0.005
            ha = "left" if d >= 0 else "right"
            ax.text(d + offset, bar.get_y() + bar.get_height() / 2,
                    f"{d:+.3f}", ha=ha, va="center",
                    fontsize=7.0, color="#222")

        ax.set_yticks(xs)
        ax.set_yticklabels(labels, fontsize=7.0, linespacing=0.88)
        ax.invert_yaxis()
        ax.set_xlabel("Learned − Initial threshold", fontsize=9)
        ax.set_title(
            f"Threshold Drift — {title}  (seed 0)\n"
            "Red = increased (tighter);  Blue = decreased (relaxed)",
            fontsize=9, fontweight="bold", pad=5)

        ax.legend(handles=[
            mpatches.Patch(color="#d62728", alpha=0.85, label="Increased (>0)"),
            mpatches.Patch(color=C_MLP,    alpha=0.85, label="Decreased (<0)"),
        ], fontsize=7, loc="lower right", framealpha=0.85)

        _save(fig, f"threshold_drift_{ds}")


# ── fig4_concept_safety ───────────────────────────────────────────────────────

def make_concept_safety():
    SAFE_THR = 0.99
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
        "is_persistent":       "persist.",
        "is_large_payload":    "large\npayload",
        "is_syn_flood":        "syn\nflood",
        "is_high_variance":    "high\nvar.",
        "is_high_rate":        "high\nrate",
        "is_udp_dominant":     "udp\ndom.",
        "is_port_scan":        "port\nscan",
        "is_short_connection": "short\nconn.",
    }

    ctu = json.loads(
        (CBM_RES / "ctu_eval_results_g0.5.json").read_text())["JointCBM"]
    cic = json.loads(
        (CBM_RES / "cic_eval_results_g0.5.json").read_text())["JointCBM"]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    fig.subplots_adjust(wspace=0.45, left=0.07, right=0.95,
                        top=0.88, bottom=0.12)

    for ax, data, short_map, ds_title in [
        (axes[0], ctu, CTU_SHORT, "CTU-IoT-23  (JointCBM γ=0.5)"),
        (axes[1], cic, CIC_SHORT, "CIC-IoT-2023  (JointCBM γ=0.5)"),
    ]:
        names  = list(data["concept_accuracies"].keys())
        accs   = list(data["concept_accuracies"].values())
        order  = np.argsort(accs)[::-1]
        names  = [names[i]  for i in order]
        accs   = [accs[i]   for i in order]

        xs    = np.arange(len(names))
        scols = [C_NESY if a >= SAFE_THR else C_BASE for a in accs]
        bars  = ax.bar(xs, accs, color=scols, alpha=0.85,
                       edgecolor="white", lw=0.5, width=0.7)
        ax.axhline(SAFE_THR, color="#333", lw=1.0, linestyle="--", zorder=5)
        ax.text(len(xs) - 0.5, SAFE_THR + 0.003, "99% threshold",
                ha="right", va="bottom", fontsize=6.5,
                color="#333", style="italic")

        short = [short_map.get(n, n.replace("is_", "").replace("_", "\n"))
                 for n in names]
        ax.set_xticks(xs)
        ax.set_xticklabels(short, rotation=0, ha="center",
                           fontsize=7.0, linespacing=0.85)
        ax.set_ylabel("Concept Accuracy", fontsize=9)
        ax.set_title(f"({'ab'[axes.tolist().index(ax)]}) {ds_title}",
                     fontsize=9, fontweight="bold", pad=5)

        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0015,
                    f"{a:.3f}", ha="center", va="bottom", fontsize=6.5)

        # Intervention delta on twin axis (CIC only has deltas in eval)
        if "intervention_deltas" in data and data["intervention_deltas"]:
            deltas = [data["intervention_deltas"].get(names[i], 0.0)
                      for i in range(len(names))]
            ax2 = ax.twinx()
            ax2.plot(xs, deltas, "D-", color="#9467bd",
                     ms=5, lw=1.4, zorder=6)
            ax2.axhline(0, color="#9467bd", lw=0.7,
                        linestyle=":", alpha=0.6)
            ax2.set_ylabel("Intervention Δ F1",
                           fontsize=8, color="#9467bd")
            ax2.tick_params(axis="y", labelcolor="#9467bd",
                            labelsize=7.5)
            ax2.set_ylim(-0.075, 0.075)

        ax.legend(handles=[
            mpatches.Patch(color=C_NESY,  alpha=0.85, label="Safe (≥99%)"),
            mpatches.Patch(color=C_BASE,  alpha=0.85, label="Unsafe (<99%)"),
        ], fontsize=7, loc="lower left", framealpha=0.85)

    _save(fig, "fig4_concept_safety")


# ── cic_concept_space_pca ─────────────────────────────────────────────────────

def make_concept_space_pca():
    import pandas as pd
    from sklearn.decomposition import PCA
    from cbm.model import JointCBM

    data_dir = PROJECT_ROOT / "data" / "cic"
    vocab = json.loads((data_dir / "vocab.json").read_text())
    feature_cols = vocab["feature_cols"]
    l2i = vocab["label_to_id"]
    known_classes = vocab["known_classes"]
    known_ids = [l2i[c] for c in known_classes]
    id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}

    sid_to_name = {}
    for cls_name in known_classes:
        short = (cls_name.replace("_Final", "")
                         .replace("Mirai-greeth_flood", "Mirai-Flood")
                         .replace("DDoS-SYN_Flood", "DDoS-SYN")
                         .replace("Recon-PortScan", "PortScan")
                         .replace("VulnerabilityScan", "VulnScan"))
        sid_to_name[id_map[l2i[cls_name]]] = short

    df = pd.read_parquet(data_dir / "test_known.parquet")
    mask = df["label_id"].isin(known_ids)
    df = df[mask].reset_index(drop=True)
    if len(df) > 3000:
        df = df.sample(n=3000, random_state=42).reset_index(drop=True)

    X = df[feature_cols].values.astype("float32")
    y = np.array([id_map[lid] for lid in df["label_id"].values],
                 dtype=np.int64)

    ckpt = torch.load(CBM_RES / "cic_JointCBM_g0.5.pt",
                      map_location="cpu", weights_only=False)
    model = JointCBM(ckpt["n_features"], ckpt["n_concepts"],
                     ckpt["n_classes"], ckpt["embed_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        concept_acts = model.get_embedding(
            torch.tensor(X, dtype=torch.float32)).numpy()

    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(concept_acts)
    var = pca.explained_variance_ratio_

    palette = [C_MLP, C_CBM, C_NESY, C_BASE, "#76b7b2"]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    fig.subplots_adjust(left=0.10, right=0.85, top=0.90, bottom=0.10)

    for i, cid in enumerate(sorted(sid_to_name.keys())):
        mask_c = (y == cid)
        ax.scatter(Z[mask_c, 0], Z[mask_c, 1],
                   c=palette[i % len(palette)],
                   label=sid_to_name[cid],
                   s=14, alpha=0.55, edgecolors="none")

    ax.set_xlabel(f"PC 1  ({var[0]*100:.1f}% var.)", fontsize=9)
    ax.set_ylabel(f"PC 2  ({var[1]*100:.1f}% var.)", fontsize=9)
    ax.set_title("Concept Space PCA — CIC-IoT-2023\n"
                 "JointCBM (γ=0.5), 8-dim concept activations → 2 PCs",
                 fontsize=9, fontweight="bold", pad=5)
    ax.legend(fontsize=7.5, loc="upper right",
              framealpha=0.85, markerscale=2.0)

    # PNG only — PCA scatter does not need vector PDF
    for d in OUTDIRS:
        fig.savefig(d / "cic_concept_space_pca.png",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  cic_concept_space_pca done")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating all paper figures ...\n")
    make_architecture()
    make_auroc_vs_f1()
    make_shap()
    make_f1_comparison()
    make_rule_selectivity()
    make_threshold_drift()
    make_concept_safety()
    make_concept_space_pca()
    print(f"\nDone. Written to:")
    for d in OUTDIRS:
        print(f"  {d}")
