"""
Generate the four figures not produced by make_figures.py:
  - threshold_drift_ctu.pdf / threshold_drift_cic.pdf
  - rule_selectivity1_ctu.pdf / rule_selectivity1_cic.pdf
  - auroc_vs_f1.pdf
  - cic_concept_space_pca.png
  - Architecture12.pdf
Run from the repo root: python -m cbm.make_missing_figures
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTDIR   = PROJECT_ROOT / "paper" / "figures"
CBM_RES  = PROJECT_ROOT / "results" / "cbm"
NESY_RES = PROJECT_ROOT / "results" / "nesy"
OUTDIR.mkdir(parents=True, exist_ok=True)

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

C_MLP   = "#4e79a7"
C_CBM   = "#f28e2b"
C_NESY  = "#59a14f"
C_BASE  = "#e15759"
C_POSTH = "#76b7b2"
C_DT    = "#edc948"
C_RF    = "#b07aa1"
C_GRAY  = "#bab0ac"


# ── FIG: threshold_drift_ctu.pdf / threshold_drift_cic.pdf ───────────────────

def make_threshold_drift():
    from nesy.model import NeSyNIDS, CTU_RULES, CIC_RULES

    configs = [
        ("ctu", CTU_RULES, "CTU-IoT-23", "threshold_drift_ctu"),
        ("cic", CIC_RULES, "CIC-IoT-2023", "threshold_drift_cic"),
    ]

    for ds, rule_templates, ds_title, fname in configs:
        ckpt_path = NESY_RES / f"{ds}_nesy_s0.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = NeSyNIDS(ckpt["n_features"], ckpt["n_classes"], rule_templates)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Build (label, drift) pairs — one bar per rule condition
        labels, drifts, colors = [], [], []
        for rule, tmpl in zip(model.rule_bank.rules, rule_templates):
            learned = rule.thresholds.detach().tolist()
            initial = tmpl.init_thresholds
            for j, (l, i, ct) in enumerate(zip(learned, initial, tmpl.condition_types)):
                drift = l - i
                sym = ">" if ct == "gt" else "<"
                labels.append(f"{rule.name}\n[cond {j+1}: {sym}]")
                drifts.append(drift)
                colors.append("#d62728" if drift > 0 else "#4e79a7")

        fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.42 * len(labels))))
        fig.subplots_adjust(left=0.38, right=0.94, top=0.92, bottom=0.08)

        xs = np.arange(len(labels))
        bars = ax.barh(xs, drifts, color=colors, alpha=0.85, edgecolor="white", lw=0.5)
        ax.axvline(0, color="#333", lw=0.9, linestyle="--", zorder=5)

        for bar, d in zip(bars, drifts):
            offset = 0.005 if d >= 0 else -0.005
            ha = "left" if d >= 0 else "right"
            ax.text(d + offset, bar.get_y() + bar.get_height() / 2,
                    f"{d:+.3f}", ha=ha, va="center", fontsize=7.0, color="#222")

        ax.set_yticks(xs)
        ax.set_yticklabels(labels, fontsize=7.0, linespacing=0.88)
        ax.invert_yaxis()
        ax.set_xlabel("Learned − Initial threshold", fontsize=9)
        ax.set_title(
            f"Threshold Drift — {ds_title}  (seed 0)\n"
            "Red = threshold increased (tighter); Blue = decreased (relaxed)",
            fontsize=9, fontweight="bold", pad=5,
        )

        pos_p = mpatches.Patch(color="#d62728", alpha=0.85, label="Increased (>0)")
        neg_p = mpatches.Patch(color="#4e79a7", alpha=0.85, label="Decreased (<0)")
        ax.legend(handles=[pos_p, neg_p], fontsize=7, loc="lower right", framealpha=0.85)

        for ext in ("pdf", "png"):
            kw = {"bbox_inches": "tight"}
            if ext == "png":
                kw["dpi"] = 200
            fig.savefig(OUTDIR / f"{fname}.{ext}", **kw)
        plt.close(fig)
        print(f"{fname} done")


# ── FIG: rule_selectivity1_ctu.pdf / rule_selectivity1_cic.pdf ───────────────

def make_rule_selectivity():
    configs = [
        ("ctu", "rule_selectivity1_ctu", "CTU-IoT-23"),
        ("cic", "rule_selectivity1_cic", "CIC-IoT-2023"),
    ]

    for ds, fname, ds_title in configs:
        sel = json.loads((NESY_RES / f"{ds}_nesy_s0_selectivity.json").read_text())

        rules   = list(sel.keys())
        classes = list(list(sel.values())[0].keys())
        matrix  = np.array([[sel[r].get(c, 0.0) for c in classes] for r in rules])

        short_rules = [
            r.replace("_", " ").title()
             .replace("Cc ", "C&C-")
             .replace("Ddos ", "DDoS ")
             .replace("Ddos-Syn ", "DDoS-SYN ")
             .replace("Udp ", "UDP ")
        for r in rules]

        short_classes = [c.replace("_Final", "").replace("Mirai-greeth_flood", "Mirai-Flood")
                          .replace("DDoS-SYN_Flood", "DDoS-SYN").replace("Recon-PortScan", "PortScan")
                          .replace("VulnerabilityScan", "VulnScan")
                         for c in classes]

        fig_h = max(3.8, 0.5 * len(rules))
        fig_w = max(4.5, 1.1 * len(classes))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(left=0.30, right=0.95, top=0.90, bottom=0.14)

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(short_classes, fontsize=8, rotation=22, ha="right")
        ax.set_yticks(range(len(rules)))
        ax.set_yticklabels(short_rules, fontsize=7.5)
        ax.set_title(
            f"Rule Class Selectivity — {ds_title}  (seed 0, k=10)\n"
            "Mean rule activation per known class after STE binarisation",
            fontsize=9, fontweight="bold", pad=4,
        )

        for i in range(len(rules)):
            for j in range(len(classes)):
                v = matrix[i, j]
                col = "white" if v > 0.55 else "#333"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=col, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.75, label="Mean activation")

        for ext in ("pdf", "png"):
            kw = {"bbox_inches": "tight"}
            if ext == "png":
                kw["dpi"] = 200
            fig.savefig(OUTDIR / f"{fname}.{ext}", **kw)
        plt.close(fig)
        print(f"{fname} done")


# ── FIG: auroc_vs_f1.pdf ─────────────────────────────────────────────────────

def make_auroc_vs_f1():
    def _load_cbm(ds):
        points = []
        gamma_files = {
            "γ=0":   f"{ds}_eval_results.json",
            "γ=0.1": f"{ds}_eval_results_g0.1.json",
            "γ=0.5": f"{ds}_eval_results_g0.5.json",
            "γ=1.0": f"{ds}_eval_results_g1.0.json",
        }
        for gtag, fname in gamma_files.items():
            path = CBM_RES / fname
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for mname, vals in data.items():
                if not isinstance(vals, dict) or "weighted_f1" not in vals:
                    continue
                if vals.get("ood_auroc") is None:
                    continue
                label = mname if gtag == "γ=0" else f"{mname} {gtag}"
                points.append((label, vals["weighted_f1"], vals["ood_auroc"]))

        base = json.loads((CBM_RES / f"{ds}_cbm_baselines.json").read_text())
        for mname, vals in base.items():
            if not isinstance(vals, dict):
                continue
            f1   = vals.get("test_f1") or vals.get("weighted_f1")
            auroc = vals.get("ood_auroc")
            if f1 is not None and auroc is not None:
                points.append((mname, f1, auroc))
        return points

    def _load_nesy(ds):
        points = []
        seed_files = sorted(NESY_RES.glob(f"{ds}_nesy_s*_eval.json"))
        for sf in seed_files:
            d = json.loads(sf.read_text())
            points.append((f"NeSy-NIDS s{d['seed']}", d["known_f1"], d["ood_auroc"]))
        nb = json.loads((NESY_RES / f"{ds}_baselines.json").read_text())
        for mname, vals in nb.items():
            f1 = vals.get("f1")
            auroc = vals.get("auroc_maha") or vals.get("auroc")
            if f1 and auroc:
                points.append((mname, f1, auroc))
        return points

    def _color(label):
        l = label.lower()
        if "nesy" in l:          return C_NESY
        if "joint" in l:         return C_CBM
        if "sequential" in l:    return C_GRAY
        if "hybrid" in l:        return C_POSTH
        if "mlp" in l:           return C_MLP
        if "randomforest" in l:  return C_RF
        if "decision" in l or "dt" in l: return C_DT
        if "posthoc" in l or "post" in l: return "#76b7b2"
        return "#999999"

    def _marker(label):
        l = label.lower()
        if "nesy" in l: return "D"
        if "joint" in l: return "o"
        if "sequential" in l: return "s"
        if "hybrid" in l: return "^"
        if "mlp" in l: return "P"
        return "X"

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    fig.subplots_adjust(wspace=0.30, left=0.07, right=0.98, top=0.88, bottom=0.10)

    for ax, ds, title in zip(axes, ["ctu", "cic"],
                              ["(a) CTU-IoT-23", "(b) CIC-IoT-2023"]):
        pts = _load_cbm(ds) + _load_nesy(ds)

        for label, f1, auroc in pts:
            col = _color(label)
            mk  = _marker(label)
            ax.scatter(f1, auroc, c=col, marker=mk, s=55, alpha=0.88,
                       edgecolors="white", linewidths=0.5, zorder=4)

        # Annotate a few key points
        key_labels = {"MLPBaseline", "NeSy-NIDS s0", "SequentialCBM",
                      "JointCBM γ=0.5", "DecisionTree", "RandomForest"}
        for label, f1, auroc in pts:
            if label in key_labels or "s0" in label:
                short = (label.replace("MLPBaseline", "MLP")
                              .replace("SequentialCBM", "SeqCBM")
                              .replace("JointCBM ", "Joint ")
                              .replace("RandomForest", "RF")
                              .replace("DecisionTree", "DT")
                              .replace("NeSy-NIDS s0", "NeSy(s0)"))
                ax.annotate(short, (f1, auroc),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=6.2, color="#333")

        ax.axhline(0.5, color="#ddd", lw=0.8, linestyle=":", zorder=0)
        ax.set_xlabel("Weighted F1 (known classes)", fontsize=9)
        ax.set_ylabel("OOD AUROC (Mahalanobis)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.20, 1.05)

    # Shared legend
    legend_entries = [
        mpatches.Patch(color=C_MLP,   label="MLP Baseline"),
        mpatches.Patch(color=C_CBM,   label="JointCBM (all γ)"),
        mpatches.Patch(color=C_GRAY,  label="SequentialCBM"),
        mpatches.Patch(color=C_POSTH, label="HybridCBM / PostHocCBM"),
        mpatches.Patch(color=C_NESY,  label="NeSy-NIDS"),
        mpatches.Patch(color=C_DT,    label="Decision Tree"),
        mpatches.Patch(color=C_RF,    label="Random Forest"),
    ]
    fig.legend(handles=legend_entries, fontsize=7, loc="lower center",
               ncol=4, framealpha=0.85, bbox_to_anchor=(0.5, -0.01))

    for ext in ("pdf", "png"):
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 200
        fig.savefig(OUTDIR / f"auroc_vs_f1.{ext}", **kw)
    plt.close(fig)
    print("auroc_vs_f1 done")


# ── FIG: cic_concept_space_pca.png ───────────────────────────────────────────

def make_concept_space_pca():
    import pandas as pd
    from sklearn.decomposition import PCA
    from cbm.model import JointCBM
    from cbm.concepts import cic_concept_labels

    data_dir = PROJECT_ROOT / "data" / "cic"
    vocab = json.loads((data_dir / "vocab.json").read_text())
    feature_cols = vocab["feature_cols"]
    l2i = vocab["label_to_id"]
    known_classes = vocab["known_classes"]
    known_ids = [l2i[c] for c in known_classes]
    id_map = {orig: i for i, orig in enumerate(sorted(known_ids))}
    sid_to_name = {}
    for cls_name in known_classes:
        orig_id = l2i[cls_name]
        mapped  = id_map[orig_id]
        short = (cls_name.replace("_Final", "")
                         .replace("Mirai-greeth_flood", "Mirai-Flood")
                         .replace("DDoS-SYN_Flood", "DDoS-SYN")
                         .replace("Recon-PortScan", "PortScan")
                         .replace("VulnerabilityScan", "VulnScan"))
        sid_to_name[mapped] = short

    df = pd.read_parquet(data_dir / "test_known.parquet")

    mask = df["label_id"].isin(known_ids)
    df = df[mask].reset_index(drop=True)

    # Subsample for speed
    MAX = 3000
    if len(df) > MAX:
        df = df.sample(n=MAX, random_state=42).reset_index(drop=True)

    X = df[feature_cols].values.astype("float32")
    y = np.array([id_map[lid] for lid in df["label_id"].values], dtype=np.int64)

    # Load JointCBM γ=0.5
    ckpt = torch.load(CBM_RES / "cic_JointCBM_g0.5.pt",
                      map_location="cpu", weights_only=False)
    model = JointCBM(ckpt["n_features"], ckpt["n_concepts"],
                     ckpt["n_classes"], ckpt["embed_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        concept_acts = model.get_embedding(X_t).numpy()  # (N, 8) — concept probabilities

    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(concept_acts)
    var = pca.explained_variance_ratio_

    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
    class_ids = sorted(sid_to_name.keys())

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    fig.subplots_adjust(left=0.10, right=0.85, top=0.90, bottom=0.10)

    for i, cid in enumerate(class_ids):
        mask_c = (y == cid)
        ax.scatter(Z[mask_c, 0], Z[mask_c, 1],
                   c=palette[i % len(palette)], label=sid_to_name[cid],
                   s=14, alpha=0.55, edgecolors="none")

    ax.set_xlabel(f"PC 1  ({var[0]*100:.1f}% var.)", fontsize=9)
    ax.set_ylabel(f"PC 2  ({var[1]*100:.1f}% var.)", fontsize=9)
    ax.set_title("Concept Space PCA — CIC-IoT-2023\n"
                 "JointCBM (γ=0.5), 8-dim concept activations → 2 PCs",
                 fontsize=9, fontweight="bold", pad=5)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85,
              markerscale=2.0)

    fig.savefig(OUTDIR / "cic_concept_space_pca.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("cic_concept_space_pca done")


# ── FIG: Architecture12.pdf ──────────────────────────────────────────────────

def make_architecture12():
    """Refined side-by-side architecture diagram with cleaner layout."""

    def _box(ax, x, y, w, h, label, color, fontsize=7.5, lw=1.0):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor="white",
                             linewidth=lw, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)

    def _arr(ax, x0, y0, x1, y1, color="#555", lw=1.0, style="-|>", conn=None):
        kw = dict(arrowstyle=style, color=color, lw=lw, mutation_scale=8)
        if conn:
            kw["connectionstyle"] = conn
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=kw, zorder=2)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
    fig.subplots_adjust(wspace=0.12, left=0.01, right=0.99, top=0.91, bottom=0.08)

    # ── Left: CBM ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a) Concept Bottleneck Models", fontsize=10,
                 fontweight="bold", pad=6)

    # Input
    ax.text(0.03, 0.60, "x", ha="left", va="center", fontsize=10,
            color="#333", fontstyle="italic")
    ax.text(0.03, 0.52, "ℝ³⁹", ha="left", va="center", fontsize=7,
            color="#666")

    # Encoder
    _box(ax, 0.24, 0.56, 0.22, 0.15, "MLP Encoder\n256→256→64", C_MLP)
    _arr(ax, 0.13, 0.56, 0.13 + 0.01, 0.56, lw=1.4)

    # Concept heads
    _box(ax, 0.56, 0.56, 0.20, 0.15, "K=8 Concept\nHeads  σ(·)", C_CBM)
    _arr(ax, 0.35, 0.56, 0.46, 0.56, lw=1.4)

    # γ·BCE label
    ax.text(0.56, 0.74, "γ · BCE", ha="center", va="bottom", fontsize=6.5,
            color="#d62728", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#fff0f0",
                      edgecolor="#d62728", lw=0.6))
    _arr(ax, 0.56, 0.74, 0.56, 0.64, color="#d62728", lw=0.8)

    # Classifier
    _box(ax, 0.83, 0.56, 0.17, 0.13, "Classifier\n8→C", C_NESY)
    _arr(ax, 0.66, 0.56, 0.745, 0.56, lw=1.4)
    ax.text(0.975, 0.56, "ŷ", ha="center", va="center", fontsize=9,
            color="#333", fontweight="bold")
    _arr(ax, 0.915, 0.56, 0.965, 0.56, lw=1.0)

    # HybridCBM skip
    _arr(ax, 0.35, 0.60, 0.745, 0.60, color="#76b7b2", lw=1.0,
         style="-|>", conn="arc3,rad=-0.30")
    ax.text(0.54, 0.78, "skip (HybridCBM)", fontsize=6, color="#76b7b2",
            ha="center", style="italic")

    # Expert intervention
    ax.annotate("Expert\nintervention", xy=(0.56, 0.485),
                xytext=(0.56, 0.30),
                fontsize=6.5, color="#9467bd", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#9467bd", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f3eeff",
                          edgecolor="#9467bd", lw=0.7))

    # OOD scoring note
    ax.text(0.56, 0.15, "OOD: Mahalanobis in 8-dim concept space\n"
            "Test-time intervention on ĉ",
            ha="center", va="center", fontsize=6.2, color="#444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8",
                      edgecolor="#ccc", lw=0.7))

    # Variant legend
    variants = [
        ("JointCBM",      C_CBM,   "encoder + concept + classifier end-to-end"),
        ("SequentialCBM", C_BASE,  "concept heads frozen; classifier trained after"),
        ("HybridCBM",     "#76b7b2","encoder skip appended to concept predictions"),
        ("Post-hoc CBM",  C_GRAY,  "frozen encoder + linear concept probes"),
    ]
    for i, (lbl, col, desc) in enumerate(variants):
        yi = 0.03 + (3 - i) * 0.058
        ax.add_patch(plt.Rectangle((0.02, yi - 0.013), 0.022, 0.024,
                                   facecolor=col, edgecolor="none", zorder=3))
        ax.text(0.048, yi, lbl, fontsize=6.3, va="center",
                color="#222", fontweight="bold")
        ax.text(0.220, yi, desc, fontsize=5.8, va="center", color="#555",
                style="italic")

    # ── Right: NeSy-NIDS ─────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) NeSy-NIDS", fontsize=10, fontweight="bold", pad=6)

    # Input
    ax.text(0.03, 0.68, "x", ha="left", va="center", fontsize=10,
            color="#333", fontstyle="italic")
    ax.text(0.03, 0.60, "ℝ³⁷/³⁹", ha="left", va="center", fontsize=7,
            color="#666")

    # Rule branch
    _box(ax, 0.43, 0.78, 0.40, 0.17,
         "R Differentiable Rules\n"
         r"$\prod_k\sigma(\beta(x_{f_k}-\theta_{rk})d_{rk})$",
         C_NESY, fontsize=7.2)
    _arr(ax, 0.12, 0.72, 0.23, 0.78, lw=1.2)

    # k-annealing note
    ax.text(0.43, 0.90, "β: 1 → k_max over T_warm epochs (k-annealing)",
            ha="center", va="bottom", fontsize=6.0, color="#2ca02c",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#f0fff0",
                      edgecolor="#2ca02c", lw=0.5))

    # MLP branch
    _box(ax, 0.43, 0.46, 0.32, 0.14, "Residual MLP\n(neural fallback)", C_MLP,
         fontsize=7.2)
    _arr(ax, 0.12, 0.62, 0.27, 0.46, lw=1.2)

    # α-gate
    _box(ax, 0.83, 0.62, 0.20, 0.28,
         "α-Gate\nα·rules\n+(1−α)·MLP", C_BASE, fontsize=6.8)
    _arr(ax, 0.63, 0.78, 0.73, 0.68, lw=1.2)
    _arr(ax, 0.59, 0.46, 0.73, 0.58, lw=1.2)

    # α-reg note
    ax.text(0.83, 0.88, "λ_α·(1−α)²\npenalises\nneural reliance",
            ha="center", va="bottom", fontsize=5.8, color="#d62728",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#fff0f0",
                      edgecolor="#d62728", lw=0.5))

    # Output
    ax.text(0.975, 0.62, "ŷ", ha="center", va="center", fontsize=9,
            color="#333", fontweight="bold")
    _arr(ax, 0.93, 0.62, 0.965, 0.62, lw=1.0)

    # OOD note
    ax.annotate("Mahalanobis OOD\nin R-dim rule space",
                xy=(0.43, 0.695), xytext=(0.20, 0.28),
                fontsize=6.2, color="#9467bd", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#9467bd", lw=0.8,
                                linestyle="dashed"),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#f3eeff",
                          edgecolor="#9467bd", lw=0.6))

    # Bottom note
    ax.text(0.50, 0.04,
            "Thresholds θ_rk learned jointly  |  "
            "STE binarisation: 100% crisp rule activations at k=k_max",
            ha="center", va="center", fontsize=6.0, color="#444",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8",
                      edgecolor="#ccc", lw=0.7))

    for ext in ("pdf", "png"):
        kw = {"bbox_inches": "tight"}
        if ext == "png":
            kw["dpi"] = 200
        fig.savefig(OUTDIR / f"Architecture12.{ext}", **kw)
    plt.close(fig)
    print("Architecture12 done")


if __name__ == "__main__":
    make_threshold_drift()
    make_rule_selectivity()
    make_auroc_vs_f1()
    make_concept_space_pca()
    make_architecture12()
    print(f"\nAll figures written to {OUTDIR}")
