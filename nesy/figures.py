# generate paper figures for nesy-nids.
#
# figures:
# 1. auroc_vs_f1.pdf          - auroc vs f1 scatter, all models x both datasets
# 2. rule_selectivity_ctu.pdf - heatmap of rule activations per class (ctu)
# 3. rule_selectivity_cic.pdf - heatmap of rule activations per class (cic)
# 4. threshold_drift.pdf      - init vs learned thresholds for each rule
#
# usage:
# python -m nesy.figures

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "nesy" / "results"
FIGS_DIR    = PROJECT_ROOT / "nesy" / "figures"

# feature index -> name for threshold drift labels
CTU_FEAT_NAMES = {
    6:  "IAT",
    8:  "resp_pkts",
    9:  "orig_pkts",
    10: "Rate",
    21: "syn_count",
    23: "duration",
    29: "fin_count",
    30: "rst_count",
}

CIC_FEAT_NAMES = {
    3:  "rate",
    12: "syn_count",
    13: "fin_count",
    14: "rst_count",
    22: "tcp",
    23: "udp",
    33: "avg",
    36: "iat",
    38: "variance",
}

# colour / marker scheme
STYLE = {
    "MLPBaseline":    dict(color="#2c2c2c", marker="^", s=120, label="MLP Baseline"),
    "JointCBM_g0":    dict(color="#1565C0", marker="o", s=100, label="JointCBM g=0"),
    "JointCBM_g05":   dict(color="#1E88E5", marker="o", s=120, label="JointCBM g=0.5"),
    "NeSy":           dict(color="#c0392b", marker="*", s=200, label="NeSy-NIDS"),
    "NeSy_aReg":      dict(color="#e74c3c", marker="*", s=200, label="NeSy + a-reg"),
    "RuleOnly":       dict(color="#e67e22", marker="s", s=100, label="Rule-only"),
    "DecisionTree":   dict(color="#27ae60", marker="D", s=110, label="Decision Tree"),
    "RandomForest":   dict(color="#16a085", marker="P", s=120, label="Random Forest"),
}

# hard-coded results (from eval runs)
# ctu results (mean over 5 seeds where applicable)
CTU_RESULTS = {
    "MLPBaseline":  dict(f1=0.9323, auroc=0.9132, tpr5=0.5959),
    "JointCBM_g0":  dict(f1=0.9322, auroc=0.882,  tpr5=0.4261),
    "JointCBM_g05": dict(f1=0.9325, auroc=0.918,  tpr5=0.6874,
                         auroc_std=0.019),
    "NeSy":         dict(f1=0.9336, auroc=0.9094, tpr5=0.553,
                         auroc_std=0.0125),
    "NeSy_aReg":    dict(f1=0.9336, auroc=0.8947, tpr5=0.505,
                         auroc_std=0.0225),
    "RuleOnly":     dict(f1=0.5612, auroc=0.8628, tpr5=0.459),
    "DecisionTree": dict(f1=0.9350, auroc=0.3351, tpr5=0.037),
    "RandomForest": dict(f1=0.9351, auroc=0.6268, tpr5=0.411),
}

# cic results
CIC_RESULTS = {
    "MLPBaseline":  dict(f1=0.8188, auroc=0.8581, tpr5=0.7153),
    "JointCBM_g0":  dict(f1=0.8168, auroc=0.5603, tpr5=0.0433),
    "JointCBM_g05": dict(f1=0.8154, auroc=0.5913, tpr5=0.1020),
    "NeSy":         dict(f1=0.8203, auroc=0.6144, tpr5=0.227,
                         auroc_std=0.0109),
    "NeSy_aReg":    dict(f1=0.8205, auroc=0.6341, tpr5=0.236,
                         auroc_std=0.0205),
    "RuleOnly":     dict(f1=0.7561, auroc=0.6926, tpr5=0.270),
    "DecisionTree": dict(f1=0.8116, auroc=0.8091, tpr5=0.560),
    "RandomForest": dict(f1=0.8045, auroc=0.6936, tpr5=0.140),
}


def load_eval_json(dataset, model, seed=0, suffix=""):
    p = RESULTS_DIR / f"{dataset}_{model}_s{seed}{suffix}_eval.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def load_baselines_json(dataset):
    p = RESULTS_DIR / f"{dataset}_baselines.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


# overwrite hard-coded values with whatever is on disk.
def update_results_from_disk():
    for dataset, res_dict in [("ctu", CTU_RESULTS), ("cic", CIC_RESULTS)]:
        # nesy (fixed rules, possibly with alpha-reg)
        for suffix in ["", "_a0.1"]:
            model_key = "NeSy_aReg" if suffix else "NeSy"
            for seed in range(5):
                tag = f"_s{seed}{suffix}"
                d = load_eval_json(dataset, "nesy", seed=seed, suffix=suffix.replace("_s0",""))
                if d:
                    if model_key not in res_dict:
                        res_dict[model_key] = {"f1": [], "auroc": [], "tpr5": []}
                    if isinstance(res_dict[model_key].get("f1"), list):
                        res_dict[model_key]["f1"].append(d["known_f1"])
                        res_dict[model_key]["auroc"].append(d["ood_auroc"])
                        res_dict[model_key]["tpr5"].append(d["tpr_at_5fpr"])

        # collapse lists to means
        for k, v in res_dict.items():
            if isinstance(v.get("f1"), list) and v["f1"]:
                res_dict[k] = {
                    "f1": np.mean(v["f1"]),
                    "auroc": np.mean(v["auroc"]),
                    "tpr5": np.mean(v["tpr5"]),
                    "f1_std": np.std(v["f1"]),
                    "auroc_std": np.std(v["auroc"]),
                }

        # baselines
        bl = load_baselines_json(dataset)
        if "DecisionTree" in bl:
            res_dict["DecisionTree"] = {
                "f1": bl["DecisionTree"]["f1"],
                "auroc": bl["DecisionTree"]["auroc_maha"],
                "tpr5": bl["DecisionTree"]["tpr5_maha"],
            }
        if "RandomForest" in bl:
            res_dict["RandomForest"] = {
                "f1": bl["RandomForest"]["f1"],
                "auroc": bl["RandomForest"]["auroc"],
                "tpr5": bl["RandomForest"]["tpr5"],
            }


# figure 1: auroc vs f1 scatter

def fig_auroc_vs_f1():
    update_results_from_disk()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle("OOD AUROC vs Classification F1 - All Models",
                 fontsize=13, fontweight='bold')

    legend_handles = []   # built once from the first subplot, reused for both

    for ax, (dataset, res_dict, title) in zip(
        axes,
        [("ctu", CTU_RESULTS, "CTU-IoT-23"),
         ("cic", CIC_RESULTS, "CIC-IoT-2023")]
    ):
        for model_key, vals in res_dict.items():
            if model_key not in STYLE:
                continue
            st    = STYLE[model_key]
            f1    = vals["f1"]
            auroc = vals["auroc"]

            sc = ax.scatter(f1, auroc,
                            color=st["color"], marker=st["marker"],
                            s=st["s"], zorder=5,
                            edgecolors='white', linewidths=0.6,
                            label=st["label"])

            if "auroc_std" in vals:
                ax.errorbar(f1, auroc, yerr=vals["auroc_std"],
                            fmt='none', color=st["color"],
                            capsize=3, zorder=4)

            # collect legend handles from the first axis only
            if ax is axes[0]:
                legend_handles.append(sc)

        ax.set_xlabel("Weighted F1 (known traffic)", fontsize=11)
        ax.set_ylabel("OOD AUROC", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0.28, 1.01)

    # single shared legend below both subplots, 4 columns to stay compact
    fig.legend(
        handles=legend_handles,
        labels=[st["label"] for mk in STYLE
                if mk in CTU_RESULTS
                for st in [STYLE[mk]]],
        loc='lower center',
        ncol=4,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor='#cccccc',
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = FIGS_DIR / "auroc_vs_f1.pdf"
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches='tight', dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# figure 2 & 3: rule selectivity heatmaps

def fig_rule_selectivity(dataset: str):
    # load eval json for seed 0
    d = load_eval_json(dataset, "nesy", seed=0)
    if d is None:
        print(f"  No eval JSON for {dataset} - skipping selectivity figure")
        return

    # extract selectivity from json if present (requires evaluate.py to save it)
    sel_path = RESULTS_DIR / f"{dataset}_nesy_s0_selectivity.json"
    if not sel_path.exists():
        print(f"  No selectivity JSON for {dataset} - skipping")
        return

    sel = json.loads(sel_path.read_text())
    rule_names = list(sel.keys())
    class_names = list(sel[rule_names[0]].keys())

    mat = np.array([[sel[r][c] for c in class_names] for r in rule_names])

    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 1.2), len(rule_names) * 0.55 + 1))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([c[:12] for c in class_names], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(rule_names)))
    ax.set_yticklabels(rule_names, fontsize=9)
    ax.set_title(f"Rule Class Selectivity - {dataset.upper()}", fontsize=11)

    for i in range(len(rule_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha='center', va='center',
                    fontsize=7.5, color='black' if mat[i, j] < 0.7 else 'white')

    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean activation (k=10)")
    plt.tight_layout()
    out = FIGS_DIR / f"rule_selectivity_{dataset}.pdf"
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches='tight', dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# figure 4: threshold drift

def fig_threshold_drift(dataset: str):
    import torch
    from nesy.model import NeSyNIDS, CTU_RULES, CIC_RULES

    rule_templates = CTU_RULES if dataset == "ctu" else CIC_RULES
    ckpt_path = RESULTS_DIR / f"{dataset}_nesy_s0.pt"
    if not ckpt_path.exists():
        print(f"  No checkpoint for {dataset} - skipping threshold drift figure")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = NeSyNIDS(ckpt["n_features"], ckpt["n_classes"], rule_templates)
    model.load_state_dict(ckpt["model_state_dict"])

    feat_names = CTU_FEAT_NAMES if dataset == "ctu" else CIC_FEAT_NAMES

    # collect init vs learned thresholds per rule
    rows = []
    for rt, rule in zip(rule_templates, model.rule_bank.rules):
        for i, (fi, ct) in enumerate(zip(rt.feature_indices, rt.condition_types)):
            init_val  = rt.init_thresholds[i]
            learned   = rule.thresholds[i].item()
            fname     = feat_names.get(fi, f"f{fi}")
            op        = ">" if ct == "gt" else "<"
            rows.append({
                "rule":  rt.name,
                "label": f"{rt.name}  [{fname} {op} theta]",
                "init":  init_val,
                "learned": learned,
                "drift": learned - init_val,
            })

    if not rows:
        return

    # horizontal bar chart - labels on y-axis at readable size
    labels = [r["label"] for r in rows]
    drift  = [r["drift"] for r in rows]
    colors = ["#c0392b" if d > 0 else "#2980b9" for d in drift]

    fig_h = max(5, len(rows) * 0.52 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y = np.arange(len(rows))
    bars = ax.barh(y, drift, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.65)
    ax.axvline(0, color='black', linewidth=0.9)

    # annotate each bar with the numeric drift value
    for bar, d in zip(bars, drift):
        x_pos = bar.get_width()
        ha    = 'left' if d >= 0 else 'right'
        off   = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0] or 1)
        ax.text(x_pos + (off if d >= 0 else -off), bar.get_y() + bar.get_height() / 2,
                f"{d:+.3f}", va='center', ha=ha, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Threshold drift (learned - init)", fontsize=10)
    ax.set_title(f"Learned Threshold Drift - {dataset.upper()}", fontsize=11)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    red_patch  = mpatches.Patch(color='#c0392b', label='threshold increased')
    blue_patch = mpatches.Patch(color='#2980b9', label='threshold decreased')
    ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc='lower right')

    plt.tight_layout()
    out = FIGS_DIR / f"threshold_drift_{dataset}.pdf"
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches='tight', dpi=150)
    print(f"  Saved: {out}")
    plt.close()


# main

def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures -> {FIGS_DIR}\n")

    fig_auroc_vs_f1()
    fig_threshold_drift("ctu")
    fig_threshold_drift("cic")
    fig_rule_selectivity("ctu")
    fig_rule_selectivity("cic")

    print("\nDone.")


if __name__ == "__main__":
    main()
