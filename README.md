# Towards Trustworthy IoT Network Security: Concept Bottlenecks and Differentiable Rules for Interpretable Open-Set Intrusion Detection

**Under review · 2026**  
Crisphine Macharia Ngari, Ning Weng — Southern Illinois University Carbondale

[![GitHub](https://img.shields.io/badge/GitHub-crisphinen%2Finterpretable--nids-blue?logo=github)](https://github.com/crisphinen/interpretable-nids)

---

## Overview

This repository contains all code, trained models, and evaluation results for our joint study of two inherently interpretable architectures for IoT network intrusion detection and open-set (OOD) detection:

- **Concept Bottleneck Models (CBMs)** — predictions routed through human-defined traffic concepts; supports test-time intervention
- **Neuro-Symbolic NIDS (NeSy-NIDS)** — domain-expert threshold rules differentiably learned via *k*-annealing and STE; exactly binary rule activations at inference

Both use Mahalanobis distance in their respective representation spaces for OOD scoring on **CTU-IoT-23** (4 known / 9 unknown classes) and **CIC-IoT-2023** (5 known / 29 unknown classes).

---

## Repository Structure

```
interpretable-nids/
├── cbm/                    # Concept Bottleneck Model code
│   ├── model.py            # MLPBaseline, JointCBM, SequentialCBM, HybridCBM
│   ├── concepts.py         # CTU and CIC concept definitions (K=8 per dataset)
│   ├── train.py            # Training entry point
│   ├── evaluate.py         # Evaluation + OOD AUROC scoring
│   ├── baselines.py        # DecisionTree, RandomForest, SHAP, Post-hoc CBM baselines
│   ├── make_figures.py     # All paper figures
│   └── run_experiments.sh  # Full experiment script
├── nesy/                   # NeSy-NIDS code
│   ├── model.py            # Rule bank, alpha-gate, STE binarisation
│   ├── train.py            # Training with k-annealing
│   ├── evaluate.py         # Evaluation + multi-seed aggregation
│   └── baselines.py        # Shared baseline utilities
├── data/
│   ├── 02_make_splits.py         # CTU-IoT-23 train/val/test split
│   ├── 04_make_splits_ciciot23.py # CIC-IoT-2023 split
│   ├── ctu/                      # CTU parquet splits + vocab
│   └── cic/                      # CIC parquet splits + vocab
│                                 # (test_unknown.parquet: download separately, see below)
├── results/
│   ├── cbm/                # Trained CBM checkpoints (.pt) + eval JSONs
│   └── nesy/               # Trained NeSy checkpoints (.pt) + eval JSONs (5 seeds)
├── paper/
│   ├── main.tex            # LaTeX source
│   ├── refs.bib            # Bibliography
│   ├── main.pdf            # Compiled paper
│   └── figures/            # All paper figures (PDF + PNG)
├── config.py               # CTU-IoT-23 dataset config
├── cic_config.py           # CIC-IoT-2023 dataset config
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

CUDA 11.8 is recommended. For CPU-only, replace `torch>=2.4.0` with the appropriate CPU wheel.

---

## Data

### CTU-IoT-23

Download from [Stratosphere Laboratory](https://www.stratosphereips.org/datasets-iot23). Pre-split parquet files are included in `data/ctu/`.

### CIC-IoT-2023

Download from the [CIC website](https://www.unb.ca/cic/datasets/iotdataset-2023.html). Then run:

```bash
python data/04_make_splits_ciciot23.py
```

> **Note:** `data/cic/test_unknown.parquet` (627 MB) is excluded from this repository due to GitHub's file-size limit. Generate it by running the split script on the raw CIC data, or request it from the authors.

---

## Reproducing Results

### CBM experiments (all variants + baselines)

```bash
cd interpretable-nids
bash cbm/run_experiments.sh          # trains all CBM variants on CTU and CIC
python -m cbm.evaluate --dataset ctu
python -m cbm.evaluate --dataset cic
```

### NeSy-NIDS (5 seeds)

```bash
python -m nesy.train  --dataset ctu --seed 0   # repeat for seeds 1-4
python -m nesy.train  --dataset cic --seed 0
python -m nesy.evaluate --dataset ctu --multi_seed 5
python -m nesy.evaluate --dataset cic --multi_seed 5
```

### Regenerate all paper figures

```bash
python -m cbm.make_figures
```

Figures are written to `paper/figures/`.

---

## Pre-trained Models

All trained checkpoints are in `results/cbm/` and `results/nesy/`. Load with:

```python
import torch
model = torch.load('results/cbm/ctu_JointCBM_g0.5.pt', map_location='cpu')
```

---

## Key Results

| Model | CTU F1 | CTU AUROC | CIC F1 | CIC AUROC |
|---|---|---|---|---|
| MLP Baseline | 0.9327 | 0.890 | 0.8210 | 0.806 |
| JointCBM (γ=0) | 0.9328 | 0.838 | 0.8197 | 0.668 |
| JointCBM (γ=0.1) | 0.9326 | **0.895** | 0.8186 | 0.654 |
| JointCBM (γ=0.5) | 0.9327 | 0.883 | 0.8187 | **0.728** |
| JointCBM (γ=1.0) | 0.9328 | 0.887 | 0.8190 | 0.707 |
| SequentialCBM | 0.5725 | 0.707 | 0.7200 | 0.675 |
| HybridCBM | 0.9326 | 0.825 | 0.8207 | 0.665 |
| Post-hoc CBM | 0.5783 | 0.864 | 0.7257 | 0.623 |
| **NeSy-NIDS** | **0.9334** | **0.906**±0.016 | 0.8192 | 0.605±0.006 |
| NeSy + α-reg | 0.9335 | 0.892±0.028 | 0.8192 | 0.623±0.020 |
| Rule-only | 0.5612 | 0.863 | 0.7561 | 0.693 |
| Decision Tree‡ | 0.9350 | 0.335 | 0.8116 | 0.809 |
| Random Forest‡ | 0.9351 | 0.627 | 0.8045 | 0.694 |

‡ OOD via raw-feature Mahalanobis. CBM results are single runs; NeSy-NIDS values are mean±std over 5 seeds. F1 is measured on the known-class validation split; AUROC and TPR@5%FPR on held-out unknown-class test samples.

**Central finding — task-coupling OOD penalty:** OOD detectability on CIC-IoT-2023 is governed by how tightly bottleneck training is coupled to the classification objective rather than by compression itself. On CIC, JointCBM peaks at γ=0.5 (AUROC 0.728), the best of any interpretable model there, while NeSy-NIDS drops to 0.605 — end-to-end rule learning couples the representation to the task most tightly. On CTU-IoT-23 the ordering reverses: NeSy-NIDS reaches 0.906±0.016 and JointCBM (γ=0.1) 0.895, both above the unconstrained MLP at 0.890, at no cost to classification F1. The Decision Tree's near-random 0.335 on CTU reflects score-polarity inversion under raw-feature Mahalanobis, confirming that raw-feature representations are not universally better than interpretable bottlenecks.

---

## Citation

```bibtex
@unpublished{ngari2026interpretable,
  author = {Ngari, Crisphine Macharia and Weng, Ning},
  title  = {Towards Trustworthy IoT Network Security: Concept Bottlenecks and
            Differentiable Rules for Interpretable Open-Set Intrusion Detection},
  note   = {Under review},
  year   = {2026},
}
```

---

## License

Code released under the MIT License. Dataset terms follow the respective dataset providers (Stratosphere Lab / UNB CIC).
