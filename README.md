# Towards Trustworthy Network Security: Evaluating Concept-Based Interpretability and Neuro-Symbolic Out-of-Distribution Detection

**IEEE Transactions on Information Forensics and Security (TIFS)**  
Crisphine Macharia Ngari, Ning Yang, Ning Weng — Southern Illinois University Carbondale

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
│   ├── model.py            # MLPBaseline, JointCBM, SequentialCBM, HybridCBM, PosthocCBM
│   ├── concepts.py         # CTU and CIC concept definitions (K=8 per dataset)
│   ├── train.py            # Training entry point
│   ├── evaluate.py         # Evaluation + OOD AUROC scoring
│   ├── baselines.py        # DecisionTree, RandomForest, SHAP baselines
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
| MLP Baseline | 0.9323 | 0.913 | 0.8188 | 0.858 |
| SequentialCBM | 0.5731 | 0.678 | 0.7173 | **0.839** |
| HybridCBM | 0.9322 | 0.816 | 0.8181 | 0.808 |
| JointCBM (γ=0.5) | 0.9325 | 0.884 | 0.8154 | 0.591 |
| **NeSy-NIDS** | **0.9336** | **0.911**±0.011 | **0.8203** | 0.616±0.013 |
| Decision Tree† | 0.9350 | 0.335 | 0.8116 | 0.809 |

† OOD via raw-feature Mahalanobis. CBM results are single runs; NeSy-NIDS is mean±std over 5 seeds.

**Central finding — task-coupling OOD penalty:** OOD detectability on CIC-IoT-2023 degrades in proportion to how strongly bottleneck training is coupled to the classification objective, not due to compression per se. SequentialCBM (concepts learned independently of the task head) achieves AUROC=0.839 on CIC — *exceeding* the Decision Tree baseline (0.809) — despite compressing 39 raw features to the same 8-dimensional concept space. JointCBM degrades monotonically from 0.627 (γ=1.0) to 0.452 (γ=0.1) as task coupling increases; NeSy-NIDS (end-to-end rule learning) reaches 0.616. On CTU-IoT-23, NeSy-NIDS achieves the highest OOD AUROC of any model (0.911±0.011). The DT's near-random AUROC=0.335 reflects its reliance on the Telnet port flag (35% importance) — a class-specific artefact providing no distributional signal for OOD, confirming that raw-feature representations are not universally better than interpretable bottlenecks.

---

## Citation

```bibtex
@article{ngari2025interpretable,
  author  = {Ngari, Crisphine Macharia and Yang, Ning and Weng, Ning},
  title   = {Towards Trustworthy Network Security: Evaluating Concept-Based
             Interpretability and Neuro-Symbolic Out-of-Distribution Detection},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2025},
}
```

---

## License

Code released under the MIT License. Dataset terms follow the respective dataset providers (Stratosphere Lab / UNB CIC).
