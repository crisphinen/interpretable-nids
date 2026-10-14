# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for a paper (under review) comparing two interpretable IoT intrusion-detection architectures — Concept Bottleneck Models (`cbm/`) and Neuro-Symbolic rule learning (`nesy/`) — on closed-set classification F1 and open-set (OOD) detection, using Mahalanobis distance in each model's representation space. Datasets: CTU-IoT-23 (4 known / 9 unknown classes) and CIC-IoT-2023 (5 known / 29 unknown). The README's results table is the source of truth for reported numbers; the archived checkpoints in `results/` come from an earlier run and do not match it exactly.

## Commands

Setup: `pip install -r requirements.txt` (Python 3.10, PyTorch ≥ 2.4). There is no test suite, linter, or build step.

Always run modules from the repo root with `python -m`. Entry points resolve paths from the repo root (most via `PROJECT_ROOT = Path(__file__).parent.parent`).

```bash
# CBM: trains MLPBaseline, JointCBM, SequentialCBM, HybridCBM (subset via --models)
python -m cbm.train    --dataset ctu [--gamma 0.5] [--models JointCBM] [--device cuda]
python -m cbm.evaluate --dataset ctu [--gamma 0.5] [--models JointCBM]

# NeSy-NIDS: one seed per run; --model rule_only for the rule-only ablation
python -m nesy.train    --dataset cic --seed 0 [--lambda_alpha 0.1] [--model rule_only]
python -m nesy.evaluate --dataset cic --seed 0 [--lambda_alpha 0.1]
python -m nesy.evaluate --dataset cic --multi_seed 5   # aggregates {ds}_nesy_s{0..4}.pt (λα=0 only)

# paper figures → paper/figures/
python -m cbm.make_paper_figures    # auroc_vs_f1, rule_selectivity1_*, and others
python -m cbm.make_alpha_figure     # alpha_sweep
python -m cbm.make_confounding_figure
```

`cbm/make_figures.py` and `cbm/make_missing_figures.py` are legacy. They expect the pre-September eval-JSON keys (`ood_auroc`, while current files use `ood_auroc_mahal`) and crash on current results. `make_missing_figures.py` also writes the same filenames into `paper/figures/` as `make_paper_figures.py`, so running it would replace the paper's versions.

The other `cbm/run_*.py` and `nesy/run_*.py` scripts are one-off experiments for the paper (multi-seed runs, λ/γ/k sweeps, temporal splits, scorer and geometry factorials).

## Gotchas

- **Data layout:** the CTU splits live directly in `data/` (`train/val/test_known/test_unknown.parquet` and `vocab.json`; these are committed). `data/ctu/vocab.json` is an unused copy. CIC lives in `data/cic/`, where only `vocab.json` is committed; the parquets must be generated with `04_make_splits_ciciot23.py` (`test_unknown.parquet` is about 627 MB).
- **Paths are repo-relative; machine-specific settings come from env vars:** `PYTHON` (interpreter for the `.sh` drivers, default `python3`), `DUCKDB_BIN` (DuckDB CLI, default `duckdb` on PATH), `IOT23_DB` (default `data/iot23_3.duckdb`), `DUCKDB_SPILL_DIR` (default `/tmp/duckdb_spill`), and `EXTRA_FIG_DIRS` (`:`-separated extra output folders for `make_paper_figures.py` / `make_alpha_figure.py`, which always write to `paper/figures/`). Keep new code free of absolute paths.
- `config.py` / `cic_config.py` are used by the split scripts (`02_make_splits.py`, `04_make_splits_ciciot23.py`, both at the repo root) and by `cbm/run_temporal.py` (which imports `config`). Training hyperparameters are module-level constants at the top of each `train.py`; label IDs and feature columns are read from `vocab.json`.
- Checkpoint filenames encode hyperparameters, and evaluate scripts rebuild the same name to find them: CBM `{ds}_{Model}{_g<gamma>}.pt`; NeSy `{ds}_{nesy|rule_only}_s{seed}{_sp<λs>}{_a<λα>}.pt`. Default (0) values get no suffix. `nesy.evaluate` rebuilds only the `_a` suffix (it has no `--lambda_sparse`), and `--multi_seed` always loads the λα=0 `{ds}_nesy_s{seed}.pt` regardless of `--model` / `--lambda_alpha`. `run_alpha_sweep.sh` skips existing checkpoints, and `cbm.train` does too unless given `--overwrite`, which protects the archived runs in `results/`.
- **Seed globs:** `results/nesy/` holds both the main seeds (`{ds}_nesy_s{0-4}_eval.json`) and the alpha-sweep runs (`{ds}_nesy_s{seed}_a{λ}_eval.json`). Match main seeds with `s[0-9]_eval.json` or exact names, never `s*_eval.json`, which also picks up all 25 sweep runs.
- `paper/` is gitignored and kept local only.

## Architecture

**Shared pipeline.** Each `train.py` has its own `load_dataset()`: it filters splits to known classes via `vocab.json`, then remaps label IDs to `0..n_classes-1` in sorted order. Each `evaluate.py` adds known-class val/test data plus a subsample of `test_unknown.parquet`. OOD scoring fits per-class Mahalanobis on training embeddings, optionally z-normalised and combined with an energy score. Metrics are AUROC and TPR at 5% FPR (`config.FPR_LEVELS` is not used by the evaluators).

**CBM (`cbm/`).** `concepts.py` defines 8 binary concepts per dataset as hand-written threshold functions on raw flow features (`ctu_concept_labels`, `cic_concept_labels`). These are the concept supervision targets. `model.py` builds every variant on a shared `MLPEncoder`:
- `JointCBM` trains concepts and class jointly (concept loss weight `LAMBDA_CONCEPT=0.5`), plus an optional `--gamma` concept-fidelity/leakage regulariser.
- `SequentialCBM` trains in two stages: the concept head first, then the classifier on frozen concepts.
- `HybridCBM` adds a residual side channel.

`evaluate.py` also runs concept accuracy and test-time intervention experiments.

**NeSy-NIDS (`nesy/model.py`).** `NeSyNIDS` = `RuleBank` + `NeuralFallback` MLP, mixed by a learned global gate α = sigmoid(param).
- Each `DiffRule` is a product of sigmoid gates `σ(k·(x−θ))` with learnable thresholds θ, built from a `RuleTemplate` in `CTU_RULES` / `CIC_RULES`.
- **Rule templates reference features by integer column index**, so they break if feature order in `vocab.json` changes.
- Training anneals k linearly from 1 to 10 over 30 epochs (`get_k` in `train.py`). Evaluation uses `K_EVAL=10` and optionally a straight-through binarisation.
- `--lambda_alpha` penalises (1−α)² to push weight toward the rules. `RuleOnlyNIDS` drops the neural fallback.
- `evaluate.py` also reports rule crispness, class selectivity, and rule-editing experiments.
