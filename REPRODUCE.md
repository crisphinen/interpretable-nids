# Reproducing the results

Commands in run order, from the repository root, with Python 3.10, the packages in
`requirements.txt`, the DuckDB CLI (`duckdb==1.4.1`) and the source databases under `data/`
(`iot23_3.duckdb`, `ciciot23.duckdb`; see the README). Every split script is deterministic: a rerun
gives byte-identical parquet files. Every training driver is resumable (finished runs are skipped)
and writes results with the dataset tag in the file name. `checks/paper_values.py` prints every
table value of the manuscript and supplement from these results.

Set `PY=python` (or the full path to your interpreter) and pick a GPU with `CUDA_VISIBLE_DEVICES`.

## CTU-IoT-23 splits

```bash
$PY 06_make_kfs_splits.py                        # data/ctuk (known-only features), data/ctuko (submitted list)
$PY 06_make_kfs_splits.py --split alt            # data/ctualt, data/ctualt_o
$PY 06_make_kfs_splits.py --split temporal       # data/ctut, data/ctut_o
$PY 06_make_kfs_splits.py --split device --holdout-host 192.168.1.198 --tag _h198     # data/ctud_h198, data/ctud_o_h198
$PY 06_make_kfs_splits.py --split device --holdout-host 192.168.100.111 --tag _h111   # data/ctud_h111, data/ctud_o_h111
$PY 06_make_kfs_splits.py --split device --holdout-host 192.168.100.108 --tag _h108   # data/ctud_h108, data/ctud_o_h108
$PY 08_make_ctuk_n.py                            # data/ctuk_n: ctuk rows, MLP/CBM input = NeSy-NIDS's 33 columns
```

## CTU-IoT-23 main runs, scorers and post-hoc detectors

```bash
$PY -m cbm.run_kfs_control --device cuda        # MLP and CBM variants, 5 seeds, ctuk + ctuko
bash nesy/run_kfs_control.sh ctuk ctuko         # NeSy-NIDS, NeSy + alpha-reg, rule-only, 5 seeds
$PY -m cbm.run_multiseed --aggregate
$PY -m cbm.run_scorer_factorial --datasets ctuk ctuko
$PY -m nesy.scorer_ablation --datasets ctuk ctuko
$PY -m nesy.run_shared_eval --datasets ctuk ctuko   # NeSy-NIDS rows on the evaluation flows of the CBM rows
$PY -m cbm.run_ood_baselines --datasets ctuk --seeds 5 --device cuda
```

## CIC-IoT-2023 split and runs

```bash
$PY 04_make_splits_ciciot23.py --name cicr --order-by-rowid     # ~60 GB RAM, ~2 min
$PY -m cbm.run_multiseed --dataset cicr --seeds 5 --device cuda
bash nesy/run_kfs_control.sh cicr
$PY -m cbm.run_multiseed --aggregate
$PY -m cbm.run_baselines --datasets ctuk ctuko cicr             # Post-hoc CBM, DT, RF
$PY -m nesy.run_shared_eval --datasets cicr                     # NeSy-NIDS rows and scorer table, CIC
$PY -m cbm.run_ood_baselines --datasets cicr --seeds 5 --device cuda   # CIC row of the post-hoc detector table
```

The Post-hoc CBM classifier is fit on the training split; rerun `cbm.run_baselines` after any
change to `cbm/run_baselines.py`.

## Robustness splits

```bash
$PY 04_make_splits_ciciot23.py --name cicr_alt --order-by-rowid --alt
for ds in ctualt ctut ctualt_o ctut_o; do $PY -m cbm.run_multiseed --dataset $ds --seeds 5 --device cuda; done
$PY -m cbm.run_multiseed --dataset cicr_alt --seeds 5 --device cuda \
    --variants MLPBaseline JointCBM_g0 JointCBM_g0.1 JointCBM_g0.5 JointCBM_g1.0
bash nesy/run_kfs_control.sh ctut ctut_o
$PY -m nesy.run_shared_eval --datasets ctut
```

## Device-disjoint splits and controls

```bash
for ds in ctud_h198 ctud_o_h198 ctud_h111 ctud_o_h111 ctud_h108 ctud_o_h108 tond_a30 tond_a31; do
  $PY -m cbm.run_multiseed --dataset $ds --seeds 5 --device cuda; done
bash nesy/run_kfs_control.sh ctud_h198 ctud_o_h198 ctud_h111 ctud_o_h111 ctud_h108 ctud_o_h108
$PY -m nesy.run_shared_eval --datasets ctud_h198 ctud_o_h198 ctud_h111 ctud_o_h111 ctud_h108 ctud_o_h108
$PY -m cbm.run_device_openset --device cuda      # known and unknown flows from the held-out device
$PY -m cbm.run_device_openset --reference        # seen-address baselines (CTU C&C, TON_IoT DoS)
$PY -m cbm.run_device_openset --mitm             # TON_IoT MITM flows of the held-out addresses, separately
$PY -m cbm.run_split_overlap                     # same-class train/test duplicate shares
$PY -m cbm.run_bottleneck_control --datasets ctuk cicr ton --device cuda   # 8-unit bottleneck, no concept labels
$PY -m cbm.run_multiseed --dataset ctuk_n --seeds 5 --device cuda \
    --variants MLPBaseline JointCBM_g0 SequentialCBM HybridCBM              # restored-features control
$PY -m cbm.run_multiseed --aggregate
```

## TON_IoT

Needs the TON_IoT `Processed_Network_dataset` CSVs under `downloads/TON_IoT/`, or set `TON_CSV_GLOB`
to their `Network_dataset_*.csv` glob.

```bash
$PY 07_make_splits_ton.py                        # data/ton, ~35 s
$PY 07_make_splits_ton.py --holdout-attacker 192.168.1.30 --holdout-normal 192.168.1.195 --tag _a30   # data/tond_a30
$PY 07_make_splits_ton.py --holdout-attacker 192.168.1.31 --holdout-normal 192.168.1.190 --tag _a31   # data/tond_a31
$PY -m cbm.run_multiseed --dataset ton --seeds 5 --device cuda
```

## Fidelity

```bash
bash nesy/run_alpha_sweep.sh ctuk               # lambda_alpha {0,...,1.0} x 5 seeds (0 and 0.1 reused)
bash nesy/run_alpha_sweep.sh cicr
$PY -m cbm.run_fidelity --datasets ctuk cicr    # concept intervention, CBM rows of the fidelity table
$PY -m nesy.run_fidelity --datasets ctuk cicr   # branch agreement, NeSy rows, rule quality, drift r,
                                                # hard_pred (predictions changed by hard activations)
$PY -m nesy.run_rule_init --datasets ctuk cicr  # rule quality at the initial expert thresholds
$PY -m nesy.make_fidelity_figures               # results/figures/threshold_drift5_*, rule_selectivity1_*
```

## Open-set (C+1) evaluation

```bash
$PY -m cbm.run_openset --datasets ctuk cicr
```

## Leave-family-out scorer selection

```bash
$PY -m cbm.run_lofo --make-folds                # data/{ctuk,cicr}_lofo_*/ (CTU folds re-select features)
$PY -m cbm.run_lofo --train --device cuda       # JointCBM gamma=0, 5 seeds per fold
$PY -m cbm.run_lofo --score --device cuda
$PY -m cbm.run_scorer_factorial --datasets cicr # true-best CIC scorer; CIC column of the scorer table
$PY -m cbm.run_scorer_factorial --datasets ton  # TON_IoT column of the scorer table
$PY -m cbm.run_scorer_selection --datasets ctuk cicr ton --device cuda   # synthetic-outlier selection rule
```

## Known-class geometry

```bash
bash nesy/run_kt_sweep.sh ctuk                  # NeSy k_T in {2,5,20} x 5 seeds (k_T=10 is the main model)
bash nesy/run_kt_sweep.sh cicr
$PY -m cbm.run_geometry --device cuda           # signflip and LOCO tables, confounding figure, NeSy r
$PY -m cbm.run_geometry_transfer                # leave-one-corpus-out prediction test
```

## Sweeps and per-class F1

```bash
$PY -m cbm.run_lambda_grid --dataset ctuk --device cuda   # lambda x gamma x 5 seeds; gamma=0 is the lambda table
$PY -m cbm.run_lambda_grid --dataset cicr --device cuda
$PY -m cbm.run_perclass --datasets ctuk cicr
```

The lambda = 0.5 cells reuse the main-table JointCBM checkpoints, so that column reproduces the
main results table exactly.

## Manuscript tables and figures

```bash
$PY -m cbm.run_bootstrap_ci --device cuda                     # 95% CIs of the AUROC comparisons
$PY checks/ctu_label_counts.py                                # CTU retained labels, unmatched (zero-filled) flows
$PY checks/paper_values.py                                    # every table value, mean and seed std
$PY -m nesy.aggregate_alpha_sweep --datasets ctuk:ctu cicr:cic --out results/nesy/alpha_sweep_main.json
$PY -m cbm.make_alpha_figure --input results/nesy/alpha_sweep_main.json   # paper/figures/alpha_sweep
$PY -m cbm.make_paper_figures --main                          # paper/figures/auroc_vs_f1
cp results/figures/*.pdf results/figures/*.png paper/figures/ # confounding, rule selectivity, drift
```
