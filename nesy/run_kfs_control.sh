#!/bin/bash
# Known-class feature-selection control, NeSy side (supplement Table kfs; Table I CTU).
# For each CTU dataset (ctuk, ctuko) and seeds 0-4, trains and evaluates
# NeSy-NIDS, NeSy + alpha-reg (lambda_alpha=0.1) and the rule-only ablation.
# Existing checkpoints are skipped, so delete one to retrain it.
#
# usage: [PYTHON=...] [DEVICE=cuda] bash nesy/run_kfs_control.sh [datasets...]
set -eu
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
RES=results/nesy
DATASETS="${*:-ctuk ctuko}"

for ds in $DATASETS; do
  for s in 0 1 2 3 4; do
    for cfg in "nesy 0" "nesy 0.1" "rule_only 0"; do
      set -- $cfg; model=$1; la=$2
      tag=""; [ "$la" != "0" ] && tag="_a${la}"
      ck="$RES/${ds}_${model}_s${s}${tag}.pt"
      if [ -f "$ck" ]; then echo "[skip] $ck"; else
        echo "[$ds] train $model la=$la seed=$s"
        "$PY" -m nesy.train --dataset "$ds" --seed "$s" --model "$model" --lambda_alpha "$la" --device "$DEVICE" > "$RES/${ds}_${model}_s${s}${tag}_train.log" 2>&1
      fi
      "$PY" -m nesy.evaluate --dataset "$ds" --seed "$s" --model "$model" --lambda_alpha "$la" --device "$DEVICE" > "$RES/${ds}_${model}_s${s}${tag}_eval.log" 2>&1
      echo "[$ds] done  $model la=$la seed=$s"
    done
  done
done
echo "KFS CONTROL (NeSy) COMPLETE"
