#!/bin/bash
# Alpha-sweep driver.
# Trains + evaluates NeSy-NIDS across lambda_alpha and seeds, one dataset per GPU.
# Covers lambda_alpha in {0,0.05,0.1,0.2,0.5,1.0} at seeds 0-4 (lambda_alpha=0 is
# the main-table model and carries no _a suffix). Existing checkpoints are skipped,
# so delete or move a checkpoint to retrain it.
#
# usage: bash nesy/run_alpha_sweep.sh
set -u
PY=/home/Ngari/Research/venv310/bin/python3
cd /home/Ngari/Research
RES=results/nesy

LAMBDAS="0 0.05 0.1 0.2 0.5 1.0"
SEEDS="0 1 2 3 4"

run_dataset () {  # $1=dataset  $2=gpu
  local ds=$1 gpu=$2
  for la in $LAMBDAS; do
    for s in $SEEDS; do
      local tag="_a${la}"; [ "$la" = "0" ] && tag=""
      local ck="$RES/${ds}_nesy_s${s}${tag}.pt" ev="$RES/${ds}_nesy_s${s}${tag}_eval.json"
      if [ -f "$ck" ]; then echo "[skip] $ck"; continue; fi
      echo "[$ds gpu$gpu] train la=$la seed=$s"
      CUDA_VISIBLE_DEVICES=$gpu $PY -m nesy.train    --dataset $ds --seed $s --lambda_alpha $la --device cuda >/dev/null 2>&1
      CUDA_VISIBLE_DEVICES=$gpu $PY -m nesy.evaluate --dataset $ds --seed $s --lambda_alpha $la --device cuda >/dev/null 2>&1
      echo "[$ds gpu$gpu] done   la=$la seed=$s -> $ev"
    done
  done
  echo "[$ds] ALL DONE"
}

run_dataset ctu 0 &
run_dataset cic 1 &
wait
echo "ALPHA SWEEP COMPLETE"
