#!/bin/bash
# Reconstructed alpha-sweep driver (the original was run ad-hoc and lost).
# Trains + evaluates NeSy-NIDS across lambda_alpha and seeds, one dataset per GPU.
# lambda_alpha in {0,0.1} already have 5-seed evals from the main run; this script
# fills the intermediate points {0.05,0.2,0.5,1.0} at seeds 0-4.
#
# usage: bash nesy/run_alpha_sweep.sh
set -u
PY=/home/Ngari/Research/venv310/bin/python3
cd /home/Ngari/Research
RES=results/nesy

LAMBDAS="0.05 0.2 0.5 1.0"
SEEDS="0 1 2 3 4"

run_dataset () {  # $1=dataset  $2=gpu
  local ds=$1 gpu=$2
  for la in $LAMBDAS; do
    for s in $SEEDS; do
      local ev="$RES/${ds}_nesy_s${s}_a${la}_eval.json"
      if [ -f "$ev" ]; then echo "[skip] $ev"; continue; fi
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
