#!/bin/bash
# NeSy-NIDS alpha-gate sweep (tab:alpha, tab:headagree, tab:fidelity NeSy rows): lambda_alpha in
# {0, 0.05, 0.1, 0.2, 0.5, 1.0} x seeds 0-4, train + evaluate. lambda_alpha 0 and 0.1 come from
# nesy/run_kfs_control.sh; existing checkpoints are skipped, evaluation always reruns.
#
# usage: [PYTHON=...] [DEVICE=cuda] bash nesy/run_alpha_sweep.sh dataset [lambdas...]
set -eu
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
RES=results/nesy
ds="$1"; shift
LAMBDAS="${*:-0 0.05 0.1 0.2 0.5 1.0}"

for la in $LAMBDAS; do
  for s in 0 1 2 3 4; do
    tag=""; [ "$la" != "0" ] && tag="_a${la}"
    ck="$RES/${ds}_nesy_s${s}${tag}.pt"
    if [ -f "$ck" ]; then echo "[skip] $ck"; else
      echo "[$ds] train nesy la=$la seed=$s"
      "$PY" -m nesy.train --dataset "$ds" --seed "$s" --model nesy --lambda_alpha "$la" --device "$DEVICE" > "$RES/${ds}_nesy_s${s}${tag}_train.log" 2>&1
    fi
    "$PY" -m nesy.evaluate --dataset "$ds" --seed "$s" --model nesy --lambda_alpha "$la" --device "$DEVICE" > "$RES/${ds}_nesy_s${s}${tag}_eval.log" 2>&1
    echo "[$ds] done  nesy la=$la seed=$s"
  done
done
echo "ALPHA SWEEP ($ds) COMPLETE"
