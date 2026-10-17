#!/bin/bash
# NeSy-NIDS annealing-steepness sweep (main text, "A manipulation and an interpretation"):
# k_T in {2, 5, 20} x seeds 0-4, train + evaluate; k_T = 10 is the main model (no tag).
# Existing checkpoints are skipped. Geometry is computed by cbm/run_geometry.py --kt.
#
# usage: [PYTHON=...] [DEVICE=cuda] bash nesy/run_kt_sweep.sh dataset [k values...]
set -eu
cd "$(dirname "$0")/.." || exit 1
PY="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
RES=results/nesy
ds="$1"; shift
KS="${*:-2 5 20}"

for k in $KS; do
  for s in 0 1 2 3 4; do
    ck="$RES/${ds}_nesy_s${s}_k${k}.pt"
    if [ -f "$ck" ]; then echo "[skip] $ck"; else
      echo "[$ds] train nesy k_T=$k seed=$s"
      "$PY" -m nesy.train --dataset "$ds" --seed "$s" --model nesy --k_final "$k" --device "$DEVICE" > "$RES/${ds}_nesy_s${s}_k${k}_train.log" 2>&1
    fi
    "$PY" -m nesy.evaluate --dataset "$ds" --seed "$s" --model nesy --k_final "$k" --device "$DEVICE" > "$RES/${ds}_nesy_s${s}_k${k}_eval.log" 2>&1
    echo "[$ds] done  nesy k_T=$k seed=$s"
  done
done
echo "KT SWEEP ($ds) COMPLETE"
