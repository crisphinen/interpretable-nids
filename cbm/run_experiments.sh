#!/bin/bash
# run all cbm training and evaluation experiments
# can be run from any directory; set PYTHON to choose the interpreter
# usage: bash cbm/run_experiments.sh

set -e
PYTHON="${PYTHON:-python3}"
PROJ="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJ"

echo "============================================================"
echo "  CBM Experiments: Training"
echo "============================================================"

echo ""
echo "--- Training CTU-IoT-23 ---"
$PYTHON -m cbm.train --dataset ctu

echo ""
echo "--- Training CIC-IoT-2023 ---"
$PYTHON -m cbm.train --dataset cic

echo ""
echo "============================================================"
echo "  CBM Experiments: Evaluation"
echo "============================================================"

echo ""
echo "--- Evaluating CTU-IoT-23 ---"
$PYTHON -m cbm.evaluate --dataset ctu

echo ""
echo "--- Evaluating CIC-IoT-2023 ---"
$PYTHON -m cbm.evaluate --dataset cic

echo ""
echo "All experiments complete. Results in results/cbm/"
