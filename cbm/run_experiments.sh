#!/bin/bash
# run all cbm training and evaluation experiments
# execute from: /home/Ngari/Research/work/hyper_iot/
# usage: bash cbm/run_experiments.sh

set -e
PYTHON=/home/Ngari/Research/venv310/bin/python3
PROJ=/home/Ngari/Research/work/hyper_iot

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
echo "All experiments complete. Results in cbm/results/"
