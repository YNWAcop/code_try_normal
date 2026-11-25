#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/root/autodl-tmp/MSA_Cloud_20261115/.venv_baselines/bin/python
mkdir -p outputs/formal
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
nohup "$PY" -u av_ablation.py train >> outputs/formal/train.log 2>&1 < /dev/null &
echo "Queue PID: $! (check status and train.log to confirm training)"
