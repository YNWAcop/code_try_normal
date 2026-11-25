#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PY=/root/autodl-tmp/MSA_Cloud_20261115/.venv_baselines/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p outputs
nohup "$PY" -u supplement_queue.py run >> outputs/launcher.log 2>&1 < /dev/null &
echo "Queue PID: $!"
echo "Log: $(pwd)/outputs/launcher.log"
