#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/root/autodl-tmp/MSA_Cloud_20261115/.venv_baselines/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/MSA_Cloud_20261115/data}"
RUN_ROOT="${RUN_ROOT:-$BASE/outputs/formal}"
mkdir -p "$RUN_ROOT"
nohup "$PY" -u "$BASE/ablation.py" evaluate --data-root "$DATA_ROOT" --run-root "$RUN_ROOT" >> "$RUN_ROOT/evaluate.log" 2>&1 < /dev/null &
printf 'Evaluation PID: %s\nLog: %s\n' "$!" "$RUN_ROOT/evaluate.log"
