#!/usr/bin/env bash
set -euo pipefail
BASE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/root/autodl-tmp/MSA_Cloud_20261115/.venv_baselines/bin/python}"
ROOT="${DATA_PROJECT:-/root/autodl-tmp/MSA_Cloud_20261115}"
OUT="$BASE/outputs"
ABLATION="${ABLATION:-/root/autodl-tmp/cloud_ablation_20261123}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$OUT"
"$PY" "$BASE/tav_queue.py" check --root "$ROOT" --out "$OUT"
nohup "$PY" -u "$BASE/tav_queue.py" run --root "$ROOT" --out "$OUT" --ablation "$ABLATION" >> "$OUT/launcher.log" 2>&1 < /dev/null &
printf 'Queue PID: %s\nLog: %s\n' "$!" "$OUT/launcher.log"
