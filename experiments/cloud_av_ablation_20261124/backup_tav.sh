#!/usr/bin/env bash
# Archive the completed supplement, including code, best/last weights and logs.
set -euo pipefail
cd /root/autodl-tmp
PY=/root/autodl-tmp/MSA_Cloud_20261115/.venv_baselines/bin/python
SRC=cloud_tav_rome_tmdc_20261124
"$PY" -c 'import json; from pathlib import Path; p=Path("cloud_tav_rome_tmdc_20261124/outputs/status.json"); d=json.loads(p.read_text()); assert d["stage"]=="complete" and d["completed"]==4 and d["test_opened"],d'
mkdir -p training_backups
TARGET=training_backups/TAV_RoME_TMDC_full.tar
if [[ -e "$TARGET" || -e "$TARGET.partial" ]]; then
  echo "Backup already exists or is partial; inspect before overwriting: $TARGET" >&2
  exit 1
fi
NEED=$(du -sb "$SRC" | cut -f1)
FREE=$(df -B1 --output=avail training_backups | tail -n1 | tr -d ' ')
if (( FREE < NEED + 2147483648 )); then
  echo "Not enough disk for a full archive plus 2 GiB reserve. Download the directory with scp -r instead." >&2
  exit 1
fi
tar -cf "$TARGET.partial" "$SRC"
mv "$TARGET.partial" "$TARGET"
cd training_backups
sha256sum TAV_RoME_TMDC_full.tar > TAV_RoME_TMDC_full.tar.sha256
ls -lh TAV_RoME_TMDC_full.tar*
