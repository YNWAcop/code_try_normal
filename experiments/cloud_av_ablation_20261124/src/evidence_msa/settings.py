"""Runtime locations only; algorithm parameters remain in the frozen protocol."""
import os
INPUT_MODE = os.environ.get("MSA_INPUT_MODE", "enhanced")
BACKBONE = os.environ.get("MSA_BACKBONE", "c5")
SMOKE = os.environ.get("MSA_SMOKE", "0") == "1"
DEVICE = os.environ.get("MSA_DEVICE", "cuda")
assert INPUT_MODE in ("native", "enhanced") and BACKBONE in ("c5", "simple")
assert DEVICE == "cuda" or (DEVICE == "cpu" and SMOKE), "CPU is reserved for synthetic smoke checks"
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get('MSA_DATA_ROOT', PROJECT / 'data')).expanduser().resolve()
RUN_ROOT = Path(os.environ.get('MSA_RUN_ROOT', PROJECT / 'runs/text_present')).expanduser().resolve()
