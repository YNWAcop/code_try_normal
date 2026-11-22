"""Runtime locations only; algorithm parameters remain in the frozen protocol."""
import os
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get('MSA_DATA_ROOT', PROJECT / 'data')).expanduser().resolve()
RUN_ROOT = Path(os.environ.get('MSA_RUN_ROOT', PROJECT / 'runs/text_present')).expanduser().resolve()
