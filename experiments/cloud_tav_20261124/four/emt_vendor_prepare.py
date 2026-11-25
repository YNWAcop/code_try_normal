from pathlib import Path
def prepare(force=False):
    if force: raise RuntimeError("Repair source is frozen")
    return Path(__file__).resolve().parent / "vendor/emt_dlfr"
