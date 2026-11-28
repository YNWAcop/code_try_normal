# Fixed experiments

Only experiments associated with the frozen paper results are retained in the current tree.

| Directory under `experiments/` | Purpose | Entry point |
|---|---|---|
| `cloud_ablation_20261123` | T / TA / TV component ablations A0–A4 | `ablation.py` |
| `cloud_av_ablation_20261124` | A / V / AV Native versus Enhanced inputs | `av_ablation.py` |
| `baseline_repair_20261123` | EMT/MISA whole-modality adaptation fixes | `adapter_emt_tfr.py`, `adapter_misa_almt.py`, `run_emt_queue.py` |

## Comparison matrix

| ID | Input | Fusion | ERU |
|---|---|---|---|
| A4 | Enhanced | Concat | Off |
| A0 | Native | TEF | Off |
| A1 | Enhanced | TEF | Off |
| A2 | Native | TEF | On |
| A3 | Enhanced | TEF | On |

A0/A2 share the native backbone; A1/A3 share the enhanced backbone. Compare A1/A0 and A3/A2 for input enhancement, A1/A4 for TEF, and A2/A0 and A3/A1 for ERU. A3/A4 changes more than TEF alone. B0/B1 compare Native/Enhanced inputs with the same text-absent predictor. Ablation runs are independent of the frozen main-table runs.

## Running the archived packages

These directories preserve source versions rather than complete self-contained training bundles. Inspect their CLI with `python ablation.py --help` or `python av_ablation.py --help` from the relevant directory.

Before formal execution, restore the original `expected_data.json` and the package-specific code manifest: `release/runtime_manifest.json` for component ablations, or `manifest.json` for A/V/AV. The required frozen features must be available under `data/ready`. Validation guards deliberately reject missing or mismatched assets. Do not replace a historical manifest with an unrelated package's manifest or disable the guard to make a command pass.

After restoring dependencies, the action order is `check`, `train`, `evaluate`, then `export`, using an explicit `--data-root` and a fresh `--run-root`. The original scripts include historical paths and must be inspected before deployment.

EMT/MISA adapters additionally require their original upstream implementations, configurations, and training plans. The repaired EMT batch/reconstruction settings and MISA CMD/DiffLoss batch statistics are preserved. Full third-party implementations are not redistributed here.

Full-modality TAV extensions are retained in Git history at `9ddcb6d`. Later TEF exploratory trials and multi-seed queues are not part of this release.
