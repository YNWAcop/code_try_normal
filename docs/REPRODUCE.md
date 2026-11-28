# Training, checkpoint replay, and evaluation

## Scope

The repository provides the frozen text-present implementation and a portable text-absent task-head entry point. Historical paper results and new training runs are distinct. In particular, the portable V1 code explicitly does not promise bit-for-bit reproduction of historical ensembles.

Use the commands in the root README with your own feature and output paths. `configs/paths.example.json` documents paths; it is not loaded automatically. Global `--data-root` and `--run-root` arguments precede the subcommand.

## Complete text-path checkpoint

For one dataset, preserve the entire `runs/{dataset}/fold_0` directory:

```text
selection_lock.json
prep_main/
main/{C5,D5,anchor,cell}/best.pt
sub_{0..3}/prep/
sub_{0..3}/models/{C5,D5,anchor,cell}/best.pt
post/complete.pt
factor/best.pt
polarity.joblib
```

Artifacts referenced by the selection lock must exist and match its hashes. A single `best.pt` is insufficient. A portable V1 run needs its condition-specific `best.pt`, `scalers.joblib`, and `validation_lock.json`. Checkpoint files are not included in Git.

## Training lifecycle

`train-text` prepares inputs, trains MOSI and MOSEI, and writes validation-selected locks. Logs are stored under the run root. `evaluate-text` requires both dataset locks. `status` reads existing status files. Training does not automatically test unless the explicit `all-text` action is used.

Use a new output directory for a new experiment. Reuse the same frozen configuration and output directory only when resuming an interrupted run. RNG-state restoration supports continuation; it does not schedule seed sweeps. Do not overwrite archived paper runs.

## Metrics and selection

Reported classification metrics exclude zero ground-truth labels and use `prediction > 0` as the positive decision. F1 is support-weighted. MAE and Pearson correlation use all samples. Accuracy/F1 values are stored as fractions and displayed as percentages. Undefined correlation for constant predictions is represented as `null`.

The implementation also provides Acc-5 and Acc-7 using clipping and NumPy round-to-even. The frozen text-path selection utility includes six metrics (Acc-7, Acc-5, non-zero Acc-2, weighted F1, MAE, Corr), even when a paper table displays only four. See `text_present/protocol.py` for exact scaling and condition weights.

## Verification and provenance

`python -m unittest discover -s tests -v` checks module imports, metric definitions, calibration boundaries, visible-modality isolation, and equivalence to frozen source snapshots. `python run.py --data-root ./data check` additionally validates runtime hashes and feature headers. These checks do not constitute a fresh end-to-end training reproduction.

Earlier release checks are retained under `release/`. Their dates and scope remain historical. Development had exposure to the original test set, as recorded by `historical_design_exposure` and `pristine_independent_test` in the frozen protocol. Repackaging or a new initialization does not turn those results into a new independent confirmation.
