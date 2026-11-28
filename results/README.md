# Experimental results

`paper/` contains aggregate tables from the current manuscript handoff:

- `main_results.csv`: frozen main results and baseline comparisons.
- `ablation_results.csv`: separately trained ablation results.
- `module_gains.csv`: paired component comparisons.

Accuracy and F1 are stored as fractions; multiply by 100 for percentages. In `module_gains.csv`, gains are newer minus older for accuracy, F1, and correlation, and older minus newer for MAE. Positive values therefore indicate improvement for every metric. These are absolute differences, not relative percentage improvements.

The README selects only the six whole-modality settings and reports the frozen Ours rows as EviSA. Source CSVs retain their original method identifiers and numerical precision. They may include more settings than the README displays.

`reference/` preserves the earlier release's reference tables, including an older across-setting summary. Those historical files are not the source of the current README table. Do not mix independent ablation runs into the frozen main results or select model variants using test scores.

No new training or test evaluation was performed to reorganize this repository. Historical protocol and provenance remain in `configs/final_protocol.json` and `release/`.
