# Validation performed during cleanup

The following checks passed on the prepared package:

- All 17 notebook code cells and both Python scripts compile.
- Notebook and exported experiment script have identical Python syntax trees.
- All notebook code outputs and execution counts are cleared.
- Synthetic cohorts with the same 266/58/23 and 85/58/23 class counts have disjoint train/validation/test subjects and exactly one held-out test appearance per subject.
- Paired bootstrap aligns shuffled subject rows, gives zero difference and zero-width intervals for identical predictions, and rejects duplicate subjects or conflicting labels.
- On nontrivial synthetic predictions, all three metrics’ point differences and CI endpoints exactly match the original cell 27 procedure using the same seed and 200 bootstrap draws. This small test checks implementation parity; production remains at 10,000 draws.
- Saved aggregate files contain seven configurations per cohort and six macro-F1 paired comparisons.
- The reported-value plotting script runs without MRI data and produces the expected output files; the macro-F1 plot was visually inspected.

Run the data-free tests from the project root:

```bash
python -m unittest discover -s tests -v
```

These checks do **not** validate new training results, GPU execution, checkpoint loading against real weight files, actual input-dataset matching, or complete end-to-end execution. PyTorch/torchvision and the MRI data, original trained checkpoints, and complete OOF CSV files were unavailable in the cleanup environment. No retraining or new MRI analysis was performed. The original environment was not reconstructed or containerized.
