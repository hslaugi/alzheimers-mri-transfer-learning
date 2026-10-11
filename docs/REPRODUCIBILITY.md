# Reproducibility notes

## Preserved experiment design

ResNet-50; ImageNet V1 or remapped RadImageNet weights; 2048→512→3 head with ReLU/dropout 0.3; 224×224 RGB input; identical ImageNet normalization; training-only augmentation; slice-frequency class weights; AdamW at 1e-4 for head and 1e-5 for trainable backbone, weight decay 1e-4; batch 128; maximum 50 epochs; scheduler factor 0.5/patience 5; early stopping patience 10; checkpoint chosen by validation subject macro-F1. Backbone BatchNorm statistics update in training even when parameters are frozen. Evaluation fixes these statistics.

Five subject-level outer folds; the first split from nine inner stratified group folds supplies validation. Seed 42. Labels, configurations, label fractions, age threshold, subject-probability averaging, and paired stratified bootstrap percentile interval procedure are preserved.

## What changed in the implementation

Repeated and obsolete analysis blocks were consolidated. The runner and OOF inference now receive cohort-specific inputs explicitly. There is one metadata join and one splitting function. This fixes the manifest overwrite and undefined age seed. Model probes, duplicate OOF passes, exploratory threshold searches, and checkpoint deletions were removed. Inputs and cached predictions have stronger validation. Destructive extraction was replaced by a documented manual unzip step. Figure generation uses newly generated CSVs; saved paper values are separate.

## Why a fresh run may not match the paper exactly

The original notebook was run interactively, with diagnostic model initializations, cache deletions, interrupted/resumed runs, and manual condition reruns. These consume random numbers in ways that are not captured by a single seed. Removing them changes the RNG progression. The clean workflow seeds once, like the original, rather than falsely claiming a reconstructed per-fold seeding history. Fold-level resume also does not restore optimizer or RNG state.

The original manifest’s slice order is not supplied as a complete file. The first scan retains filesystem order and saves it for reuse. Different file orders, GPU kernels, package versions, and execution histories can alter exact predictions. No promise of bitwise or four-decimal agreement is made. Reproduction of the *reported* paired intervals requires the original subject prediction files. Those are not present in this package.

`run_config.json` prevents basic configuration/input mismatches in new cleaned runs; it is not proof of original-checkpoint provenance. Preserve the manifest and fold assignment for every checkpoint. Increment the implementation identifier/use a new output directory after code changes.

## Reported tables

`results/reported/fold_summary.csv` contains rounded full/age-restricted five-fold means and SDs from the supplied notebook’s final result/figure blocks. `paired_macro_f1.csv` contains the six reported paired differences and percentile intervals. These are aggregate transcriptions, not raw fold results or rerun estimates. The quick plotting script only visualizes these saved values.

## Items the authors still need to supply

- The exact source and version of the derived slice dataset and its permitted access route.
- A verified environment export from the training environment, if recoverable; otherwise disclose its absence and validate/document a new environment separately.
- Original matching checkpoints, manifests/splits, and OOF CSVs if sharing is permitted and exact-result reproduction is intended.
- An author-selected software license and the final proceedings citation when available.

No additional model training was performed during cleanup. The checks in VALIDATION.md are narrower than end-to-end experimental replication.
