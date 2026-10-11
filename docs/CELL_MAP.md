# Changes from the original notebook

Cell numbers below are **zero-based positions in the 48-cell file**, not the inconsistent “Cell N” headings inside its code. Source: `AD_SubjectLevel_TL_1 (7).ipynb`.

| Original position | Action |
|---|---|
| 0 | Replaced the old notebook-comparison introduction with a plain-language project overview. |
| 1, 3 | Consolidated imports, one optional Drive mount, editable paths, run switches, and configuration. Removed personal Drive paths and duplicate worker setting. |
| 2 | Kept the RadImageNet key remapping function and coverage check; these are necessary for training. |
| 4 | Kept filename parsing and class mapping; removed automatic folder deletion/ZIP extraction; added manifest reuse and fail-fast input checks. |
| 5 | Generalized the same five-fold outer / first-of-nine inner split rule for both cohorts. |
| 6–9 | Retained datasets, transforms, architecture, optimizer, training, evaluation, and fold-level runner. Made cohort inputs explicit; removed setup-only model probes. |
| 10, 14 | Consolidated into one seven-configuration training section. |
| 11 | Replaced draft summaries with a single five-fold summary routine and final plotting section. |
| 12, 13, 17 | Removed stale result/checkpoint deletion scripts. |
| 15 | Removed obsolete fold-based significance analysis; kept the final subject-level bootstrap method from 27/38. |
| 16 | Removed selected-model diagnostic; final per-class reports and confusion matrices now use the three paired OOF models. |
| 18 | Retained three configurations and 25/50/100% label fractions; removed instructions to manually change “best” adaptation modes. |
| 19, 20 | Removed standalone weight diagnostics; retained the functional loader and coverage check from 2. |
| 21, 22 | Consolidated subject-ID-aware OOF inference, restricted to the three models used in the final paired comparisons. |
| 23–26 | Removed repeated mounting, path setup, and duplicate OOF loading; setup/loading occurs in the main sequence. |
| 27 | Retained paired class-stratified subject bootstrap and percentile CIs. Removed unreported bootstrap sign-proportion p-values. |
| 28, 29 | Consolidated class-specific reporting into the final OOF summaries; removed duplicate confusion-plot drafts. |
| 30 | Removed file-listing diagnostic. |
| 31 | Retained fold variability summary and count of RadImageNet-versus-ImageNet wins, without overwriting the image manifest. |
| 32 | Retained matching, earliest-MR selection, demographic summaries, missingness, and CDR agreement. Removed unrelated exploratory tests and verbose diagnostic output. |
| 33 | Retained the final age ≥60 cohort definition; removed candidate-threshold exploration from the public main workflow. |
| 34–36 | Removed duplicate device and variable-presence checks. |
| 37 | Retained fresh age-restricted splits and seven configurations with explicit inputs; no temporary overwriting of global manifest/splits/output directory. |
| 38 | Consolidated age-restricted OOF and paired statistics into the common routines. Removed undefined AGE60_SEED dependency and hard-coded assertions that a rerun must reproduce old model scores. |
| 39–41 | Removed superseded figure drafts and reconstruction machinery. |
| 42, 43 | Consolidated final plotting concepts into CSV-driven plots. Kept rounded reported values separately for quick preview; no hidden dependency on earlier plot cells. |
| 44–47 | Removed empty cells. |

All old execution outputs and Colab session metadata were cleared. No MRI data, subject-level metadata, trained weights, or private execution paths are included in the repository. Old outputs were not relabeled as results of the cleaned code.
