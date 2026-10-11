# Transfer learning for Alzheimer’s MRI classification

Code for **Transfer Learning for Alzheimer’s MRI Classification: A Subject-Level Study of ImageNet and RadImageNet Pretraining**, by Hemshankar Laugi and Haiyang Tang. Accepted to the IEEE BIBM ML4BMI workshop (2026).

We compare seven ResNet-50 configurations: scratch/full and ImageNet or RadImageNet with frozen, partial, or full adaptation. We also study limited training labels and an age ≥60 sensitivity cohort. Evaluation is at the subject level: all images from one person stay in the same split, and slice probabilities are averaged for one prediction per person.

The results do not establish a consistent pretraining-source advantage. Performance decreased after age restriction, but age balance and reduced sample size cannot be separated. This is research code, not a clinical diagnostic tool.

## Start here

- `notebooks/Alzheimers_MRI_Transfer_Learning.ipynb`: the complete, cleaned workflow with plain-language instructions.
- `scripts/experiment.py`: the same code exported as a Python script. Edit its paths and run flags before using it. Use either the notebook or script, not both for the same run.
- `results/reported/`: rounded aggregate values from the supplied paper/notebook, not outputs from a new execution.
- `scripts/plot_reported_results.py`: plot those saved aggregate values without MRI data or training.
- `docs/CELL_MAP.md`: what was retained, removed, or consolidated from the original 48-cell notebook.
- `docs/REPRODUCIBILITY.md`: preserved methods, implementation changes, and limits.
- `docs/VALIDATION.md`: what was actually checked.

## Quick preview without data or training

From the repository root:

```bash
python -m pip install numpy pandas matplotlib
python scripts/plot_reported_results.py
```

Plots are saved under `results/reported_figures/`. Their error bars are fold standard deviations. The CI table contains paired bootstrap intervals. These are different quantities. The age distribution plot cannot be reconstructed from aggregate means; the experiment notebook creates it from the actual metadata.

## Run the experiments

1. Install a compatible PyTorch/torchvision pair for your hardware using [PyTorch’s installation instructions](https://pytorch.org/get-started/locally/). Then install `requirements.txt`. In Colab, use a GPU runtime.
2. Obtain the image dataset, official metadata, and pretrained weights described below. They are not included here.
3. Open the notebook and set `DATA_DIR`, `METADATA_PATH`, `RAD_WEIGHTS`, and a new `OUTPUT_DIR`. Enable Drive mounting only if using Colab with Drive storage.
4. Read the instructions and run cells in order. Set `RUN_TRAINING=True` to train the seven full-cohort models, label-fraction comparisons, and seven age-restricted models.
5. Set `RUN_INFERENCE=True` to create subject-level OOF prediction files from trained checkpoints. Set `RUN_BOOTSTRAP=True` to calculate the paired comparisons and class-specific summaries. These steps do not train models.
6. Keep the saved manifest, splits, configuration, environment files, fold CSVs, checkpoints, and OOF files together. New plots read new-run CSVs, never the paper’s rounded aggregate tables.

**Training and inference are off by default.** The experiment notebook still requires valid image, metadata, and pretrained-weight paths for setup. Use the quick-preview script if you have none of these inputs. Training can take many hours or longer depending on hardware; the cleaned notebook has not been retrained here.

For inference-only use, supply matching checkpoints and configuration in the output folder. If all OOF files already exist, bootstrap can be run with inference off. The full notebook still validates the original input files and cohort. Older checkpoints should only be used with their original manifest, fold assignments, and training configuration; do not rename unrelated checkpoint files to satisfy the loader.

## Required inputs

### Images

The study used 86,437 pre-extracted axial image slices from 347 subjects. It did not extract these slices from raw NIfTI scans in this notebook. Expected layout:

```text
data/AD Datasets/Data/
  Non Demented/
  Very mild Dementia/
  Mild Dementia/
  Moderate Dementia/
```

A filename has the form `OAS1_0001_MR1_mpr-1_001.jpg` (PNG, JPEG, and BMP are also accepted). The subject ID, session, acquisition, and slice index are parsed from this name. Classes are 266 Non-Demented, 58 Very Mild, and 23 Mild+Moderate subjects. Moderate Dementia is merged with Mild because it has only two subjects.

**Provenance still needs author completion:** the supplied notebook does not establish the precise downloadable release/version of this derived slice dataset. The [OASIS project](https://sites.wustl.edu/oasisbrains/) is the source study, not a claim that its raw data directly match this folder layout. Record the exact derived dataset URL/version and permissions before describing this repository as independently reproducible from public inputs. Do not substitute a different Alzheimer’s image dataset.

### Metadata

Provide the OASIS-1 cross-sectional metadata file as `.xlsx`, `.xls`, or `.csv`. Required columns: `ID`, `Age`, `M/F`, `CDR`, `MMSE`, `Educ`. Keep the original subject/session identifiers. The script selects the earliest numbered MR metadata row per subject. Metadata is not used as a predictive model input.

### Pretrained weights

- ImageNet: torchvision `ResNet50_Weights.IMAGENET1K_V1`.
- RadImageNet: the [official RadImageNet repository](https://github.com/BMEII-AI/RadImageNet), accompanying Mei et al. (2022), provides links to PyTorch weight releases. The supplied notebook expects `RadImageNet-ResNet50_notop_torch.pth` with `backbone.*` keys. A loader maps them to torchvision ResNet-50 and checks backbone coverage. A filename alone is not enough; a different release with different keys will fail the check.

MRI data and pretrained weights are not redistributed here. Follow each provider’s access and use conditions.

## Environment and reproducibility status

`requirements.txt` is a dependency list, not a tested exact environment or a claim about the original software versions. Each new run writes `environment.json`, `requirements-resolved.txt`, and `run_config.json`. The source notebook’s saved CPU-version printout was not sufficient to identify the GPU training environment. No verified container or original environment lockfile is included.

No new MRI training or inference was performed when preparing this repository. The original checkpoint files and complete subject-level prediction CSVs were not part of the provided attachment. Aggregate tables cannot regenerate bootstrap CIs. See `docs/REPRODUCIBILITY.md` for why cleaned reruns may differ numerically.

## Citation

Hemshankar Laugi, Haiyang Tang, and Xi Zhu. “Transfer Learning for Alzheimer’s MRI Classification: A Subject-Level Study of ImageNet and RadImageNet Pretraining.” Accepted to the IEEE BIBM ML4BMI workshop, 2026.
