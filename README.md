# OASIS-1 CDR classification: whole-brain volume vs. deep learning, brain age and regional hippocampal features

Code for a leakage-controlled benchmark on the OASIS-1 cross-sectional MRI cohort. The study asks whether a 3D CNN, brain-age gaps, tissue-block features or regional (Harvard–Oxford) features separate CDR > 0 from CDR = 0 better than normalised whole-brain volume (nWBV), and how repeatable each measure is across repeat scans.

**Author:** M K Julkarnain

## Summary of results

Shared subject-level 5-fold cross-validation, 235 labelled participants (135 CDR 0, 70 CDR 0.5, 28 CDR 1, 2 CDR 2), AUROC for CDR > 0 vs 0:

| Model | AUROC (95% CI) |
|---|---|
| nWBV alone (reference) | 0.787 (0.728–0.846) |
| Age + sex + eTIV + nWBV | 0.773 |
| 3D CNN, age-pretrained | 0.794 (0.732–0.851) |
| 3D CNN brain-age gap | 0.655 |
| Regional volumes, six pre-specified regions | 0.822 |
| Volume + radiomic-style features, six regions (pre-specified primary) | 0.816 (gain over nWBV not significant) |
| Sparse top-1 feature F\*: hippocampal GM–CSF contact | 0.851 (0.798–0.896) |

F\* is the fraction of hippocampal grey-matter voxels that touch CSF. It remained ahead of nWBV in repeated CV with fresh seeds, at CDR 0.5 vs 0 and in an age-matched subset, and it was highly repeatable on 20 repeat scans (ICC 0.987). The regional result is exploratory and needs external validation. Deep learning and brain-age gaps did not beat nWBV.

## Repository structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── run_all.py                      runs the cells in order in one shared namespace
└── src/
    ├── 00_setup/                   configuration, Google Drive (rclone), restore from Drive
    ├── 01_baselines/               clinical features, volumetric brain age, global reliability, MMSE / stage analyses
    ├── 02_tissue_blocks/           8 mm tissue-block features, block brain age
    ├── 03_cnn/                     3D CNN: image loading, model, 5-fold training, results
    ├── 04_benchmark/               unified benchmark on identical folds, final paper tables
    ├── 05_regional_features/       Harvard–Oxford ROI features, ablation, radiomics, sparse selection, robustness, regional reliability
    └── 06_manuscript_pack/         Table 1, software versions, key statistics, publication figures
```

| Cell | File | What it does |
|---|---|---|
| 1 | `00_setup/cell01_setup.py` | paths, rclone config, `up()` upload helper |
| 1C | `00_setup/cell01c_restore.py` | downloads intermediate tables from Drive (skips Cells 2, 5–11, 13–15) |
| 2 | `01_baselines/cell02_features.py` | clinical table and tissue volumes from FSL_SEG |
| 3 | `01_baselines/cell03_brainage_models.py` | cross-fitted volumetric brain age (normal-only training) |
| 4 | `01_baselines/cell04_brainage_association.py` | brain-age gap vs CDR and MMSE |
| 13 | `01_baselines/cell13_reliability_global.py` | test–retest of global measures |
| 14 | `01_baselines/cell14_secondary_analyses.py` | stage-stratified AUROC, MMSE regression |
| 5–7 | `02_tissue_blocks/` | 8 mm blocks, block brain age, PCA + logistic regression |
| 8–11 | `03_cnn/` | image loading, 3D CNN, CV (scratch vs age-pretrained), results |
| 12 | `04_benchmark/cell12_unified_benchmark.py` | all models on identical folds, paired bootstrap |
| 18 | `04_benchmark/cell18_final_tables.py` | final tables and figures from saved out-of-fold scores |
| 15–17 | `05_regional_features/` | ROI features, ROI vs nWBV, ablation and age matching |
| 19–22 | `05_regional_features/` | radiomic-style features, regional test–retest, sparse selection, robustness |
| 23 | `06_manuscript_pack/cell23_manuscript_pack.py` | Table 1, versions, figures, zip and upload |

The scripts were written as Kaggle notebook cells and share variables (`OUT`, `lab`, `D`, `y`, `FR`, `rid`, ...). They are not independent modules and must run in order in one process, which `run_all.py` does.

## Data

OASIS-1 is not included. Download it from <https://www.oasis-brains.org> under its data use agreement (Marcus et al., 2007, *J Cogn Neurosci* 19:1498–1507). The code expects the distributed atlas-registered, skull-stripped images and FSL_SEG segmentations in one folder per session (`OAS1_xxxx_MR1`, `OAS1_xxxx_MR2`) plus the clinical table, under `DATA_DIR`.

Acknowledgement required by OASIS: *Data were provided by OASIS: Cross-Sectional: Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991, P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382.*

## Installation

```bash
git clone <this repository>
cd oasis1-hippocampal-csf-contact
pip install -r requirements.txt
```

A GPU is needed only for the CNN cells (Cells 9–10; the study used one NVIDIA T4). Internet is needed on the first run for pip, the Harvard–Oxford atlas download and rclone.

## Configuration

Edit the top of `src/00_setup/cell01_setup.py`:

| Variable | Meaning |
|---|---|
| `OUT` | working folder for all outputs |
| `DATA_DIR` | folder containing the OASIS-1 data |
| `RCLONE_REMOTE`, `DRIVE_FOLDER_ID`, `DRIVE_SUBDIR` | Google Drive destination for uploads |

On Kaggle the rclone token is read from the secret `RCLONE_DRIVE_TOKEN`. Without it, uploads print a warning and the analysis still runs and saves locally.

## Running

```bash
python run_all.py --mode full      # from raw OASIS-1 (long; CNN cross-validation dominates)
python run_all.py --mode restore   # from intermediate tables already on Drive (Cells 1, 1C, 3, 12, 15, 19, 18, 23)
python run_all.py --list --mode full
```

Full order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 18, 23. To run in a Kaggle notebook instead, paste the files into cells in this order.

Outputs go to `OUT` (and Drive): per-cell result tables, out-of-fold scores (`unified_oof_scores.csv`, `roi_oof_scores.csv`), and `paper_pack/` with the figures and Table 1 produced by Cell 23.

## Evaluation design

- One stratified, subject-level 5-fold partition (seed 0) shared by every model, so out-of-fold scores are paired.
- Repeated 5×5 CV (seed 7) and a fresh 10×5 CV (seed 2024) for robustness.
- Paired subject bootstrap (2000 resamples) for AUROC and AUROC differences; equivalence margin ±0.05 fixed in advance.
- Brain-age models trained on normal participants only (CDR 0 in training folds plus 181 younger unlabelled participants); gaps always computed out of fold.
- Six regions fixed in advance: hippocampus, amygdala, anterior and posterior parahippocampal gyrus, temporal pole, lateral ventricle (24 volume + 72 radiomic-style = 96 features). The sparse model ranks the 96 features by univariate AUROC inside each training fold.
- Sensitivity analyses: CDR 0.5 vs 0, CDR ≥ 1 vs 0, and an age-matched subset (±2 years).
- Test–retest on the 20 participants with an MR2 session: ICC(2,1) and repeatability coefficient.

## Software used

Python 3.13, NumPy 2.1.3, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.6.1, PyTorch 2.11.0, nibabel 5.4.2, nilearn 0.14.1, Matplotlib 3.10.0.

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`). OASIS-1 data remain under the OASIS data use agreement.
