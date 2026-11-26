# Osteosarcoma AutoScore-Survival — analysis code

Single-file, self-contained pipeline that reproduces every number, table and
figure in the manuscript.

## What it does

| Step | Content |
|---|---|
| 1 | Load the SEER development cohort and the Chinese external cohort |
| 2 | missForest imputation, fitted on the training set only |
| 3 | Random-survival-forest permutation importance (`n = 10` candidates; race excluded a priori) |
| 4 | Parsimony curve by repeated 5-fold cross-validation **inside the training set** |
| 5 | Variable selection: an addition is kept only if it raises the cross-validated C-index by ≥ 0.01 |
| 6 | AutoScore conversion: ridge Cox (L2, α = 1.0) → integer points |
| 7 | Risk-band cut-offs derived from predicted 3-year risk (10% / 40%) in the training set |
| 8 | Internal and external evaluation with bootstrap 95% CIs (C-index, AUC, slope, O/E, Brier) |
| 9 | Band calibration, decile calibration, decision-curve analysis |
| 10 | Subgroup discrimination |
| 11 | Histological response (Huvos) linkage and analysis |
| 12 | Tables 1–3 and Figures 1–4 plus Supplementary Figure S1 |

## Requirements

```
python >= 3.10
numpy, pandas, scipy, scikit-learn, scikit-survival >= 0.22, matplotlib, openpyxl
```

Install with:

```bash
pip install -r requirements.txt
```

## Input files

Place these next to the script (or point to them with environment variables):

| File | Sheets used |
|---|---|
| `骨肉瘤生存期预测--最终预测版本 - 副本.xlsx` | `SEER`, `外部验证` |
| `301---骨肉瘤随访  筛选完 - 副本.xlsx` | `Sheet1` (Huvos grades) |

Override the paths when needed:

```bash
set OS_INPUT=D:\data\seer_external.xlsx
set OS_HUVOS=D:\data\pathology.xlsx
```

## Run

```bash
python analysis_pipeline.py
```

A full run takes roughly 10–15 minutes on a laptop. For a quick structural
check (fewer bootstrap replicates; **the variable selection may differ** because
cross-validation noise increases):

```bash
set OS_FAST=1
python analysis_pipeline.py
```

All output is written to `results/`:

```
results/
├── summary.json              every reported quantity
├── table1_baseline.csv
├── table2_score_raw.csv
├── table3_performance.csv
├── parsimony_curve.csv
├── bands.csv  deciles.csv  dca.csv  subgroups.csv  huvos.csv
├── run.log                   full console transcript
└── figures/
    ├── Figure1.tif           study flow
    ├── Figure2.tif           parsimony curve and Kaplan-Meier by band
    ├── Figure3.tif           calibration and decision curve
    ├── Figure4.tif           subgroup forest plot
    └── SuppFigure_S1.tif     histological response
```

## Reproducibility notes

* Random seed `2026` throughout (split, forest, cross-validation, bootstrap).
* The 30% validation set is **not** used for any selection decision; the number
  of variables is chosen by cross-validation within the training set, so the
  internal estimate is a genuine hold-out value.
* The imputer is fitted on the training set and applied unchanged to the
  validation and external cohorts.
* Treatment variables are excluded a priori; race is excluded because the
  external cohort is ethnically homogeneous.
* Histological response is linked on age band, sex, tumour size, primary site,
  grade, nodal status, metastatic status, treatment delay and vital status, with
  survivor follow-up times aligned by a constant offset estimated from the
  unambiguous matches.

## Expected console output (abridged)

```
[1] cohorts: SEER 2066 (821 deaths), external 816 (314 deaths)
[3] permutation importance ...
[4] parsimony curve ...
    first increment below 0.010 at k=5; retained 4 variables
[5] integer score ... maximum 17 points
    cut-offs: low <=3, intermediate 4-9, high >=10
[6] external: C 0.692 (0.665-0.721) | AUC 0.708 ...
```
