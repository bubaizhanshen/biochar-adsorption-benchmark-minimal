# Biochar Adsorption Benchmark: Minimal GitHub Release

This release contains only the code and curated data tables needed to reproduce the core analyses used in the manuscript. It intentionally excludes:

- manuscript files
- supplementary-information files
- source-paper PDFs
- precomputed results
- internal drafting and revision utilities

## Repository layout

```text
github_release_minimal/
├── code/
├── data/
├── outputs/
├── requirements.txt
└── .gitignore
```

## Included data

- `data/HM2.xlsx` (`Dataset I`)
- `data/HMI_data.xlsx` (`Dataset II`)
- `data/EC.xlsx` (`Dataset III`)
- `data/PFAS.xlsx` (`Dataset IV`)
- `data/pfas_name_map.csv` (helper map for PFAS common names used in the benchmark task list)
- `data/zhao2025_hg0_case.csv` (row-level Hg0-removal case-study table extracted from the Supporting Information of Zhao et al. 2025; used only as an independent external corroboration case)

## Included code

- `code/benchmark_table1.py`
  - nested RS/LOBO benchmark for the 21 manuscript tasks
  - outputs task-level `R²`, `MAE`, and `RMSE`
- `code/nn_distance_rs_vs_lobo.py`
  - nearest-neighbor distance diagnostics under RS and LOBO
  - exports numeric summaries only
- `code/common_models.py`
  - shared model and dataset definitions used by SHAP analysis
- `code/shap_cross_model_ec.py`
  - shared-split RF/XGB SHAP comparison for Dataset III tasks
- `code/run_zhao2025_external_case.py`
  - independent external literature-based case study on the Zhao et al. Hg0-removal dataset
  - compares repeated random splitting with leave-one-reference-out evaluation
  - outputs predictive, screening, and nearest-neighbor diagnostics

All generated files are written to `outputs/`, which is gitignored by default.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Suggested run order

### 1. Main benchmark table

```bash
python code/benchmark_table1.py
```

### 2. Overlap diagnostics

```bash
python code/nn_distance_rs_vs_lobo.py
```

### 3. Cross-model SHAP analysis

```bash
python code/shap_cross_model_ec.py --task CBZ --task IBU
```

### 4. Independent external corroboration case

```bash
python code/run_zhao2025_external_case.py
```

## Notes

- The repository is intentionally minimal and manuscript-agnostic.
- Plotting scripts were intentionally omitted from this release.
- The Zhao case study is not merged into the 21-task benchmark because it uses a different target system and response definition.
- The PFAS helper map is included so the benchmark can reproduce the manuscript task names without shipping any paper files.
- If you publish this repository, add the citation information you want users to follow for the compiled datasets and the original source studies.
