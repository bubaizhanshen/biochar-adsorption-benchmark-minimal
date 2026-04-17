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
├── requirements.txt
└── .gitignore
```

## Included data

- `data/HM2.xlsx` (`Dataset I`)
- `data/HMI_data.xlsx` (`Dataset II`)
- `data/EC.xlsx` (`Dataset III`)
- `data/PFAS.xlsx` (`Dataset IV`)
- `data/pfas_name_map.csv` (helper map for PFAS common names used in the benchmark task list)

## Included code

- `code/benchmark_table1.py`
  - nested RS/LOBO benchmark for the 21 manuscript tasks
  - outputs task-level `R²`, `MAE`, and `RMSE`
- `code/figure3_summary.py`
  - summary statistics and filtering structure for the four datasets
- `code/figure4_feature_ablation.py`
  - feature-ablation plot from benchmark output
- `code/figure5_tsne_overlap.py`
  - t-SNE visualization of RS and LOBO train/test overlap
- `code/nn_distance_rs_vs_lobo.py`
  - nearest-neighbor distance diagnostics under RS and LOBO
- `code/common_models.py`
  - shared model and dataset definitions used by SHAP analysis
- `code/shap_cross_model_ec.py`
  - shared-split RF/XGB SHAP comparison for Dataset III tasks
- `code/figure6_cbz_panels.py`
  - CBZ cross-model attribution panel generation

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

### 2. Figure 3 summary statistics

```bash
python code/figure3_summary.py
```

### 3. Figure 4 feature ablation

Run after `benchmark_table1.py` so that `outputs/benchmark_table1/summary_long.csv` exists.

```bash
python code/figure4_feature_ablation.py
```

### 4. Figure 5 overlap diagnostics

```bash
python code/figure5_tsne_overlap.py
python code/nn_distance_rs_vs_lobo.py
```

### 5. Figure 6 CBZ cross-model SHAP analysis

```bash
python code/shap_cross_model_ec.py --task CBZ --task IBU
python code/figure6_cbz_panels.py
```

## Notes

- The repository is intentionally minimal and manuscript-agnostic.
- The PFAS helper map is included so the benchmark can reproduce the manuscript task names without shipping any paper files.
- If you publish this repository, add the citation information you want users to follow for the compiled datasets and the original source studies.
