# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

XGBoost binary classifier that detects sequencing errors in mitochondrial DNA (mtDNA) primer files. Input: fluorescent peak trace data (Tracy ABI JSON format). Output: per-file probability that a given primer/sample combination contains a sequencing error.

## Pipeline (run in order)

```bash
# Step 1: Extract features + train v1 classifier (cross-validated)
python build_classifier.py

# Step 2: Create reproducible train/test split at profile level
python build_dataset.py

# Step 3: Train final v2 classifier on clean split, evaluate on held-out test
python train_final_v2.py

# Optional: add pipeline_results paths to metadata
python map_results.py
```

All scripts use hardcoded paths pointing to `/home/minhtq/mtDNA_proj/mtdna_rerun`. **Update paths at the top of each script when running in a different environment.**

## Key Data

| File/Dir | Description |
|---|---|
| `metadata_rerun.tsv` | Ground truth labels — which primers failed per sample (Issues1/2/3 columns) |
| `pipeline_results/` | ~440 run folders, each containing Tracy JSON files per sample/primer |
| `classifier_output/features.csv` | 44 extracted features for 2302 labeled files (produced by `build_classifier.py`) |
| `dataset/train_manifest.csv` | 1886-row train split with features + labels |
| `dataset/test_manifest.csv` | 418-row isolated test split |
| `classifier_output/classifier_v2.pkl` | Final model bundle: XGBoost model, imputer, label encoder, feature list, threshold |

### JSON trace file path convention
`pipeline_results/{run_folder}/{sample_id}/{well}_{date}_{sample_id}_{primer}_{index}.json`

### Labeling logic
A file is label 1 (error) if its primer appears in the `Issues` column of the corresponding sample row in `metadata_rerun.tsv`. Label 0 = OK.

## Feature Engineering (44 features)

Extracted from Tracy JSON fields (`peakA/C/G/T`, `basecallQual`, `align1score`, `allele1fraction`, etc.):

- **Signal statistics**: `signal_mean`, `signal_max`, `signal_cv`, per-channel baseline (A/C/G/T)
- **Coverage**: `coverage_len`, `coverage_frac`, `read_start`, `read_end`, `read_length_frac`
- **Mixed signal** (secondary peak ratio): `sec_ratio_mean/max/p75/p90`, `n_mixed_pos`, `frac_mixed_pos`
- **Dyeblob detection**: `early_max_ratio`, `late_max_ratio`
- **Quartile analysis**: `q1/q2/q3_mean`, `q1/q2/q3_std`, `q1/q2/q3_sec_mean`
- **Quality**: `qual_mean`, `qual_p10`, `qual_below20_frac`, `qual_below40_frac`
- **Alignment**: `align1score`, `align2score`, `allele1frac`, `allele2frac`, `dropoff_pos`
- **Categorical**: `primer_enc` (label-encoded primer name) — top feature by importance

## Model Architecture

- **Algorithm**: XGBoost with `n_estimators=300, max_depth=5, learning_rate=0.05`
- **Class imbalance**: `scale_pos_weight` ≈ 1.74 (OK:Error ratio)
- **Validation**: `StratifiedGroupKFold(n_splits=5)` grouped by `sample_id` (prevents leakage — same sample across multiple runs stays in one fold)
- **Threshold**: 0.50 (stored in pkl; `error_files_thr020.csv` uses 0.20 for high-recall scanning)
- **v1 vs v2**: v1 uses run-folder grouping; v2 uses sample-level grouping (better generalization)

## Domain Notes

- Primers: HV1F, HV1R, HV2F, HV3R, HV3F, and variants (R389, F16190, etc.) covering mtDNA hypervariable regions
- Error types in metadata: dyeblobs, polyC stretches, high baseline, mixed signal, deletions
- `SIGNAL_THRESHOLD = 5` used to define readable coverage in feature extraction
- Code comments and metadata content are in Vietnamese
