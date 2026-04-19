"""
Shared path config — chỉ cần sửa file này khi chuyển sang máy mới.
"""
import os

# ── Code (writable) ───────────────────────────────────────────────────────────
BASE_DIR  = "/kaggle/working/mtdna_rerun"

# ── Data gốc (read-only Kaggle input) ────────────────────────────────────────
DATA_DIR  = "/kaggle/input/datasets/arrebol314/data-origin"

# ── Path gốc hardcode trong manifest json_path (để remap) ────────────────────
OLD_BASE  = "/home/minhtq/mtDNA_proj/mtdna_rerun"

# ── Các path derived — không cần chỉnh ───────────────────────────────────────
PIPELINE_DIR  = os.path.join(DATA_DIR,  "pipeline_results/pipeline_results")
METADATA_TSV  = os.path.join(DATA_DIR,  "metadata_rerun.tsv")
DATASET_DIR   = os.path.join(BASE_DIR,  "dataset")
TRAIN_CSV     = os.path.join(DATASET_DIR, "train_manifest.csv")
TEST_CSV      = os.path.join(DATASET_DIR, "test_manifest.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR,  "classifier_output")
FEATURES_CSV  = os.path.join(OUTPUT_DIR, "features.csv")
