"""
Shared path config — chỉ cần sửa file này khi chuyển sang máy mới.
"""
import os

# ── Đổi dòng này khi clone sang máy/server khác ───────────────────────────────
# BASE_DIR = "/Users/nguyenthithutam/Desktop/mtdna_rerun"
BASE_DIR = "/kaggle/working/mtdna_rerun"
# Path gốc được hardcode trong train/test manifest (để remap json_path)
OLD_BASE = "/home/minhtq/mtDNA_proj/mtdna_rerun"

BASE1_DIR = "/kaggle/input/datasets/arrebol314/data-origin"
# ── Các path derived — không cần chỉnh ────────────────────────────────────────
PIPELINE_DIR  = os.path.join(BASE1_DIR, "pipeline_results")
METADATA_TSV  = os.path.join(BASE_DIR, "metadata_rerun.tsv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "classifier_output")
DATASET_DIR   = os.path.join(BASE1_DIR, "dataset")
FEATURES_CSV  = os.path.join(OUTPUT_DIR, "features.csv")
TRAIN_CSV     = os.path.join(DATASET_DIR, "train_manifest.csv")
TEST_CSV      = os.path.join(DATASET_DIR, "test_manifest.csv")
