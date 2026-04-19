"""
Dataset Builder for mtDNA Sequencing Error Classifier
======================================================
Tổ chức dữ liệu thành train/test theo cấp profile (sample_id).

Một profile = 1 sample (id như 253719, LN_25_AA0143, ...) gồm
tối đa 4 file JSON: HV1F, HV1R, HV2F/HV3R (tuỳ run).

Split tại cấp profile (không tách file của cùng 1 sample ra 2 nhánh).

Cấu trúc output:
  dataset/
  ├── train/
  │   ├── label_0/   ← symlinks đến JSON file "OK"
  │   └── label_1/   ← symlinks đến JSON file lỗi
  ├── test/
  │   ├── label_0/
  │   └── label_1/
  ├── train_manifest.csv   ← metadata + features
  ├── test_manifest.csv
  └── split_info.json      ← ghi lại cách split để reproducible
"""

import os
import json
import random
import shutil
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FEATURES_CSV  = "/home/minhtq/mtDNA_proj/mtdna_rerun/classifier_output/features.csv"
DATASET_DIR   = "/home/minhtq/mtDNA_proj/mtdna_rerun/dataset"
TEST_RATIO    = 0.20      # 20% profiles vào test
RANDOM_SEED   = 42
USE_SYMLINKS  = True      # True: tạo symlink; False: copy file (nặng hơn)

# ──────────────────────────────────────────────
# STEP 1: Load và tóm tắt dữ liệu
# ──────────────────────────────────────────────
print("Loading features.csv...")
df = pd.read_csv(FEATURES_CSV)
print(f"  {len(df)} labeled files, {df['label'].eq(1).sum()} errors, {df['label'].eq(0).sum()} OK")

# ──────────────────────────────────────────────
# STEP 2: Tính profile-level stats
# ──────────────────────────────────────────────
profile_df = df.groupby('sample_id').agg(
    n_files  = ('label', 'count'),
    n_error  = ('label', 'sum'),
    primers  = ('primer', lambda x: sorted(set(x))),
    runs     = ('run_folder', lambda x: sorted(set(x))),
).reset_index()

# Gán stratification label cho từng profile
# → 0: tất cả files OK; 1: có ≥1 file lỗi
profile_df['has_error'] = (profile_df['n_error'] > 0).astype(int)

print(f"\nTotal profiles: {len(profile_df)}")
print(f"  With errors:  {profile_df['has_error'].sum()}")
print(f"  All-OK:       {(profile_df['has_error']==0).sum()}")
print(f"\nFiles per profile:")
print(profile_df['n_files'].value_counts().sort_index().to_string())

# ──────────────────────────────────────────────
# STEP 3: Stratified split tại cấp profile
# Split dựa trên has_error để đảm bảo cả train và test
# đều có tỉ lệ profiles lỗi tương tự nhau.
# ──────────────────────────────────────────────
rng = random.Random(RANDOM_SEED)

train_profiles = []
test_profiles  = []

# Split riêng từng nhóm (has_error=0 và has_error=1)
for group_val in [0, 1]:
    group = profile_df[profile_df['has_error'] == group_val]['sample_id'].tolist()
    rng.shuffle(group)
    n_test = max(1, int(len(group) * TEST_RATIO))
    test_profiles.extend(group[:n_test])
    train_profiles.extend(group[n_test:])

train_set = set(train_profiles)
test_set  = set(test_profiles)

print(f"\nSplit result:")
print(f"  Train profiles: {len(train_set)}")
print(f"  Test profiles:  {len(test_set)}")

# Assign split column vào df
df['split'] = df['sample_id'].apply(
    lambda sid: 'train' if sid in train_set else ('test' if sid in test_set else 'unknown')
)

df_train = df[df['split'] == 'train']
df_test  = df[df['split'] == 'test']

print(f"\n  Train files: {len(df_train)} (err={df_train['label'].eq(1).sum()}, ok={df_train['label'].eq(0).sum()})")
print(f"  Test  files: {len(df_test)}  (err={df_test['label'].eq(1).sum()}, ok={df_test['label'].eq(0).sum()})")

# ──────────────────────────────────────────────
# STEP 4: Tạo folder structure
# ──────────────────────────────────────────────
for split in ['train', 'test']:
    for label in ['label_0', 'label_1']:
        Path(f"{DATASET_DIR}/{split}/{label}").mkdir(parents=True, exist_ok=True)

print(f"\nCreated folder structure at {DATASET_DIR}/")

# ──────────────────────────────────────────────
# STEP 5: Tạo symlinks / copies
# Tên file trong folder: <sample_id>__<primer>__<run>__<original_name>
# → đảm bảo unique, còn truy vết được back về file gốc
# ──────────────────────────────────────────────
def make_link(src: str, dst_dir: str, filename: str):
    dst = Path(dst_dir) / filename
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if USE_SYMLINKS:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


n_linked = 0
name_collisions = defaultdict(int)

for _, row in df.iterrows():
    if row['split'] not in ('train', 'test'):
        continue

    label_dir = f"{DATASET_DIR}/{row['split']}/label_{int(row['label'])}"
    src_path  = row['json_path']

    # Build unique filename
    orig_name = Path(src_path).name           # e.g. H07_20250918_2537826_HV3R_22.json
    fname = f"{row['sample_id']}__{row['primer']}__{row['run_folder']}__{orig_name}"

    # Truncate if too long (filesystem limit ~255 bytes)
    if len(fname) > 200:
        h = hashlib.md5(fname.encode()).hexdigest()[:8]
        fname = f"{row['sample_id']}__{row['primer']}__{h}.json"

    name_collisions[fname] += 1
    if name_collisions[fname] > 1:
        # Add counter suffix if collision
        fname = fname.replace('.json', f'_{name_collisions[fname]}.json')

    make_link(src_path, label_dir, fname)
    n_linked += 1

print(f"{'Symlinked' if USE_SYMLINKS else 'Copied'} {n_linked} files into dataset/")

# ──────────────────────────────────────────────
# STEP 6: Lưu manifests (train/test CSV)
# ──────────────────────────────────────────────
META_COLS = ['json_path', 'split', 'run_folder', 'sample_id', 'primer', 'label']
FEAT_COLS = [c for c in df.columns if c not in META_COLS + ['well', 'date']]

df_train.to_csv(f"{DATASET_DIR}/train_manifest.csv", index=False)
df_test.to_csv(f"{DATASET_DIR}/test_manifest.csv", index=False)
print(f"Saved train_manifest.csv and test_manifest.csv")

# ──────────────────────────────────────────────
# STEP 7: Lưu split_info.json (reproducibility)
# ──────────────────────────────────────────────
split_info = {
    "random_seed": RANDOM_SEED,
    "test_ratio": TEST_RATIO,
    "stratified_by": "has_error (profile-level)",
    "total_profiles": int(len(profile_df)),
    "train_profiles": len(train_set),
    "test_profiles":  len(test_set),
    "train_files": int(len(df_train)),
    "test_files":  int(len(df_test)),
    "train_error_files": int(df_train['label'].eq(1).sum()),
    "train_ok_files":    int(df_train['label'].eq(0).sum()),
    "test_error_files":  int(df_test['label'].eq(1).sum()),
    "test_ok_files":     int(df_test['label'].eq(0).sum()),
    "train_sample_ids":  sorted(train_set),
    "test_sample_ids":   sorted(test_set),
    "use_symlinks": USE_SYMLINKS,
    "source_features_csv": FEATURES_CSV,
}

with open(f"{DATASET_DIR}/split_info.json", 'w') as f:
    json.dump(split_info, f, indent=2, ensure_ascii=False)

print(f"Saved split_info.json")

# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print("\n" + "="*50)
print("DATASET SUMMARY")
print("="*50)
print(f"Location: {DATASET_DIR}/")
print(f"\nTrain ({len(df_train)} files, {len(train_set)} profiles):")
print(f"  label_1 (ERROR): {df_train['label'].eq(1).sum():4d} files")
print(f"  label_0 (OK):    {df_train['label'].eq(0).sum():4d} files")
print(f"\nTest ({len(df_test)} files, {len(test_set)} profiles):")
print(f"  label_1 (ERROR): {df_test['label'].eq(1).sum():4d} files")
print(f"  label_0 (OK):    {df_test['label'].eq(0).sum():4d} files")
print(f"\nFolder structure:")
print(f"  dataset/")
print(f"  ├── train/")
print(f"  │   ├── label_0/   ({df_train['label'].eq(0).sum()} symlinks)")
print(f"  │   └── label_1/   ({df_train['label'].eq(1).sum()} symlinks)")
print(f"  ├── test/")
print(f"  │   ├── label_0/   ({df_test['label'].eq(0).sum()} symlinks)")
print(f"  │   └── label_1/   ({df_test['label'].eq(1).sum()} symlinks)")
print(f"  ├── train_manifest.csv")
print(f"  ├── test_manifest.csv")
print(f"  └── split_info.json")
print("\nDone!")
