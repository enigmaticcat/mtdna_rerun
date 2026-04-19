"""
Tạo thư mục pipeline_results_subset/ chứa đúng các JSON files
được reference trong train/test manifest — dùng để upload lên server
thay vì toàn bộ pipeline_results/.

Dùng symlink (mặc định) để không tốn thêm disk space.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from config import TRAIN_CSV, TEST_CSV, BASE_DIR, OLD_BASE

USE_SYMLINKS = True   # False = copy thật (tốn disk)

SUBSET_DIR = os.path.join(BASE_DIR, "pipeline_results_subset")

# ── Load manifests ────────────────────────────────────────────────────────────
df = pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)], ignore_index=True)
df['json_path'] = df['json_path'].str.replace(OLD_BASE, BASE_DIR, regex=False)

paths = df['json_path'].dropna().unique()
print(f"JSON files cần thiết: {len(paths)}")

# ── Tạo subset ────────────────────────────────────────────────────────────────
ok, missing = 0, []

for src in paths:
    if not os.path.exists(src):
        missing.append(src)
        continue

    # giữ nguyên cấu trúc thư mục bên trong subset
    rel  = os.path.relpath(src, BASE_DIR)          # pipeline_results/run/sample/file.json
    dest = os.path.join(SUBSET_DIR, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        ok += 1
        continue

    if USE_SYMLINKS:
        os.symlink(os.path.abspath(src), dest)
    else:
        shutil.copy2(src, dest)
    ok += 1

print(f"  Created : {ok}")
print(f"  Missing : {len(missing)}")
if missing:
    print("  Missing files (first 10):")
    for p in missing[:10]:
        print(f"    {p}")

print(f"\nSubset saved to: {SUBSET_DIR}/")
print("Cấu trúc giữ nguyên pipeline_results/<run>/<sample>/<file>.json")
