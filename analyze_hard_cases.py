"""
Phân tích hard cases: error samples bị model cho probability thấp.
Mục tiêu: hiểu tại sao model không nhận ra chúng là lỗi.
"""

import pickle
import numpy as np
import pandas as pd
from config import TEST_CSV, TRAIN_CSV, OUTPUT_DIR, METADATA_TSV, DATA_DIR, OLD_BASE, BASE_DIR
import os

PROB_THRESHOLD = 0.3   # hard case nếu prob < giá trị này
MODEL_PKL = os.path.join(OUTPUT_DIR, "classifier_v2.pkl")

# ── Load model và data ────────────────────────────────────────────────────────
with open(MODEL_PKL, 'rb') as f:
    bundle = pickle.load(f)

model    = bundle['model']
imp      = bundle['imputer']
le       = bundle['label_encoder']
feat_cols = bundle['feature_cols']

df_test  = pd.read_csv(TEST_CSV)
df_train = pd.read_csv(TRAIN_CSV)
df_test['json_path']  = df_test['json_path'].str.replace(OLD_BASE, DATA_DIR, regex=False)
df_train['json_path'] = df_train['json_path'].str.replace(OLD_BASE, DATA_DIR, regex=False)

META_COLS    = ['json_path', 'run_folder', 'sample_id', 'primer', 'label', 'split', 'well', 'date', 'primer_safe']
FEATURE_COLS = [c for c in df_test.columns if c not in META_COLS]

def prepare_X(df):
    X_raw = df[FEATURE_COLS].values.astype(float)
    p     = le.transform(df['primer'].fillna('UNK')).reshape(-1, 1)
    return imp.transform(np.hstack([X_raw, p]))

X_test  = prepare_X(df_test)
X_train = prepare_X(df_train)

test_probs  = model.predict_proba(X_test)[:, 1]
train_probs = model.predict_proba(X_train)[:, 1]

df_test['prob_error']  = test_probs
df_train['prob_error'] = train_probs

# ── Xác định hard cases ───────────────────────────────────────────────────────
hard = df_test[(df_test['label'] == 1) & (df_test['prob_error'] < PROB_THRESHOLD)].copy()
easy = df_test[(df_test['label'] == 1) & (df_test['prob_error'] >= PROB_THRESHOLD)].copy()

print(f"Hard cases (prob < {PROB_THRESHOLD}): {len(hard)}")
print(f"Easy errors (prob >= {PROB_THRESHOLD}): {len(easy)}")

# ── Tra metadata: loại lỗi của từng hard case ────────────────────────────────
try:
    meta = pd.read_csv(METADATA_TSV, sep='\t')
    meta.columns = meta.columns.str.strip()

    # tìm cột sample id và issues
    id_col     = next((c for c in meta.columns if 'sample' in c.lower() and 'id' in c.lower()), meta.columns[1])
    issue_cols = [c for c in meta.columns if 'issue' in c.lower() or 'require' in c.lower()]

    print(f"\nMetadata — dùng id_col='{id_col}', issue_cols={issue_cols}")
    meta[id_col] = meta[id_col].astype(str).str.strip()

    print("\n" + "="*70)
    print("HARD CASES — CHI TIẾT METADATA")
    print("="*70)
    for _, row in hard.sort_values('prob_error').iterrows():
        sid = str(row['sample_id']).strip()
        match = meta[meta[id_col] == sid]
        issues = ""
        if not match.empty:
            issues = " | ".join(
                str(match.iloc[0][c]) for c in issue_cols if pd.notna(match.iloc[0][c])
            )
        print(f"\n  {sid} / {row['primer']} / {row['run_folder']}")
        print(f"  prob={row['prob_error']:.4f}")
        print(f"  Issues: {issues if issues else 'NOT FOUND in metadata'}")
except Exception as e:
    print(f"Không load được metadata: {e}")

# ── So sánh features: hard vs easy errors ────────────────────────────────────
print("\n" + "="*70)
print("FEATURE COMPARISON: hard cases vs typical errors")
print("="*70)

hard_feat  = df_test.loc[hard.index, FEATURE_COLS]
easy_feat  = df_test.loc[easy.index, FEATURE_COLS]
all_err_feat = df_test.loc[df_test['label']==1, FEATURE_COLS]

comparison = pd.DataFrame({
    'hard_mean':  hard_feat.mean(),
    'easy_mean':  easy_feat.mean(),
    'diff':       hard_feat.mean() - easy_feat.mean(),
    'diff_pct':   ((hard_feat.mean() - easy_feat.mean()) / (easy_feat.mean().abs() + 1e-9) * 100),
}).sort_values('diff_pct', key=abs, ascending=False)

print("\nTop features khác biệt nhất (hard vs easy errors):")
print(f"  {'feature':<25} {'hard_mean':>10} {'easy_mean':>10} {'diff%':>8}")
for feat, row in comparison.head(15).iterrows():
    print(f"  {feat:<25} {row['hard_mean']:>10.3f} {row['easy_mean']:>10.3f} {row['diff_pct']:>7.1f}%")

# ── Phân phối primer trong hard cases ────────────────────────────────────────
print("\n" + "="*70)
print("PRIMER DISTRIBUTION")
print("="*70)
print("\nHard cases:")
print(hard['primer'].value_counts().to_string())
print("\nAll errors (test):")
print(df_test[df_test['label']==1]['primer'].value_counts().to_string())

# ── Hard cases trong train set ────────────────────────────────────────────────
train_hard = df_train[(df_train['label'] == 1) & (df_train['prob_error'] < PROB_THRESHOLD)]
print(f"\n" + "="*70)
print(f"TRAIN SET hard cases (prob < {PROB_THRESHOLD}): {len(train_hard)}/{(df_train['label']==1).sum()}")
print(f"  ({100*len(train_hard)/(df_train['label']==1).sum():.1f}% của tổng error trong train)")
if len(train_hard):
    print("\nPrimer breakdown:")
    print(train_hard['primer'].value_counts().to_string())
