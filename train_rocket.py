"""
ROCKET + Tabular hybrid classifier for mtDNA sequencing error detection.

MiniRocket transforms raw 4-channel traces into features,
concatenated with tabular features, then trained with XGBoost.

Requires: pip install sktime
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from config import TRAIN_CSV, TEST_CSV, OUTPUT_DIR, OLD_BASE, DATA_DIR

try:
    from sktime.transformations.panel.rocket import MiniRocket
except ImportError:
    raise ImportError("Chạy trước: pip install sktime")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_LEN   = 5000
N_KERNELS = 10_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: LOAD MANIFESTS
# ─────────────────────────────────────────────
print("Loading data...")
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

df_train['json_path'] = df_train['json_path'].str.replace(OLD_BASE, DATA_DIR, regex=False)
df_test['json_path']  = df_test['json_path'].str.replace(OLD_BASE, DATA_DIR, regex=False)

META_COLS    = ['json_path', 'split', 'run_folder', 'sample_id', 'primer',
                'label', 'well', 'date', 'primer_safe']
FEATURE_COLS = [c for c in df_train.columns if c not in META_COLS]

print(f"  Train: {len(df_train)} ({df_train['label'].sum()} errors)")
print(f"  Test:  {len(df_test)} ({df_test['label'].sum()} errors)")

# ─────────────────────────────────────────────
# STEP 2: TABULAR FEATURES
# ─────────────────────────────────────────────
le = LabelEncoder()
le.fit(pd.concat([df_train['primer'], df_test['primer']]).fillna('UNK'))

def get_tabular(df):
    X = df[FEATURE_COLS].values.astype(float)
    p = le.transform(df['primer'].fillna('UNK')).reshape(-1, 1)
    return np.hstack([X, p])

X_train_tab = get_tabular(df_train)
X_test_tab  = get_tabular(df_test)

imp = SimpleImputer(strategy='median')
X_train_tab = imp.fit_transform(X_train_tab)
X_test_tab  = imp.transform(X_test_tab)

y_train = df_train['label'].values.astype(int)
y_test  = df_test['label'].values.astype(int)
groups  = df_train['sample_id'].values

# ─────────────────────────────────────────────
# STEP 3: LOAD RAW TRACES
# ─────────────────────────────────────────────
def load_trace(path: str) -> np.ndarray:
    try:
        with open(path) as f:
            d = json.load(f)
        trace = np.stack([
            np.array(d['peakA'], dtype=np.float32),
            np.array(d['peakC'], dtype=np.float32),
            np.array(d['peakG'], dtype=np.float32),
            np.array(d['peakT'], dtype=np.float32),
        ], axis=0)
    except Exception:
        return np.zeros((4, MAX_LEN), dtype=np.float32)

    for i in range(4):
        mu, sigma = trace[i].mean(), trace[i].std()
        trace[i] = (trace[i] - mu) / (sigma + 1e-6)

    L = trace.shape[1]
    if L >= MAX_LEN:
        return trace[:, :MAX_LEN]
    pad = np.zeros((4, MAX_LEN - L), dtype=np.float32)
    return np.concatenate([trace, pad], axis=1)


print("\nLoading traces...")
traces_train = np.stack([load_trace(p) for p in df_train['json_path']])
traces_test  = np.stack([load_trace(p) for p in df_test['json_path']])
print(f"  Train: {traces_train.shape}  Test: {traces_test.shape}")

# ─────────────────────────────────────────────
# STEP 4: MINIROCKET TRANSFORM
# ─────────────────────────────────────────────
print(f"\nFitting MiniRocket ({N_KERNELS} kernels)...")
rocket = MiniRocket(num_kernels=N_KERNELS, random_state=42)
rocket.fit(traces_train)

print("Transforming traces...")
X_rocket_train = rocket.transform(traces_train)
X_rocket_test  = rocket.transform(traces_test)
print(f"  ROCKET features: {X_rocket_train.shape[1]}")

# Combine ROCKET + tabular
X_train = np.hstack([X_rocket_train, X_train_tab])
X_test  = np.hstack([X_rocket_test,  X_test_tab])
print(f"  Combined features: {X_train.shape[1]}")

# ─────────────────────────────────────────────
# STEP 5: CROSS-VALIDATION
# ─────────────────────────────────────────────
scale_pw = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
print(f"\nscale_pos_weight: {scale_pw:.2f}")

cv       = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_prob = np.full(len(y_train), -1.0)
aucs     = []

print("Running 5-fold CV...")
for fold, (tr_i, val_i) in enumerate(cv.split(X_train, y_train, groups)):
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.3,
        scale_pos_weight=scale_pw, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )
    m.fit(X_train[tr_i], y_train[tr_i],
          eval_set=[(X_train[val_i], y_train[val_i])], verbose=False)

    p = m.predict_proba(X_train[val_i])[:, 1]
    oof_prob[val_i] = p
    auc = roc_auc_score(y_train[val_i], p)
    aucs.append(auc)
    print(f"  Fold {fold+1}: AUC={auc:.4f}")

print(f"  Mean CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# ─────────────────────────────────────────────
# STEP 6: THRESHOLD (recall=1 constraint)
# ─────────────────────────────────────────────
best_thr, best_pre = 0.01, 0.0

print("\nOOF threshold scan (recall=1 candidates):")
print(f"  {'thr':>6}  {'recall':>6}  {'prec':>6}  {'FP':>5}  {'FN':>5}")
for thr in np.arange(0.01, 0.60, 0.01):
    preds = (oof_prob >= thr).astype(int)
    rec = recall_score(y_train, preds, zero_division=0)
    pre = precision_score(y_train, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
    if rec >= 1.0:
        print(f"  {thr:>6.2f}  {rec:>6.3f}  {pre:>6.3f}  {fp:>5}  {fn:>5}  ←")
        if pre > best_pre:
            best_pre = pre
            best_thr = thr

print(f"\nBest threshold: {best_thr:.2f}  (OOF precision={best_pre:.3f})")

# ─────────────────────────────────────────────
# STEP 7: FINAL MODEL + EVALUATION
# ─────────────────────────────────────────────
print("\nTraining final model on all train data...")
final_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.3,
    scale_pos_weight=scale_pw, eval_metric='logloss',
    random_state=42, n_jobs=-1, verbosity=0,
)
final_model.fit(X_train, y_train)

test_probs = final_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_thr).astype(int)
test_auc   = roc_auc_score(y_test, test_probs)

print("\n" + "="*52)
print("TEST SET EVALUATION")
print("="*52)
print(f"ROC-AUC  : {test_auc:.4f}")
print(f"Threshold: {best_thr:.2f}")
print(classification_report(y_test, test_preds, target_names=['OK', 'ERROR']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_preds))

error_probs = test_probs[y_test == 1]
print(f"\nError probability distribution (n={len(error_probs)}):")
for c in [0.1, 0.2, 0.3, 0.5]:
    n = (error_probs < c).sum()
    print(f"  prob < {c}: {n}/{len(error_probs)} ({100*n/len(error_probs):.1f}%)")

hard_mask = (y_test == 1) & (test_probs < 0.3)
if hard_mask.sum() > 0:
    hdf = df_test[hard_mask].copy()
    hdf['prob'] = test_probs[hard_mask]
    print(f"\nHard cases (prob < 0.3): {hard_mask.sum()}")
    print(hdf[['sample_id', 'primer', 'run_folder', 'prob']].sort_values('prob').to_string(index=False))
else:
    print("\nKhông có hard cases (prob < 0.3).")

# ─────────────────────────────────────────────
# STEP 8: SAVE
# ─────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "classifier_rocket.pkl"), 'wb') as f:
    pickle.dump({
        'model':         final_model,
        'rocket':        rocket,
        'imputer':       imp,
        'label_encoder': le,
        'feature_cols':  FEATURE_COLS,
        'threshold':     best_thr,
        'test_auc':      test_auc,
    }, f)

print(f"\n✓ Saved classifier_rocket.pkl to {OUTPUT_DIR}/")
print("Done!")
