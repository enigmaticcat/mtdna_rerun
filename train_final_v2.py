"""
mtDNA Sequencing Error Classifier (v2)
=====================================
Training script utilizing the profile-based train/test split.

Features:
- Loads data specifically from dataset/train_manifest.csv and dataset/test_manifest.csv.
- Uses StratifiedGroupKFold on train set (grouped by sample_id).
- Trains final XGBoost model on all train data.
- Performs final evaluation on the isolated test set.
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,
                              confusion_matrix, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = "/home/minhtq/mtDNA_proj/mtdna_rerun"
TRAIN_CSV = os.path.join(BASE_DIR, "dataset/train_manifest.csv")
TEST_CSV  = os.path.join(BASE_DIR, "dataset/test_manifest.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "classifier_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

META_COLS = ['json_path', 'split', 'run_folder', 'sample_id', 'primer', 'label', 'well', 'date']
FEATURE_COLS = [c for c in df_train.columns if c not in META_COLS and c != 'primer_safe']

print(f"  Train: {len(df_train)} files ({df_train['label'].sum()} errors)")
print(f"  Test:  {len(df_test)} files ({df_test['label'].sum()} errors)")
print(f"  Features: {len(FEATURE_COLS)}")

# Encode primers
le = LabelEncoder()
all_primers = pd.concat([df_train['primer'], df_test['primer']]).fillna('UNK')
le.fit(all_primers)

def prepare_xy(df):
    X_raw = df[FEATURE_COLS].values.astype(float)
    p_enc = le.transform(df['primer'].fillna('UNK')).reshape(-1, 1)
    X = np.hstack([X_raw, p_enc])
    y = df['label'].values.astype(int)
    return X, y

X_train, y_train = prepare_xy(df_train)
X_test, y_test   = prepare_xy(df_test)
train_groups     = df_train['sample_id'].values
final_feat_cols  = FEATURE_COLS + ['primer_enc']

# Impute NaNs
imp = SimpleImputer(strategy='median')
X_train = imp.fit_transform(X_train)
X_test  = imp.transform(X_test)

# Class weight
scale_pw = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
print(f"  scale_pos_weight: {scale_pw:.2f}")

# ─────────────────────────────────────────────
# STEP 2: CROSS-VALIDATION (ON TRAIN SET)
# ─────────────────────────────────────────────
print("\nRunning Cross-Validation on Train Set (Stratified by Sample)...")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.full(len(y_train), -1.0)
fold_aucs = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train, train_groups)):
    X_tr_f, X_val_f = X_train[tr_idx], X_train[val_idx]
    y_tr_f, y_val_f = y_train[tr_idx], y_train[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pw, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_tr_f, y_tr_f, eval_set=[(X_val_f, y_val_f)], verbose=False)

    probs = model.predict_proba(X_val_f)[:, 1]
    oof_probs[val_idx] = probs
    auc = roc_auc_score(y_val_f, probs)
    fold_aucs.append(auc)
    print(f"  Fold {fold+1}: AUC = {auc:.4f}")

print(f"  Mean CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# ─────────────────────────────────────────────
# STEP 3: FINAL TRAINING
# ─────────────────────────────────────────────
print("\nTraining final model on full train set...")
final_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pw, eval_metric='logloss',
    random_state=42, n_jobs=-1, verbosity=0
)
final_model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# STEP 4: EVALUATION ON UNSEEN TEST SET
# ─────────────────────────────────────────────
print("\n" + "="*52)
print("INDEPENDENT TEST SET EVALUATION")
print("="*52)

test_probs = final_model.predict_proba(X_test)[:, 1]
test_auc   = roc_auc_score(y_test, test_probs)

# Find best threshold on OOF to avoid leakage
best_f1, best_thr = 0, 0.5
for thr in np.arange(0.1, 0.9, 0.05):
    f1 = f1_score(y_train, (oof_probs >= thr).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

test_preds = (test_probs >= best_thr).astype(int)

print(f"ROC-AUC  : {test_auc:.4f}")
print(f"Best Thr : {best_thr:.2f} (from OOF)")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, test_preds, target_names=['OK', 'ERROR']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_preds))

# ─────────────────────────────────────────────
# STEP 5: SAVE MODEL AND ASSETS
# ─────────────────────────────────────────────
importance = pd.DataFrame({
    'feature': final_feat_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_v2.csv"), index=False)

with open(os.path.join(OUTPUT_DIR, "classifier_v2.pkl"), 'wb') as f:
    pickle.dump({
        'model': final_model,
        'imputer': imp,
        'label_encoder': le,
        'feature_cols': final_feat_cols,
        'threshold': best_thr,
        'test_auc': test_auc
    }, f)

print(f"\n✓ Saved model to {OUTPUT_DIR}/classifier_v2.pkl")
print(f"✓ Saved features to {OUTPUT_DIR}/feature_importance_v2.csv")
print("Done!")
