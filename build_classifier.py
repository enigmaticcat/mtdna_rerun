"""
mtDNA Sequencing Error Classifier
==================================
Binary classifier: given a Tracy JSON file, predict if that primer's sequencing has an error.

Label logic:
 - From metadata_rerun.tsv, extract which primers are mentioned in Issues/Requirement
   as having problems → those JSON files = label 1.
 - All other JSON files of the same sample (non-bad primers) = label 0.
 - Files where Issues = "None" → all primer files for that sample = label 0.
"""

import os
import re
import json
import csv
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PIPELINE_DIR = "/home/minhtq/mtDNA_proj/mtdna_rerun/pipeline_results"
METADATA_TSV = "/home/minhtq/mtDNA_proj/mtdna_rerun/metadata_rerun.tsv"
OUTPUT_DIR   = "/home/minhtq/mtDNA_proj/mtdna_rerun/classifier_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIGNAL_THRESHOLD = 5

# ─────────────────────────────────────────────
# PRIMER KEYWORD → CANONICAL TOKEN
# ─────────────────────────────────────────────
PRIMER_PATTERNS = [
    (r'\b(?:HV1F|1F)\b',  'HV1F'),
    (r'\b(?:HV1R|1R)\b',  'HV1R'),
    (r'\b(?:HV2F|2F)\b',  'HV2F'),
    (r'\b(?:HV2R|2R)\b',  'HV2R'),
    (r'\b(?:HV3F|3F)\b',  'HV3F'),
    (r'\b(?:HV3R|3R)\b',  'HV3R'),
    (r'\bR389\b',          'R389'),
    (r'\bF16190\b',        'HV1-16190F'),
    (r'\bR16258\b',        'HV1-16258R'),
    (r'\bF109\b',          'HV2-109F'),
    (r'\bR285a?\b',        'HV2-285aR'),
    (r'\bF15\b',           'HV2-6F'),   # approx; F15 is alternate primer
]


def extract_primers(text: str) -> set:
    found = set()
    if not text or text.strip().lower() in ('none', '', '-', 'n/a'):
        return found
    for pattern, token in PRIMER_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.add(token)
    return found


# ─────────────────────────────────────────────
# STEP 1: BUILD (sample_id, primer) → label
# ─────────────────────────────────────────────

def build_labels(metadata_tsv: str):
    """
    Parse metadata_rerun.tsv to determine:
      bad_keys  : set of (sample_id, primer_token) that are confirmed errors (label 1)
      ok_samples: set of sample_id that have at least one run recorded as Issues=None
                  → all primer files in that sample that are NOT in bad_keys → label 0
    """
    bad_keys   = set()   # (sample_id, primer) → label 1
    ok_samples = set()   # sample_id where at least one run has Issues=None → label 0 eligible

    with open(metadata_tsv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header

        for row in reader:
            if len(row) < 5:
                continue
            sample_id = row[1].strip()
            if not sample_id or sample_id in ('Sample', '1387'):
                continue

            run_blocks = [
                (2, 3, 4),   # Run1 folder, Issues, Requirement
                (5, 6, 7),   # Run2
                (8, 9, 10),  # Run3
            ]

            for ri, ii, qi in run_blocks:
                run_val    = row[ri].strip() if len(row) > ri else ''
                issues_val = row[ii].strip() if len(row) > ii else ''
                req_val    = row[qi].strip() if len(row) > qi else ''
                if not run_val:
                    continue

                is_none = issues_val.lower() in ('none', '', '-', 'n/a')

                if is_none:
                    ok_samples.add(sample_id)
                    continue

                # Try to extract bad primers from Issues first
                bad = extract_primers(issues_val)
                if not bad:
                    # Fallback: Requirement tells which primers to redo
                    bad = extract_primers(req_val)
                if not bad:
                    continue  # couldn't determine primer, skip

                for p in bad:
                    bad_keys.add((sample_id, p))

    return bad_keys, ok_samples


print("Building labels from metadata...")
bad_keys, ok_samples = build_labels(METADATA_TSV)
print(f"  Bad (sample, primer) pairs: {len(bad_keys)}")
print(f"  Samples with at least one 'None' run: {len(ok_samples)}")


# ─────────────────────────────────────────────
# STEP 2: SCAN JSON FILES & ASSIGN LABELS
# ─────────────────────────────────────────────

JSON_FNAME_RE = re.compile(
    r'(?P<well>[A-H]\d{2})_'
    r'(?P<date>\d{8})_'
    r'(?P<sample>[^_]+)_'
    r'(?P<primer>[^_]+)_'
    r'(?P<idx>\d+)\.json$',
    re.IGNORECASE
)


def parse_fname(fname: str):
    m = JSON_FNAME_RE.search(fname)
    return m.groupdict() if m else None


print("Scanning JSON files...")
records = []

for run_dir in sorted(os.scandir(PIPELINE_DIR), key=lambda e: e.name):
    if not run_dir.is_dir():
        continue
    for samp_dir in sorted(os.scandir(run_dir.path), key=lambda e: e.name):
        if not samp_dir.is_dir():
            continue
        for jf in sorted(Path(samp_dir.path).glob("*.json")):
            p = parse_fname(jf.name)
            if not p:
                continue

            sample_id = p['sample']
            primer    = p['primer']
            key       = (sample_id, primer)

            if key in bad_keys:
                label = 1
            elif sample_id in ok_samples:
                # This sample had runs marked "None" → non-bad primers are OK
                label = 0
            else:
                # No label information → skip
                label = -1

            records.append({
                'json_path':  str(jf),
                'run_folder': run_dir.name,
                'sample_id':  sample_id,
                'primer':     primer,
                'well':       p['well'],
                'date':       p['date'],
                'label':      label,
            })

df_records = pd.DataFrame(records)
print(f"  Total JSON files: {len(df_records)}")
print(f"  Label distribution:\n{df_records['label'].value_counts().to_string()}")

df_labeled = df_records[df_records['label'].isin([0, 1])].copy()
print(f"\n  Labeled files: {len(df_labeled)}"
      f" (pos={df_labeled['label'].eq(1).sum()}, neg={df_labeled['label'].eq(0).sum()})")

df_records.to_csv(f"{OUTPUT_DIR}/records_raw.csv", index=False)


# ─────────────────────────────────────────────
# STEP 3: FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features(json_path: str) -> dict | None:
    feats = {}
    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception:
        return None

    try:
        A = np.array(d['peakA'], dtype=float)
        C = np.array(d['peakC'], dtype=float)
        G = np.array(d['peakG'], dtype=float)
        T = np.array(d['peakT'], dtype=float)
    except KeyError:
        return None

    n = len(A)
    if n < 20:
        return None

    sig     = np.stack([A, C, G, T], axis=1)
    primary = sig.max(axis=1)
    secondary = np.sort(sig, axis=1)[:, -2]

    # Global signal stats
    feats['signal_mean'] = float(primary.mean())
    feats['signal_max']  = float(primary.max())
    feats['signal_std']  = float(primary.std())
    feats['signal_cv']   = float(primary.std() / (primary.mean() + 1e-6))

    # Per-channel baselines (bottom 10th percentile)
    feats['baseline']    = float(np.percentile(primary, 10))
    for ch, arr in zip('ACGT', [A, C, G, T]):
        feats[f'baseline_{ch}'] = float(np.percentile(arr, 10))

    # Coverage / readable length
    readable = primary > SIGNAL_THRESHOLD
    feats['coverage_len']  = float(readable.sum())
    feats['coverage_frac'] = float(readable.mean())
    if readable.any():
        s = int(np.argmax(readable))
        e = int(len(readable) - np.argmax(readable[::-1]) - 1)
        feats['read_start']       = s / n
        feats['read_end']         = e / n
        feats['read_length_frac'] = (e - s) / n
    else:
        feats['read_start'] = feats['read_end'] = feats['read_length_frac'] = 0.0

    # Dropoff position
    thr = primary.max() * 0.2
    above = primary > thr
    feats['dropoff_pos'] = float(np.where(above)[0][-1]) / n if above.any() else 0.0

    # Secondary peak ratio (mixed signal)
    sr = secondary / (primary + 1e-6)
    feats['sec_ratio_mean']  = float(sr.mean())
    feats['sec_ratio_max']   = float(sr.max())
    feats['sec_ratio_p75']   = float(np.percentile(sr, 75))
    feats['sec_ratio_p90']   = float(np.percentile(sr, 90))
    feats['n_mixed_pos']     = float((sr > 0.3).sum())
    feats['frac_mixed_pos']  = float((sr > 0.3).mean())

    # Dyeblob: isolated spike at start or end
    e5 = max(int(n * 0.05), 10)
    mid_mean = primary[e5:-e5].mean() + 1e-6
    feats['early_max_ratio'] = float(primary[:e5].max() / mid_mean)
    feats['late_max_ratio']  = float(primary[-e5:].max() / mid_mean)

    # Signal in quartile windows
    for qname, qs, qe in [
        ('q1', int(n*0.25), int(n*0.40)),
        ('q2', int(n*0.40), int(n*0.55)),
        ('q3', int(n*0.55), int(n*0.75)),
    ]:
        seg = primary[qs:qe]
        sec_seg = secondary[qs:qe]
        feats[f'{qname}_mean']     = float(seg.mean()) if len(seg) else 0.0
        feats[f'{qname}_std']      = float(seg.std()) if len(seg) else 0.0
        feats[f'{qname}_sec_mean'] = float(sec_seg.mean()) if len(sec_seg) else 0.0

    # Readable region uniformity
    if readable.sum() > 10:
        rs = primary[readable]
        feats['readable_cv']    = float(rs.std() / (rs.mean() + 1e-6))
        feats['readable_p10']   = float(np.percentile(rs, 10))
        feats['readable_p90']   = float(np.percentile(rs, 90))
        feats['readable_range'] = float((feats['readable_p90'] - feats['readable_p10'])
                                        / (feats['signal_max'] + 1e-6))
    else:
        feats['readable_cv'] = feats['readable_p10'] = feats['readable_p90'] = feats['readable_range'] = 0.0

    # Basecall quality
    quals = np.array(d.get('basecallQual') or [], dtype=float)
    if len(quals) > 0:
        feats['qual_mean']            = float(quals.mean())
        feats['qual_p10']             = float(np.percentile(quals, 10))
        feats['qual_below20_frac']    = float((quals < 20).mean())
        feats['qual_below40_frac']    = float((quals < 40).mean())
        feats['n_basecalls']          = float(len(quals))
    else:
        feats['qual_mean'] = feats['qual_p10'] = np.nan
        feats['qual_below20_frac'] = feats['qual_below40_frac'] = np.nan
        feats['n_basecalls'] = 0.0

    # Alignment & allele fraction
    feats['align1score']  = float(d.get('align1score') or np.nan)
    feats['align2score']  = float(d.get('align2score') or np.nan)
    feats['allele1frac']  = float(d.get('allele1fraction') or np.nan)
    feats['allele2frac']  = float(d.get('allele2fraction') or np.nan)

    return feats


print("\nExtracting features...")
feature_rows = []
for _, row in df_labeled.iterrows():
    feats = extract_features(row['json_path'])
    if feats is None:
        continue
    feats.update({
        'json_path':  row['json_path'],
        'run_folder': row['run_folder'],
        'sample_id':  row['sample_id'],
        'primer':     row['primer'],
        'label':      row['label'],
    })
    feature_rows.append(feats)
    if len(feature_rows) % 200 == 0:
        print(f"  {len(feature_rows)} files processed...")

df_feat = pd.DataFrame(feature_rows)
print(f"\nFeature extraction complete: {len(df_feat)} files, {len(df_feat.columns)} columns")
print(f"Label distribution:\n{df_feat['label'].value_counts().to_string()}")
df_feat.to_csv(f"{OUTPUT_DIR}/features.csv", index=False)


# ─────────────────────────────────────────────
# STEP 4: TRAIN / EVALUATE XGBOOST
# ─────────────────────────────────────────────

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb

META_COLS = ['json_path', 'run_folder', 'sample_id', 'primer', 'label']
FEAT_COLS = [c for c in df_feat.columns if c not in META_COLS]

le = LabelEncoder()
primer_enc = le.fit_transform(df_feat['primer'].fillna('UNK')).reshape(-1, 1)

X = np.hstack([df_feat[FEAT_COLS].values.astype(float), primer_enc])
y = df_feat['label'].values.astype(int)
groups = df_feat['run_folder'].values
FEAT_COLS = FEAT_COLS + ['primer_enc']

imp = SimpleImputer(strategy='median')
X = imp.fit_transform(X)

scale_pw = float((y == 0).sum()) / max((y == 1).sum(), 1)
print(f"\nClass balance: pos={y.sum()}, neg={(y==0).sum()}, scale_pos_weight={scale_pw:.2f}")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.full(len(y), -1.0)
fold_aucs = []

print("\n── Cross-Validation ──")
for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y, groups)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    if y_val.sum() == 0 or (y_val == 0).sum() == 0:
        print(f"  Fold {fold+1}: only one class in val → skip")
        continue

    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    preds = m.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = preds
    auc = roc_auc_score(y_val, preds)
    fold_aucs.append(auc)
    print(f"  Fold {fold+1}: ROC-AUC={auc:.4f}  pos={y_val.sum()} neg={(y_val==0).sum()}")

# OOF metrics
mask = oof_preds >= 0
if mask.sum() > 0 and len(fold_aucs) > 0:
    oof_auc = roc_auc_score(y[mask], oof_preds[mask])
    print(f"\nOOF ROC-AUC: {oof_auc:.4f}")
    print(f"Mean fold AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    best_f1, best_thresh = 0, 0.5
    for thr in np.arange(0.1, 0.91, 0.05):
        pb = (oof_preds[mask] >= thr).astype(int)
        f1 = f1_score(y[mask], pb, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thr

    print(f"Best threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    pb = (oof_preds[mask] >= best_thresh).astype(int)
    print("\nClassification Report:")
    print(classification_report(y[mask], pb, target_names=['OK', 'ERROR'], zero_division=0))
    print("Confusion Matrix (rows=actual, cols=pred):")
    print(confusion_matrix(y[mask], pb))
else:
    best_thresh = 0.5
    print("WARNING: not enough folds to compute OOF metrics")


# ─────────────────────────────────────────────
# STEP 5: FINAL MODEL + FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\nTraining final model on all labeled data...")
final = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pw,
    eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0,
)
final.fit(X, y)

imp_df = pd.DataFrame({'feature': FEAT_COLS, 'importance': final.feature_importances_})
imp_df = imp_df.sort_values('importance', ascending=False)
print("\nTop 20 features:")
print(imp_df.head(20).to_string(index=False))
imp_df.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)

with open(f"{OUTPUT_DIR}/classifier.pkl", 'wb') as f:
    pickle.dump({
        'model': final, 'imputer': imp,
        'label_encoder': le, 'feature_cols': FEAT_COLS,
        'threshold': best_thresh,
    }, f)

print(f"\n✓ Saved: {OUTPUT_DIR}/classifier.pkl")
print(f"✓ Saved: {OUTPUT_DIR}/feature_importance.csv")
print(f"✓ Saved: {OUTPUT_DIR}/features.parquet")
print("Done!")
