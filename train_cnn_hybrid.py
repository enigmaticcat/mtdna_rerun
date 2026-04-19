"""
Hybrid 1D CNN + Tabular classifier for mtDNA sequencing error detection.

Branches:
  - CNN  : 1D convolutions on raw 4-channel trace (peakA/C/G/T), 5000 timepoints
  - Tab  : MLP on 46 extracted features + primer encoding
  - Head : concat → FC → Dropout → logit
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                              confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from config import TRAIN_CSV, TEST_CSV, OUTPUT_DIR, OLD_BASE, DATA_DIR

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN    = 5000
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3
PATIENCE   = 10
DROPOUT    = 0.3
TAB_DIM    = 64

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")

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

print(f"  Train: {len(df_train)} files ({df_train['label'].sum()} errors)")
print(f"  Test:  {len(df_test)} files ({df_test['label'].sum()} errors)")
print(f"  Tabular features: {len(FEATURE_COLS)}")

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
n_tab   = X_train_tab.shape[1]

# ─────────────────────────────────────────────
# STEP 3: LOAD RAW TRACES
# ─────────────────────────────────────────────
def load_trace(path: str) -> np.ndarray:
    """Load 4-channel trace → z-score normalize → pad/truncate to MAX_LEN."""
    try:
        with open(path) as f:
            d = json.load(f)
        trace = np.stack([
            np.array(d['peakA'], dtype=np.float32),
            np.array(d['peakC'], dtype=np.float32),
            np.array(d['peakG'], dtype=np.float32),
            np.array(d['peakT'], dtype=np.float32),
        ], axis=0)  # (4, L)
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


print("\nLoading traces into memory...")
traces_train = np.stack([load_trace(p) for p in df_train['json_path']])
traces_test  = np.stack([load_trace(p) for p in df_test['json_path']])
print(f"  Train: {traces_train.shape}  Test: {traces_test.shape}")

# ─────────────────────────────────────────────
# STEP 4: DATASET & MODEL
# ─────────────────────────────────────────────
class TraceDataset(Dataset):
    def __init__(self, traces, tabular, labels):
        self.traces  = torch.tensor(traces,  dtype=torch.float32)
        self.tabular = torch.tensor(tabular, dtype=torch.float32)
        self.labels  = torch.tensor(labels,  dtype=torch.float32)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return self.traces[i], self.tabular[i], self.labels[i]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k//2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class HybridCNN(nn.Module):
    def __init__(self, n_tab, dropout=DROPOUT):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(4,   32, 15), nn.MaxPool1d(4),
            ConvBlock(32,  64,  7), nn.MaxPool1d(4),
            ConvBlock(64, 128,  5), nn.MaxPool1d(4),
            ConvBlock(128,128,  3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.tab = nn.Sequential(
            nn.Linear(n_tab, TAB_DIM),
            nn.BatchNorm1d(TAB_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + TAB_DIM, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, trace, tab):
        c = self.cnn(trace).squeeze(-1)
        t = self.tab(tab)
        return self.head(torch.cat([c, t], dim=1)).squeeze(-1)


# ─────────────────────────────────────────────
# STEP 5: TRAIN / EVAL HELPERS
# ─────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum = 0
    for tr, tb, lb in loader:
        tr, tb, lb = tr.to(DEVICE), tb.to(DEVICE), lb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(tr, tb), lb)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(lb)
    return loss_sum / len(loader.dataset)


@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    out = []
    for tr, tb, _ in loader:
        out.append(torch.sigmoid(model(tr.to(DEVICE), tb.to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def fit(traces, tabular, labels, pos_w):
    ds  = TraceDataset(traces, tabular, labels)
    ldr = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model     = HybridCNN(tabular.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_loss, best_state, wait = float('inf'), None, 0
    for _ in range(EPOCHS):
        loss = run_epoch(model, ldr, optimizer, criterion)
        scheduler.step(loss)
        if loss < best_loss:
            best_loss  = loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────
# STEP 6: CROSS-VALIDATION
# ─────────────────────────────────────────────
pos_w = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
print(f"\npos_weight: {pos_w:.2f}")
print("Running 5-fold CV...")

cv       = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_prob = np.full(len(y_train), -1.0)
aucs     = []

for fold, (tr_i, val_i) in enumerate(cv.split(X_train_tab, y_train, groups)):
    m = fit(traces_train[tr_i], X_train_tab[tr_i], y_train[tr_i], pos_w)

    val_ds  = TraceDataset(traces_train[val_i], X_train_tab[val_i], y_train[val_i])
    val_ldr = DataLoader(val_ds, batch_size=BATCH_SIZE)
    p = get_probs(m, val_ldr)

    oof_prob[val_i] = p
    auc = roc_auc_score(y_train[val_i], p)
    aucs.append(auc)
    print(f"  Fold {fold+1}: AUC={auc:.4f}")

print(f"  Mean CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# ─────────────────────────────────────────────
# STEP 7: THRESHOLD (recall=1 constraint)
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
# STEP 8: FINAL MODEL
# ─────────────────────────────────────────────
print("\nTraining final model on all train data...")
final_model = fit(traces_train, X_train_tab, y_train, pos_w)

# ─────────────────────────────────────────────
# STEP 9: TEST SET EVALUATION
# ─────────────────────────────────────────────
test_ds  = TraceDataset(traces_test, X_test_tab, y_test)
test_ldr = DataLoader(test_ds, batch_size=BATCH_SIZE)
test_probs = get_probs(final_model, test_ldr)
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

# Hard cases
hard_mask = (y_test == 1) & (test_probs < 0.3)
if hard_mask.sum() > 0:
    hdf = df_test[hard_mask].copy()
    hdf['prob'] = test_probs[hard_mask]
    print(f"\nHard cases (error, prob < 0.3): {hard_mask.sum()}")
    print(hdf[['sample_id', 'primer', 'run_folder', 'prob']].sort_values('prob').to_string(index=False))
else:
    print("\nNo hard cases (prob < 0.3) — model phân biệt tốt.")

# ─────────────────────────────────────────────
# STEP 10: SAVE
# ─────────────────────────────────────────────
torch.save(final_model.state_dict(), os.path.join(OUTPUT_DIR, "classifier_cnn.pt"))
with open(os.path.join(OUTPUT_DIR, "classifier_cnn_meta.pkl"), 'wb') as f:
    pickle.dump({
        'imputer':      imp,
        'label_encoder': le,
        'feature_cols': FEATURE_COLS,
        'n_tab':        n_tab,
        'threshold':    best_thr,
        'test_auc':     test_auc,
    }, f)

print(f"\n✓ Saved classifier_cnn.pt + classifier_cnn_meta.pkl to {OUTPUT_DIR}/")
print("Done!")
