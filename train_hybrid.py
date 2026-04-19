"""
mtDNA Sequencing Error Classifier — Hybrid MatchboxNet + Tabular
================================================================
Architecture:
  - MatchboxNet branch: raw 4-channel fluorescent trace (peakA/C/G/T)
  - Tabular MLP branch: 45 engineered features
  - Primer embedding layer
  - Fusion head: concat → FC → sigmoid

Same CV strategy and threshold selection as train_final_v2.py for
direct comparison of results.
"""

import os, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from config_hybrid import (
    OLD_BASE, BASE_DIR, DEVICE as CFG_DEVICE,
    MAX_LEN, BATCH_SIZE, EPOCHS, LR, PATIENCE,
    TRACE_EMBED_DIM, TAB_EMBED_DIM, PRIMER_EMBED_DIM, DROPOUT,
)

# ─────────────────────────────────────────────
# PATHS  (derived from config)
# ─────────────────────────────────────────────
TRAIN_CSV  = os.path.join(BASE_DIR, "dataset/train_manifest.csv")
TEST_CSV   = os.path.join(BASE_DIR, "dataset/test_manifest.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "classifier_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# honour config device but fall back gracefully
if CFG_DEVICE == "mps" and not torch.backends.mps.is_available():
    DEVICE = "cpu"
elif CFG_DEVICE == "cuda" and not torch.cuda.is_available():
    DEVICE = "cpu"
else:
    DEVICE = CFG_DEVICE
print(f"Device: {DEVICE}")

META_COLS = ['json_path', 'run_folder', 'sample_id', 'primer',
             'label', 'split', 'well', 'date', 'primer_safe']

# ─────────────────────────────────────────────
# STEP 1: LOAD METADATA
# ─────────────────────────────────────────────
print("Loading manifests...")
df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

# remap paths if manifests were built on a different machine
for df in [df_train, df_test]:
    df['json_path'] = df['json_path'].str.replace(OLD_BASE, BASE_DIR, regex=False)

FEATURE_COLS = [c for c in df_train.columns if c not in META_COLS]
print(f"  Train: {len(df_train)}  Test: {len(df_test)}  Tabular features: {len(FEATURE_COLS)}")

# Encode primers
le_primer = LabelEncoder()
all_primers = pd.concat([df_train['primer'], df_test['primer']]).fillna('UNK')
le_primer.fit(all_primers)
N_PRIMERS = len(le_primer.classes_)

# Impute tabular NaNs
imp = SimpleImputer(strategy='median')
X_train_tab = imp.fit_transform(df_train[FEATURE_COLS].values.astype(float))
X_test_tab  = imp.transform(df_test[FEATURE_COLS].values.astype(float))

train_primers = le_primer.transform(df_train['primer'].fillna('UNK'))
test_primers  = le_primer.transform(df_test['primer'].fillna('UNK'))
y_train = df_train['label'].values.astype(int)
y_test  = df_test['label'].values.astype(int)
train_groups = df_train['sample_id'].values

pos_weight = torch.tensor(
    [(y_train == 0).sum() / max((y_train == 1).sum(), 1)],
    dtype=torch.float32
).to(DEVICE)
print(f"  pos_weight: {pos_weight.item():.2f}")

# ─────────────────────────────────────────────
# STEP 2: LOAD & CACHE ALL TRACES
# ─────────────────────────────────────────────
def load_trace(path: str) -> np.ndarray:
    """Load 4-channel trace from Tracy JSON → float32 (4, MAX_LEN)."""
    with open(path) as f:
        d = json.load(f)
    channels = np.array(
        [d['peakA'], d['peakC'], d['peakG'], d['peakT']], dtype=np.float32
    )
    # per-file z-score normalisation
    mu  = channels.mean(axis=1, keepdims=True)
    std = channels.std(axis=1, keepdims=True) + 1e-8
    channels = (channels - mu) / std

    L = channels.shape[1]
    if L >= MAX_LEN:
        return channels[:, :MAX_LEN]
    pad = np.zeros((4, MAX_LEN - L), dtype=np.float32)
    return np.concatenate([channels, pad], axis=1)

def cache_traces(df: pd.DataFrame, desc: str) -> np.ndarray:
    traces = []
    failed = 0
    for path in df['json_path']:
        try:
            traces.append(load_trace(path))
        except Exception:
            traces.append(np.zeros((4, MAX_LEN), dtype=np.float32))
            failed += 1
    if failed:
        print(f"  WARNING: {failed} files could not be loaded in {desc}")
    return np.stack(traces)

print("Caching train traces...")
traces_train = cache_traces(df_train, "train")
print("Caching test traces...")
traces_test  = cache_traces(df_test,  "test")
print(f"  Trace shape: {traces_train.shape}")

# ─────────────────────────────────────────────
# STEP 3: DATASET & DATALOADER
# ─────────────────────────────────────────────
class MtDNADataset(Dataset):
    def __init__(self, traces, tab_features, primers, labels):
        self.traces   = torch.from_numpy(traces)
        self.tab      = torch.from_numpy(tab_features.astype(np.float32))
        self.primers  = torch.from_numpy(primers.astype(np.int64))
        self.labels   = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.traces[i], self.tab[i], self.primers[i], self.labels[i]

# ─────────────────────────────────────────────
# STEP 4: MODEL ARCHITECTURE
# ─────────────────────────────────────────────
class DWSBlock(nn.Module):
    """Depthwise-separable conv block with residual."""
    def __init__(self, channels, kernel_size, n_sub=2):
        super().__init__()
        pad = kernel_size // 2
        layers = []
        for _ in range(n_sub):
            layers += [
                nn.Conv1d(channels, channels, kernel_size,
                          padding=pad, groups=channels, bias=False),
                nn.Conv1d(channels, channels, 1, bias=False),
                nn.BatchNorm1d(channels),
                nn.ReLU(inplace=True),
            ]
        self.block    = nn.Sequential(*layers)
        self.residual = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.residual(x))


class HybridClassifier(nn.Module):
    def __init__(self, n_tab_features, n_primers, trace_embed_dim=64,
                 tab_embed_dim=32, primer_embed_dim=8):
        super().__init__()

        # ── MatchboxNet branch ────────────────────────────────────────────
        self.trace_prologue = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.trace_blocks = nn.Sequential(
            DWSBlock(64,  kernel_size=13, n_sub=2),
            DWSBlock(64,  kernel_size=17, n_sub=2),
            DWSBlock(64,  kernel_size=21, n_sub=2),
        )
        self.trace_epilogue = nn.Sequential(
            nn.Conv1d(64, trace_embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(trace_embed_dim),
            nn.ReLU(inplace=True),
        )

        # ── Tabular branch ────────────────────────────────────────────────
        self.tab_mlp = nn.Sequential(
            nn.BatchNorm1d(n_tab_features),
            nn.Linear(n_tab_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, tab_embed_dim),
            nn.ReLU(inplace=True),
        )

        # ── Primer embedding ──────────────────────────────────────────────
        self.primer_embed = nn.Embedding(n_primers, primer_embed_dim)

        # ── Fusion head ───────────────────────────────────────────────────
        fused_dim = trace_embed_dim + tab_embed_dim + primer_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, trace, tab, primer):
        t = self.trace_prologue(trace)
        t = self.trace_blocks(t)
        t = self.trace_epilogue(t)
        t = t.mean(dim=2)                     # global average pool

        f = self.tab_mlp(tab)
        p = self.primer_embed(primer)

        x = torch.cat([t, f, p], dim=1)
        return self.head(x).squeeze(1)        # logits


# ─────────────────────────────────────────────
# STEP 5: TRAIN / EVAL HELPERS
# ─────────────────────────────────────────────
def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train(train)
    losses, all_logits, all_labels = [], [], []
    with torch.set_grad_enabled(train):
        for trace, tab, primer, label in loader:
            trace  = trace.to(DEVICE)
            tab    = tab.to(DEVICE)
            primer = primer.to(DEVICE)
            label  = label.to(DEVICE)

            logits = model(trace, tab, primer)
            loss   = criterion(logits, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            all_logits.append(logits.detach().cpu())
            all_labels.append(label.cpu())

    logits_cat = torch.cat(all_logits).numpy()
    labels_cat = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(torch.from_numpy(logits_cat)).numpy()
    auc   = roc_auc_score(labels_cat, probs) if len(np.unique(labels_cat)) > 1 else 0.0
    return float(np.mean(losses)), auc, probs


def train_one_model(X_trace, X_tab, X_primer, y,
                    val_idx=None, seed=42):
    torch.manual_seed(seed)
    if val_idx is not None:
        tr_mask = np.ones(len(y), bool); tr_mask[val_idx] = False
        tr_idx  = np.where(tr_mask)[0]
        tr_ds = MtDNADataset(X_trace[tr_idx], X_tab[tr_idx], X_primer[tr_idx], y[tr_idx])
        va_ds = MtDNADataset(X_trace[val_idx], X_tab[val_idx], X_primer[val_idx], y[val_idx])
    else:
        tr_ds = MtDNADataset(X_trace, X_tab, X_primer, y)
        va_ds = None

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=64,         shuffle=False, num_workers=0) if va_ds else None

    model     = HybridClassifier(
            X_tab.shape[1], N_PRIMERS,
            trace_embed_dim=TRACE_EMBED_DIM,
            tab_embed_dim=TAB_EMBED_DIM,
            primer_embed_dim=PRIMER_EMBED_DIM,
            dropout=DROPOUT,
        ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    best_auc, best_state, no_imp = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_auc, _ = run_epoch(model, tr_loader, optimizer, criterion, train=True)
        if va_loader:
            _, va_auc, _ = run_epoch(model, va_loader, optimizer, criterion, train=False)
            scheduler.step(va_auc)
            if va_auc > best_auc:
                best_auc  = va_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    early stop at epoch {epoch}  best_val_auc={best_auc:.4f}")
                break
        if epoch % 5 == 0:
            msg = f"    epoch {epoch:3d}  tr_loss={tr_loss:.4f}  tr_auc={tr_auc:.4f}"
            if va_loader:
                msg += f"  va_auc={va_auc:.4f}"
            print(msg)

    if best_state:
        model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────
# STEP 6: CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\nRunning 5-fold CV (Stratified by sample_id)...")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.full(len(y_train), -1.0)
fold_aucs = []

for fold, (_, val_idx) in enumerate(
        cv.split(traces_train, y_train, train_groups)):
    print(f"\n  Fold {fold+1}/5  (val={len(val_idx)} samples)")
    model = train_one_model(
        traces_train, X_train_tab, train_primers, y_train,
        val_idx=val_idx, seed=42 + fold
    )
    va_ds     = MtDNADataset(
        traces_train[val_idx], X_train_tab[val_idx],
        train_primers[val_idx], y_train[val_idx]
    )
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)
    _, auc, probs = run_epoch(model, va_loader, None, nn.BCEWithLogitsLoss(), train=False)
    oof_probs[val_idx] = probs
    fold_aucs.append(auc)
    print(f"  Fold {fold+1} AUC = {auc:.4f}")

print(f"\n  Mean CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# ─────────────────────────────────────────────
# STEP 7: FINAL TRAINING ON FULL TRAIN SET
# ─────────────────────────────────────────────
print("\nTraining final model on full train set...")
final_model = train_one_model(
    traces_train, X_train_tab, train_primers, y_train,
    val_idx=None, seed=42
)

# ─────────────────────────────────────────────
# STEP 8: EVALUATION ON TEST SET
# ─────────────────────────────────────────────
print("\n" + "="*52)
print("INDEPENDENT TEST SET EVALUATION")
print("="*52)

te_ds     = MtDNADataset(traces_test, X_test_tab, test_primers, y_test)
te_loader = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=0)
_, test_auc, test_probs = run_epoch(
    final_model, te_loader, None, nn.BCEWithLogitsLoss(), train=False
)

# Threshold from OOF (same logic as train_final_v2.py)
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
# STEP 9: SAVE
# ─────────────────────────────────────────────
test_pred_df = pd.DataFrame({
    'json_path':  df_test['json_path'],
    'sample_id':  df_test['sample_id'],
    'primer':     df_test['primer'],
    'label':      y_test,
    'prob_error': test_probs,
    'pred':       test_preds,
})
test_pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions_hybrid.csv"), index=False)

torch.save(final_model.state_dict(),
           os.path.join(OUTPUT_DIR, "classifier_hybrid.pt"))

with open(os.path.join(OUTPUT_DIR, "classifier_hybrid_meta.pkl"), 'wb') as f:
    pickle.dump({
        'imputer':       imp,
        'label_encoder': le_primer,
        'feature_cols':  FEATURE_COLS,
        'threshold':     best_thr,
        'test_auc':      test_auc,
        'cv_auc_mean':   float(np.mean(fold_aucs)),
        'cv_auc_std':    float(np.std(fold_aucs)),
        'n_primers':     N_PRIMERS,
        'max_len':       MAX_LEN,
    }, f)

print(f"\n✓ Model weights → {OUTPUT_DIR}/classifier_hybrid.pt")
print(f"✓ Metadata      → {OUTPUT_DIR}/classifier_hybrid_meta.pkl")
print(f"✓ Predictions   → {OUTPUT_DIR}/test_predictions_hybrid.csv")
print("Done!")
