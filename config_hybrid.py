"""
Config for train_hybrid.py — chỉ cần sửa config.py khi chuyển máy.
"""
from config import BASE_DIR, OLD_BASE  # noqa: F401

# ── Device ────────────────────────────────────────────────────────────────────
# "mps"  → Apple M-series  |  "cuda" → NVIDIA GPU  |  "cpu" → fallback
DEVICE = "mps"

# ── Trace ─────────────────────────────────────────────────────────────────────
MAX_LEN = 5000        # pad / truncate all traces to this length

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 40
LR         = 1e-3
PATIENCE   = 7        # early stopping (epochs without val AUC improvement)

# ── Model capacity ────────────────────────────────────────────────────────────
TRACE_EMBED_DIM  = 64   # MatchboxNet output dim after global avg pool
TAB_EMBED_DIM    = 32   # tabular MLP output dim
PRIMER_EMBED_DIM = 8    # primer embedding dim
DROPOUT          = 0.3
