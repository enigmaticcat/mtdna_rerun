"""
Config for train_hybrid.py — edit this file before running on a new machine.
"""

# ── Paths ─────────────────────────────────────────────────────────────────────
OLD_BASE  = "/home/minhtq/mtDNA_proj/mtdna_rerun"   # original path in manifests
BASE_DIR  = "/Users/nguyenthithutam/Desktop/mtdna_rerun"   # current machine

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
