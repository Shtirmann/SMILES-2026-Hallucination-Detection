"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.

Strategy — Nested Stratified 5-Fold Cross-Validation
-----------------------------------------------------
Each fold uses a **different test set** (outer fold), providing honest
generalisation estimates.  The remaining data is split 85/15 into
train / val for threshold tuning.

This produces 5 evaluation rounds where every sample appears in the test
set exactly once, giving an unbiased estimate of generalisation error on
only 689 samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# ── Configurable constants ────────────────────────────────────────────
N_FOLDS = 5
# ──────────────────────────────────────────────────────────────────────


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    Uses Stratified K-Fold so the test set rotates across folds.
    For each fold: the outer fold = test, the remaining data is split
    85/15 into train/val for threshold tuning.

    Every sample appears in the test set exactly once across all folds.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
                      (Unused — test size is determined by ``N_FOLDS``.)
        val_size:     Fraction of non-test data used for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.
    """
    idx = np.arange(len(y))

    # Outer K-Fold: each fold produces a different test set
    outer_kf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for dev_rel, test_rel in outer_kf.split(idx, y):
        idx_dev = idx[dev_rel]
        idx_test = idx[test_rel]

        # Inner split: dev → train + val
        idx_train, idx_val = train_test_split(
            idx_dev,
            test_size=val_size,
            random_state=random_state,
            stratify=y[idx_dev],
        )

        splits.append((idx_train, idx_val, idx_test))

    return splits