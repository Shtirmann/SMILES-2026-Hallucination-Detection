"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary classifier that detects
hallucinations from hidden-state feature vectors.  Called from
``solution.py`` via ``evaluate.run_evaluation``.  All four public methods
(``fit``, ``fit_hyperparameters``, ``predict``, ``predict_proba``) must be
implemented and their signatures must not change.

Strategy (V5 — Synergistic Structural-Semantic Hybrid)
------------------------------------------------------
Input: 2738-dim vector from ``aggregation.py``:
  - Block 1 (0:50):     geometric features → RobustScaler
  - Block 2 (50:946):   semantic mean-pool  → (skipped in V5)
  - Block 3 (946:2738):  lexical max-pool   → StandardScaler → PCA(32)

Combined 82-dim vector → BaggingClassifier(LogisticRegression)
→ threshold tuned on internal calibration split for accuracy.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# ── Feature layout constants ─────────────────────────────────────────
N_GEO = 50          # geometric features (block 1)
N_SEM = 896         # semantic mean-pool (block 2, skipped in V5)
# Block 3 (lexical max-pool) starts at offset N_GEO + N_SEM = 946
# ──────────────────────────────────────────────────────────────────────


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; the architecture uses a Bagging ensemble of
    L2-regularised Logistic Regression classifiers with separate scaling for
    geometric and lexical feature blocks.

    The probe applies PCA(32) to the high-dimensional lexical block internally,
    producing 50 + 32 = 82 input features for the classifier.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._geo_scaler: RobustScaler | None = None
        self._adv_scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._clf: BaggingClassifier | None = None
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_features(self, X: np.ndarray):
        """Split raw feature matrix into Geo (50) and Adv (1792) blocks.

        Block 2 (896-dim semantic mean-pool) is skipped — superseded by
        Block 3 (max-pool) which captures lexical anomalies better.

        Returns:
            Tuple of ``(X_geo, X_adv)`` arrays.
        """
        X_geo = X[:, :N_GEO]
        X_adv = X[:, N_GEO + N_SEM:]  # skip block 2
        return X_geo, X_adv

    def _preprocess(self, X: np.ndarray, *, fit: bool = False) -> np.ndarray:
        """Scale geometric features and PCA-compress lexical features.

        Args:
            X:   Full feature matrix ``(n_samples, 2738)``.
            fit: If True, fit scalers and PCA; otherwise, transform only.

        Returns:
            Processed array of shape ``(n_samples, 82)``
            = 50 (geo, RobustScaled) + 32 (adv, PCA-compressed).
        """
        X_geo, X_adv = self._split_features(X)

        if fit:
            self._geo_scaler = RobustScaler()
            X_geo_s = self._geo_scaler.fit_transform(X_geo)

            self._adv_scaler = StandardScaler()
            X_adv_s = self._adv_scaler.fit_transform(X_adv)

            self._pca = PCA(n_components=32, random_state=self.random_state)
            X_adv_pca = self._pca.fit_transform(X_adv_s)
        else:
            X_geo_s = self._geo_scaler.transform(X_geo)
            X_adv_s = self._adv_scaler.transform(X_adv)
            X_adv_pca = self._pca.transform(X_adv_s)

        return np.hstack([X_geo_s, X_adv_pca])

    # ------------------------------------------------------------------
    # Public API (signatures must not change)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Steps:
          1. Reserve 10 % of data for internal threshold calibration.
          2. Preprocess: RobustScaler(geo) + StandardScaler→PCA(32)(adv).
          3. Train BaggingClassifier(LogisticRegression).
          4. Tune threshold on calibration split to maximise accuracy.
          5. Refit on 100 % of data with the same preprocessing.

        Args:
            X: Feature matrix of shape ``(n_samples, 2738)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        # 1. Internal calibration split
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y,
            test_size=0.10,
            stratify=y,
            random_state=self.random_state,
        )

        # 2. Preprocess
        X_tr_proc = self._preprocess(X_tr, fit=True)

        # 3. Base classifier: L2-regularised Logistic Regression
        base_lr = LogisticRegression(
            C=0.01,
            max_iter=1000,
            random_state=self.random_state,
        )

        # 4. Bagging ensemble (variance reduction)
        self._clf = BaggingClassifier(
            estimator=base_lr,
            n_estimators=self.n_estimators,
            max_samples=0.8,
            max_features=0.8,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._clf.fit(X_tr_proc, y_tr)

        # 5. Threshold tuning on calibration split
        X_cal_proc = self._preprocess(X_cal, fit=False)
        cal_probs = self._clf.predict_proba(X_cal_proc)[:, 1]

        best_threshold, best_acc = 0.5, -1.0
        for t in np.linspace(0.1, 0.9, 100):
            acc = accuracy_score(y_cal, (cal_probs >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_threshold = float(t)
        self._threshold = best_threshold

        # 6. Refit on 100 % of data
        X_all_proc = self._preprocess(X, fit=True)
        self._clf.fit(X_all_proc, y)

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise accuracy.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 201)])
        )

        best_threshold, best_acc = 0.5, -1.0
        for t in candidates:
            acc = accuracy_score(y_val, (probs >= t).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_processed = self._preprocess(X, fit=False)
        return self._clf.predict_proba(X_processed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — not used; probe uses sklearn internally."""
        raise NotImplementedError("Use predict() / predict_proba().")