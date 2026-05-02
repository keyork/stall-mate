# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

import numpy as np

_log = logging.getLogger(__name__)


class WeightCalculator:
    def variance_weights(self, score_matrix: np.ndarray) -> np.ndarray:
        variances = np.var(score_matrix, axis=0, ddof=0)
        if np.all(variances < 1e-12):
            n = score_matrix.shape[1]
            return np.ones(n, dtype=np.float64) / n
        return variances / variances.sum()

    def entropy_weights(self, score_matrix: np.ndarray) -> np.ndarray:
        n_rows, n_cols = score_matrix.shape
        epsilon = 1e-10
        col_sums = score_matrix.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        prob = score_matrix / col_sums + epsilon
        h = -np.sum(prob * np.log(prob), axis=0)
        h_max = np.log(n_rows) if n_rows > 1 else 1.0
        d = 1.0 - h / h_max
        total = d.sum()
        if total == 0:
            return np.ones(n_cols, dtype=np.float64) / n_cols
        return d / total

    def ensemble_weights(self, score_matrix: np.ndarray) -> np.ndarray:
        vw = self.variance_weights(score_matrix)
        ew = self.entropy_weights(score_matrix)
        combined = (vw + ew) / 2.0
        return combined / combined.sum()
