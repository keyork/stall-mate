# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

import numpy as np

from stall_mate.cshda.layer2_quantification.embedder import Embedder

_log = logging.getLogger(__name__)


class PolarityScorer:
    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def score_attribute(
        self,
        value_description: str,
        positive_anchor: str,
        negative_anchor: str,
    ) -> float:
        if not positive_anchor or not negative_anchor:
            return 0.5

        desc_vec = self._embedder.embed(value_description)
        pos_vec = self._embedder.embed(positive_anchor)
        neg_vec = self._embedder.embed(negative_anchor)

        axis = pos_vec - neg_vec
        centered = desc_vec - neg_vec
        axis_norm_sq = float(np.dot(axis, axis))

        if axis_norm_sq < 1e-10:
            return 0.5

        raw = float(np.dot(centered, axis)) / axis_norm_sq
        return float(np.clip(raw, 0.0, 1.0))

    def score_numeric(self, value: float, all_values: list[float]) -> float:
        min_val = min(all_values)
        max_val = max(all_values)
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
