# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon


def mcr(choices: list[int]) -> float:
    if not choices:
        return 0.0
    freq = Counter(choices)
    max_count = max(freq.values())
    return max_count / len(choices)


def choice_entropy(choices: list[int], num_stalls: int) -> float:
    if not choices or num_stalls <= 0:
        return 0.0
    counts = np.zeros(num_stalls, dtype=np.float64)
    for c in choices:
        if 1 <= c <= num_stalls:
            counts[c - 1] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    pk = counts / total
    nonzero = pk[pk > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def normalized_entropy(choices: list[int], num_stalls: int) -> float:
    if num_stalls <= 1:
        return 0.0
    h = choice_entropy(choices, num_stalls)
    max_h = np.log2(num_stalls)
    return h / max_h


def jsd_between_distributions(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    prob_a = dist_a / dist_a.sum() if dist_a.sum() > 0 else dist_a
    prob_b = dist_b / dist_b.sum() if dist_b.sum() > 0 else dist_b
    js_dist = jensenshannon(prob_a, prob_b, base=2)
    return float(js_dist ** 2)


def chi2_uniform_test(choices: list[int], num_stalls: int) -> tuple[float, float]:
    if not choices or num_stalls <= 1:
        return (0.0, 1.0)
    counts = np.zeros(num_stalls, dtype=np.float64)
    for c in choices:
        if 1 <= c <= num_stalls:
            counts[c - 1] += 1
    nonzero_bins = counts[counts > 0]
    if len(nonzero_bins) <= 1:
        return (0.0, 1.0)
    result = stats.chisquare(counts)
    return (float(result.statistic), float(result.pvalue))


def chi2_independence_test(
    choices_a: list[int], choices_b: list[int], num_stalls: int
) -> tuple[float, float]:
    if not choices_a or not choices_b or num_stalls <= 1:
        return (0.0, 1.0)
    counts_a = np.zeros(num_stalls, dtype=np.float64)
    counts_b = np.zeros(num_stalls, dtype=np.float64)
    for c in choices_a:
        if 1 <= c <= num_stalls:
            counts_a[c - 1] += 1
    for c in choices_b:
        if 1 <= c <= num_stalls:
            counts_b[c - 1] += 1
    table = np.array([counts_a, counts_b])
    nonzero_cols = table.sum(axis=0) > 0
    if nonzero_cols.sum() < 2:
        return (0.0, 1.0)
    table = table[:, nonzero_cols]
    result = stats.chi2_contingency(table, correction=False)
    return (float(result.statistic), float(result.pvalue))


def choice_frequencies(choices: list[int], num_stalls: int) -> dict[int, float]:
    if not choices:
        return {i: 0.0 for i in range(1, num_stalls + 1)}
    freq = Counter(choices)
    total = len(choices)
    return {i: freq.get(i, 0) / total for i in range(1, num_stalls + 1)}


def endpoint_preference(choices: list[int], num_stalls: int) -> float:
    if not choices:
        return 0.0
    endpoints = {1, num_stalls}
    count = sum(1 for c in choices if c in endpoints)
    return count / len(choices)


def middle_preference(choices: list[int], num_stalls: int) -> float:
    if not choices or num_stalls < 3:
        return 0.0
    mid = num_stalls // 2
    middles = {mid, mid + 1} if num_stalls % 2 == 0 else {mid + 1}
    count = sum(1 for c in choices if c in middles)
    return count / len(choices)


def relative_position(choices: list[int], num_stalls: int) -> list[float]:
    if num_stalls <= 1:
        return [0.0] * len(choices)
    return [(c - 1) / (num_stalls - 1) for c in choices]
