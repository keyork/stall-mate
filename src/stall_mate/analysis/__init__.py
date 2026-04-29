# SPDX-License-Identifier: Apache-2.0
"""数据分析 | Experiment data analysis — metrics, statistics, and visualization."""

from stall_mate.analysis.loader import ConditionGroup, choice_distribution, group_by_condition, load_experiment_data
from stall_mate.analysis.metrics import (
    chi2_independence_test,
    chi2_uniform_test,
    choice_entropy,
    choice_frequencies,
    endpoint_preference,
    jsd_between_distributions,
    mcr,
    middle_preference,
    normalized_entropy,
    relative_position,
)
from stall_mate.analysis.report import generate_phase1_report

__all__ = [
    "ConditionGroup",
    "chi2_independence_test",
    "chi2_uniform_test",
    "choice_distribution",
    "choice_entropy",
    "choice_frequencies",
    "endpoint_preference",
    "generate_phase1_report",
    "group_by_condition",
    "jsd_between_distributions",
    "load_experiment_data",
    "mcr",
    "middle_preference",
    "normalized_entropy",
    "relative_position",
]
