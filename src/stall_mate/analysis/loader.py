# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from stall_mate.recorder import JSONLRecorder
from stall_mate.types import ExperimentRecord


@dataclass
class ConditionGroup:
    experiment_group: str
    num_stalls: int
    temperature: float
    template: str
    records: list[ExperimentRecord] = field(default_factory=list)

    @property
    def choices(self) -> list[int]:
        """提取所有有效选择 | Extract all non-None extracted choices from records."""
        return [r.extracted_choice for r in self.records if r.extracted_choice is not None]

    @property
    def label(self) -> str:
        """生成分组标签字符串 | Generate a human-readable label for this condition group."""
        return f"G={self.experiment_group} N={self.num_stalls} T={self.temperature} tpl={self.template}"


def load_experiment_data(data_dir: Path) -> dict[str, list[ExperimentRecord]]:
    """从目录加载所有实验 JSONL 文件 | Load all experiment JSONL files from a directory."""
    data_dir = Path(data_dir)
    result: dict[str, list[ExperimentRecord]] = {}
    for path in sorted(data_dir.glob("phase*.jsonl")):
        stem = path.stem
        if stem.startswith("phase1_"):
            experiment_id = stem.replace("phase1_", "")
        elif stem.startswith("phase2"):
            experiment_id = "2"
        else:
            experiment_id = stem
        recorder = JSONLRecorder(path)
        records = recorder.read_all()
        if records:
            result[experiment_id] = records
    return result


def group_by_condition(records: list[ExperimentRecord]) -> list[ConditionGroup]:
    """按实验条件（组别、坑位数、温度、模板）分组 | Group records by condition (group, stalls, temperature, template)."""
    buckets: dict[tuple, list[ExperimentRecord]] = {}
    for r in records:
        key = (r.experiment_group, r.num_stalls, r.temperature, r.prompt_template.value)
        buckets.setdefault(key, []).append(r)

    groups: list[ConditionGroup] = []
    for (eg, ns, temp, tpl), recs in sorted(buckets.items()):
        groups.append(ConditionGroup(
            experiment_group=eg,
            num_stalls=ns,
            temperature=temp,
            template=tpl,
            records=recs,
        ))
    return groups


def choice_distribution(choices: list[int], num_stalls: int) -> np.ndarray:
    """计算每个坑位的选择次数数组 | Compute raw choice count array indexed by stall position."""
    counts = np.zeros(num_stalls, dtype=np.float64)
    for c in choices:
        if 1 <= c <= num_stalls:
            counts[c - 1] += 1
    return counts
