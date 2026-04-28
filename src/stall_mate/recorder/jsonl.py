# SPDX-License-Identifier: Apache-2.0
"""
JSONL 数据记录器 | JSONL data recorder.

将实验记录以 JSONL 格式写入文件，每行一条 JSON 记录。
Writes experiment records as JSONL — one JSON object per line.
"""

from __future__ import annotations

import json
from pathlib import Path

from stall_mate.types import ExperimentRecord


class JSONLRecorder:
    """JSONL 格式数据记录器 / JSONL format data recorder."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: ExperimentRecord) -> None:
        """追加一条记录 / Append a single record to the JSONL file."""
        line = json.dumps(record.model_dump(mode="json"), default=str, ensure_ascii=False)
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def record_batch(self, records: list[ExperimentRecord]) -> None:
        """追加多条记录 / Append multiple records."""
        for r in records:
            self.record(r)

    def read_all(self) -> list[ExperimentRecord]:
        """读取所有记录 / Read all records. Returns [] if file doesn't exist."""
        if not self.output_path.exists():
            return []
        records = []
        with self.output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(ExperimentRecord.model_validate_json(line))
        return records

    def count(self) -> int:
        """统计记录数 / Count records without loading all."""
        if not self.output_path.exists():
            return 0
        with self.output_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def clear(self) -> None:
        """删除记录文件 / Delete the JSONL file."""
        if self.output_path.exists():
            self.output_path.unlink()
