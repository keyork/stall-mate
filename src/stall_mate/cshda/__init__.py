# SPDX-License-Identifier: Apache-2.0
"""
CSHDA 通用决策引擎 | CSHDA Universal Decision Engine.

Consistent Symbolic-Heuristic Decision Architecture —
将自然语言决策问题转化为确定性符号求解的四层流水线。

Four-layer pipeline: 语义提取 → 量化建模 → 符号求解 → 一致性保障。
"""

from stall_mate.cshda.engine import CSHDAEngine
from stall_mate.cshda.schema.output import FinalOutput

__all__ = ["CSHDAEngine", "FinalOutput"]
