#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CSHDA 决策引擎 Demo | CSHDA Decision Engine Demo.

运行方法 / Usage:
    uv run python scripts/demo_cshda.py

演示厕所选择问题通过 CSHDA 四层流水线的完整决策过程。
"""

from __future__ import annotations

import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from stall_mate.cshda import CSHDAEngine

console = Console()

TOILET_PROMPT = (
    "你走进一间公共厕所，面前有一排5个独立的坑位，编号1到5。"
    "其中第2个和第4个有人正在使用。你会选择哪一个？"
)


def main() -> None:
    console.print(Panel.fit(
        "[bold cyan]CSHDA 通用决策引擎 Demo[/]\n"
        "[dim]Consistent Symbolic-Heuristic Decision Architecture[/]",
        border_style="bright_blue",
    ))

    console.print(f"\n[bold]输入问题:[/]\n{TOILET_PROMPT}\n")

    engine = CSHDAEngine(
        model="glm-5.1",
        base_url="http://localhost:3000/v1",
        extraction_rounds=3,
        device="cpu",
    )

    console.print("[bold yellow]运行中... (Layer 1 需要 LLM API 调用，约 30-60 秒)[/]\n")

    result = engine.decide(TOILET_PROMPT)

    dr = result.decision
    cr = result.consistency_report
    cb = result.confidence_breakdown

    console.print(Panel.fit(
        f"[bold green]决策结果[/]\n\n"
        f"选择: [bold]{dr.chosen}[/]\n"
        f"排序: {dr.ranking}\n"
        f"效用值: {dr.objective_value:.4f}\n"
        f"余量 (margin): {dr.margin:.4f}\n"
        f"求解器: {dr.solver_name}",
        border_style="green",
    ))

    if dr.intermediate_values:
        table = Table(title="中间计算结果")
        table.add_column("指标")
        table.add_column("值")
        for k, v in dr.intermediate_values.items():
            if isinstance(v, dict):
                table.add_row(k, json.dumps(v, ensure_ascii=False, indent=2))
            else:
                table.add_row(k, str(v))
        console.print(table)

    console.print(Panel.fit(
        f"[bold]一致性报告[/]\n"
        f"确定性: {cr.determinism} | 约束满足: {cr.constraint_satisfaction}\n"
        f"传递性: {cr.transitivity} | IIA: {cr.iia} | 框架不变性: {cr.frame_invariance}\n\n"
        f"[bold]置信度[/]: {result.confidence_score:.2f}\n"
        f"  提取稳定性: {cb.extraction_stability:.2f}\n"
        f"  量化鲁棒性: {cb.quantification_robustness:.2f}\n"
        f"  解余量: {cb.solution_margin:.4f}",
        border_style="cyan",
    ))

    console.print("\n[bold green]Demo 完成! | Demo complete![/]")


if __name__ == "__main__":
    main()
