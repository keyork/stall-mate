# SPDX-License-Identifier: Apache-2.0
"""终端显示模块 | Terminal display — rich-based beautiful output for experiment execution."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.progress import ProgressColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from stall_mate.runner.experiment import RunStats
    from stall_mate.types import ExperimentRecord

_STATUS_COLORS: dict[str, str] = {
    "VALID": "green",
    "REFUSED": "magenta",
    "AMBIGUOUS": "yellow",
    "ERROR": "red",
}

_STATUS_ICONS: dict[str, str] = {
    "VALID": "✓",
    "REFUSED": "⊘",
    "AMBIGUOUS": "?",
    "ERROR": "✗",
}


class LastResultColumn(ProgressColumn):
    """自定义进度列：显示最近一次调用的结果 | Custom progress column showing the latest call result."""

    def __init__(self) -> None:
        super().__init__()
        self._last_text: Text = Text("")

    def update(self, text: Text) -> None:
        self._last_text = text

    def render(self, task: object) -> Text:
        return self._last_text


class ExperimentDisplay:
    """实验终端显示 | Rich-based terminal display for experiment execution.

    Handles all visual output: progress bars, headers, summaries, retry notifications.
    """

    def __init__(self) -> None:
        self._console = Console()
        self._last_result_col = LastResultColumn()

    def print_experiment_header(
        self,
        exp_id: str,
        description: str,
        total_calls: int,
        output_path: Path,
    ) -> None:
        """打印实验头部信息 | Print styled experiment header panel."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        content = Text.from_markup(
            f"[bold cyan]🧪 {exp_id}[/]\n"
            f"[dim]{description}[/]\n\n"
            f"[white]📊 总调用次数 / Total calls:[/] [bold]{total_calls}[/]\n"
            f"[white]📁 输出路径 / Output:[/]       [dim]{output_path}[/]\n"
            f"[white]🕐 开始时间 / Started:[/]      [dim]{now}[/]"
        )
        panel = Panel(
            content,
            title="[bold yellow]🚽 坑位博弈实验 | Stall Decision Experiment[/]",
            subtitle="[dim]让大模型做出人生中最难的抉择[/]",
            border_style="bright_blue",
            padding=(1, 2),
        )
        self._console.print(panel)
        self._console.print()

    def create_progress(self, total: int, label: str) -> Progress:
        """创建配置好的进度条 | Create a configured Rich Progress instance."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", justify="left"),
            BarColumn(bar_width=40, complete_style="bright_cyan", finished_style="green"),
            TextColumn("🚽 [progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),
            TimeRemainingColumn(compact=True, elapsed_when_finished=True),
            TextColumn("  "),
            self._last_result_col,
            console=self._console,
            transient=False,
            auto_refresh=True,
        )

    def format_record_status(self, record: ExperimentRecord) -> Text:
        """格式化单条记录为状态文本 | Format a record into a colored status Text."""
        status: str = record.choice_status.value
        color = _STATUS_COLORS.get(status, "white")
        icon = _STATUS_ICONS.get(status, "?")

        choice_str = f"#{record.extracted_choice}" if record.extracted_choice is not None else "-"
        latency_str = f"{record.latency_ms / 1000:.1f}s"

        text = Text.from_markup(
            f"[{color}]{icon} #{choice_str} {latency_str}[/]"
        )
        self._last_result_col.update(text)
        return text

    def print_retry_round(self, round_num: int, max_retries: int, failed_count: int) -> None:
        """打印重试轮次信息 | Print retry round notification."""
        self._console.print()
        self._console.print(Rule(
            f"🔄 重试第 [bold yellow]{round_num}[/]/[dim]{max_retries}[/] 轮  "
            f"| [red]{failed_count}[/] 次失败调用",
            style="yellow",
        ))

    def print_retry_exhausted(self, failed_count: int, max_retries: int) -> None:
        """打印重试耗尽警告 | Print warning when retries are exhausted."""
        self._console.print()
        self._console.print(
            Panel(
                f"[bold red]⚠️  {failed_count} 次调用在 {max_retries} 轮重试后仍然失败\n"
                f"[dim]Some calls failed after all retry rounds.[/]",
                border_style="red",
                padding=(0, 2),
            )
        )

    def print_experiment_summary(self, stats: RunStats) -> None:
        """打印单次实验总结 | Print experiment summary with statistics table."""
        self._console.print()
        self._console.print(Rule("📊 坑位选择报告 | Stall Choice Report", style="bright_cyan"))

        success_rate = (
            f"{stats.valid / stats.total_calls * 100:.1f}%"
            if stats.total_calls > 0
            else "N/A"
        )
        avg_latency = (
            f"{stats.total_latency_ms / stats.total_calls:.0f}ms"
            if stats.total_calls > 0
            else "N/A"
        )
        elapsed = stats.elapsed_seconds
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

        table = Table(show_header=True, header_style="bold cyan", border_style="bright_blue")
        table.add_column("指标 / Metric", style="white", min_width=24)
        table.add_column("值 / Value", justify="right", style="bold")

        table.add_row("🧪 总调用 / Total Calls", str(stats.total_calls))
        table.add_row("✅ 有效 / Valid", f"[green]{stats.valid}[/]")
        table.add_row("🚫 拒绝 / Refused", f"[magenta]{stats.refused}[/]")
        table.add_row("❓ 歧义 / Ambiguous", f"[yellow]{stats.ambiguous}[/]")
        table.add_row("❌ 错误 / Error", f"[red]{stats.error}[/]")
        table.add_row("🎯 成功率 / Success Rate", f"[bold]{success_rate}[/]")
        table.add_row("⏱️  平均延迟 / Avg Latency", avg_latency)
        table.add_row("🔄 重试次数 / Retries", str(stats.retries_used))
        table.add_row("⏰ 耗时 / Elapsed", elapsed_str)

        self._console.print(table)
        self._console.print()

    def print_global_summary(self, stats: RunStats, data_dir: Path) -> None:
        """打印全局总结 | Print global summary across all experiments."""
        self._console.print()
        self._console.print(
            Rule("🏁 全部实验完成 | All Experiments Complete", style="bright_green")
        )

        self.print_experiment_summary(stats)

        self._console.print(
            Panel(
                f"📁 数据目录 / Data dir: [dim]{data_dir}[/]",
                border_style="green",
                padding=(0, 2),
            )
        )
