#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""10 次调用功能验证 | 10-call functional verification.

验证完整实验管线：配置加载 → 提示词构建 → API 调用 → 响应解析 → JSONL 记录。
Verifies the full pipeline: config → prompt → API call → parse → JSONL record.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table

from stall_mate.client import LLMClient
from stall_mate.config import (
    load_classification_config,
    load_model_config,
    load_prompt_templates,
)
from stall_mate.prompt import build_prompt, build_system_message
from stall_mate.recorder import JSONLRecorder
from stall_mate.runner import ExperimentRunner


def main():
    console = Console()

    model_cfg = load_model_config(ROOT / "configs" / "models.yaml")
    templates = load_prompt_templates(ROOT / "configs" / "prompt_templates" / "phase1.yaml")
    classification_cfg = load_classification_config(ROOT / "configs" / "classification.yaml")

    client = LLMClient(
        endpoint=model_cfg.endpoint,
        model=model_cfg.name,
        api_key=model_cfg.api_key,
        timeout=model_cfg.timeout,
        max_retries=model_cfg.max_retries,
        probe_message=model_cfg.probe_message,
    )
    output_path = ROOT / "data" / "verify_10calls.jsonl"
    recorder = JSONLRecorder(output_path)
    recorder.clear()
    runner = ExperimentRunner(
        client=client,
        recorder=recorder,
        model_config=model_cfg,
        refusal_keywords=classification_cfg.refusal_keywords,
        extraction_patterns=classification_cfg.to_extraction_patterns(),
    )

    calls = [
        {"num_stalls": 5, "temp": 0.0, "template": "A", "label": "T0-tmplA-N5"},
        {"num_stalls": 5, "temp": 0.0, "template": "B", "label": "T0-tmplB-N5"},
        {"num_stalls": 5, "temp": 0.0, "template": "C", "label": "T0-tmplC-N5"},
        {"num_stalls": 3, "temp": 0.7, "template": "A", "label": "T0.7-tmplA-N3"},
        {"num_stalls": 7, "temp": 0.7, "template": "A", "label": "T0.7-tmplA-N7"},
        {"num_stalls": 10, "temp": 0.7, "template": "A", "label": "T0.7-tmplA-N10"},
        {"num_stalls": 5, "temp": 0.0, "template": "D", "label": "consistency-1"},
        {"num_stalls": 5, "temp": 0.0, "template": "D", "label": "consistency-2"},
        {"num_stalls": 15, "temp": 0.7, "template": "B", "label": "T0.7-tmplB-N15"},
        {"num_stalls": 20, "temp": 0.7, "template": "C", "label": "T0.7-tmplC-N20"},
    ]

    console.print(Panel(
        "[bold cyan]🚽 Stall Mate — 10-call Functional Verification[/]\n"
        "[dim]10 次调用功能验证[/]",
        border_style="bright_blue",
        padding=(1, 2),
    ))

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, complete_style="bright_cyan", finished_style="green"),
        TextColumn("🚽 [progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        console=console,
    ) as progress:
        task_id = progress.add_task("🧠 验证调用中...", total=10)
        for i, call in enumerate(calls):
            prompt = build_prompt(templates.templates[call["template"]], call["num_stalls"])
            system_msg = build_system_message(
                call["num_stalls"], template=templates.system_message_template,
            )
            metadata = {
                "experiment_phase": "Phase1",
                "experiment_group": "verify",
                "prompt_template": call["template"],
                "prompt_text": prompt,
            }

            record = runner.run_single(
                prompt=prompt,
                system_message=system_msg,
                temperature=call["temp"],
                num_stalls=call["num_stalls"],
                metadata=metadata,
            )
            results.append(record)

            status = record.choice_status.value
            choice = f"#{record.extracted_choice}" if record.extracted_choice is not None else "-"
            latency = f"{record.latency_ms / 1000:.1f}s"
            progress.update(
                task_id,
                advance=1,
                description=f"[{status}] {call['label']} → {choice} {latency}",
            )

    print_summary(console, results, calls)

    passed = verify_jsonl(output_path, expected_count=10)
    if passed:
        console.print(Panel(
            "[bold green]✅ PASS — All 10 calls verified successfully[/]\n"
            "[dim]全部 10 次调用验证成功[/]",
            border_style="green",
            padding=(1, 2),
        ))
        sys.exit(0)
    else:
        console.print(Panel(
            "[bold red]❌ FAIL — Verification failed[/]\n"
            "[dim]验证失败[/]",
            border_style="red",
            padding=(1, 2),
        ))
        sys.exit(1)


def print_summary(console: Console, results, calls):
    console.print(Rule("📊 Verification Summary", style="bright_cyan"))

    table = Table(show_header=True, header_style="bold cyan", border_style="bright_blue")
    table.add_column("#", justify="right", width=3)
    table.add_column("Label", style="white", min_width=18)
    table.add_column("Choice", justify="center", width=7)
    table.add_column("Status", width=10)
    table.add_column("Tokens", justify="right", width=7)
    table.add_column("Latency", justify="right", width=9)

    status_colors = {"VALID": "green", "REFUSED": "magenta", "AMBIGUOUS": "yellow", "ERROR": "red"}

    for i, (r, c) in enumerate(zip(results, calls)):
        choice = str(r.extracted_choice) if r.extracted_choice is not None else "N/A"
        color = status_colors.get(r.choice_status.value, "white")
        table.add_row(
            str(i + 1),
            c["label"],
            f"[bold]{choice}[/]",
            f"[{color}]{r.choice_status.value}[/]",
            str(r.response_tokens),
            f"{r.latency_ms}ms",
        )

    console.print(table)

    valid_count = sum(1 for r in results if r.choice_status.value == "VALID")
    avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
    console.print(f"  Valid: [bold green]{valid_count}[/]/10 | Avg latency: [bold]{avg_latency:.0f}ms[/]")


def verify_jsonl(path: Path, expected_count: int) -> bool:
    if not path.exists():
        print(f"\n❌ JSONL file not found: {path}")
        return False

    recorder = JSONLRecorder(path)
    records = recorder.read_all()

    if len(records) != expected_count:
        print(f"\n❌ Expected {expected_count} records, found {len(records)}")
        return False

    required_fields = [
        "record_id", "experiment_phase", "experiment_group", "model_name",
        "temperature", "prompt_template", "prompt_text", "num_stalls",
        "raw_response", "choice_status", "timestamp",
    ]

    for i, r in enumerate(records):
        for field in required_fields:
            val = getattr(r, field, None)
            if val is None or val == "":
                print(f"\n❌ Record {i} missing or empty field: {field}")
                return False

    print(f"\n✅ JSONL verification: {len(records)} records, all fields present")
    return True


if __name__ == "__main__":
    main()
