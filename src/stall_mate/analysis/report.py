# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
from stall_mate.analysis.visualize import (
    plot_choice_distribution,
    plot_choice_heatmap,
    plot_entropy_comparison,
    plot_jsd_matrix,
    plot_mcr_comparison,
    plot_temperature_comparison,
)

EXPERIMENT_INFO: dict[str, str] = {
    "1.1": "基准选择 | Baseline Choice — N=3,5,7,10 × T=0.0,0.7 × templates=A,B,C × 30 reps",
    "1.2": "对称性检验 | Symmetry Check — N=5 × T=0.0,0.7 × templates=A,B,C × 30 reps",
    "1.3": "选项数量效应 | Option Count Effect — N=3,5,7,10,15,20 × T=0.0,0.7 × templates=A,B,C × 30 reps",
}


def _compute_group_metrics(group: ConditionGroup) -> dict:
    ch = group.choices
    ns = group.num_stalls
    freqs = choice_frequencies(ch, ns)
    mode_choice = max(freqs, key=freqs.get) if freqs else None
    mean_pos = np.mean(relative_position(ch, ns)) if ch else 0.0
    return {
        "label": group.label,
        "n_recs": len(group.records),
        "mcr": mcr(ch),
        "entropy": choice_entropy(ch, ns),
        "norm_entropy": normalized_entropy(ch, ns),
        "endpoint_pref": endpoint_preference(ch, ns),
        "middle_pref": middle_preference(ch, ns),
        "mean_rel_pos": float(mean_pos),
        "mode_choice": mode_choice,
        "chi2_stat": chi2_uniform_test(ch, ns)[0],
        "chi2_p": chi2_uniform_test(ch, ns)[1],
    }


def _print_metrics_table(console: Console, metrics_list: list[dict]) -> None:
    table = Table(title="核心指标 | Core Metrics", show_lines=True)
    table.add_column("条件 | Condition", style="cyan", max_width=40)
    table.add_column("n", justify="right")
    table.add_column("MCR", justify="right", style="bold")
    table.add_column("H(bits)", justify="right")
    table.add_column("H_norm", justify="right")
    table.add_column("端点偏好", justify="right")
    table.add_column("中间偏好", justify="right")
    table.add_column("均值位置", justify="right")
    table.add_column("众数", justify="right")
    table.add_column("χ²", justify="right")
    table.add_column("p值", justify="right")

    for m in metrics_list:
        sig = "*" if m["chi2_p"] < 0.05 else ""
        table.add_row(
            m["label"],
            str(m["n_recs"]),
            f"{m['mcr']:.3f}",
            f"{m['entropy']:.3f}",
            f"{m['norm_entropy']:.3f}",
            f"{m['endpoint_pref']:.3f}",
            f"{m['middle_pref']:.3f}",
            f"{m['mean_rel_pos']:.3f}",
            str(m["mode_choice"]),
            f"{m['chi2_stat']:.2f}{sig}",
            f"{m['chi2_p']:.4f}",
        )
    console.print(table)


def _compute_cross_template_jsd(groups: list[ConditionGroup]) -> list[dict]:
    nt_map: dict[tuple, dict[str, ConditionGroup]] = defaultdict(dict)
    for g in groups:
        nt_map[(g.experiment_group, g.num_stalls, g.temperature)][g.template] = g

    results: list[dict] = []
    templates = sorted({t for v in nt_map.values() for t in v})
    for key in sorted(nt_map):
        tmap = nt_map[key]
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                ta, tb = templates[i], templates[j]
                if ta in tmap and tb in tmap:
                    da = choice_distribution(tmap[ta].choices, tmap[ta].num_stalls)
                    db = choice_distribution(tmap[tb].choices, tmap[tb].num_stalls)
                    val = jsd_between_distributions(da, db)
                    results.append({
                        "condition": f"G={key[0]} N={key[1]} T={key[2]}",
                        "pair": f"{ta} vs {tb}",
                        "jsd": val,
                    })
    return results


def _print_jsd_table(console: Console, jsd_results: list[dict]) -> None:
    if not jsd_results:
        console.print("[dim]No cross-template pairs to compare.[/dim]")
        return
    table = Table(title="跨模板 JSD | Cross-Template JSD", show_lines=True)
    table.add_column("条件 | Condition", style="cyan")
    table.add_column("模板对 | Template Pair")
    table.add_column("JSD", justify="right", style="bold")
    for r in jsd_results:
        table.add_row(r["condition"], r["pair"], f"{r['jsd']:.6f}")
    console.print(table)


def _compute_chi2_independence(groups: list[ConditionGroup]) -> list[dict]:
    nt_map: dict[tuple, dict[str, ConditionGroup]] = defaultdict(dict)
    for g in groups:
        nt_map[(g.experiment_group, g.num_stalls, g.temperature)][g.template] = g

    results: list[dict] = []
    templates = sorted({t for v in nt_map.values() for t in v})
    for key in sorted(nt_map):
        tmap = nt_map[key]
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                ta, tb = templates[i], templates[j]
                if ta in tmap and tb in tmap:
                    chi2_val, p_val = chi2_independence_test(
                        tmap[ta].choices, tmap[tb].choices, tmap[ta].num_stalls
                    )
                    results.append({
                        "condition": f"G={key[0]} N={key[1]} T={key[2]}",
                        "pair": f"{ta} vs {tb}",
                        "chi2": chi2_val,
                        "p": p_val,
                        "sig": p_val < 0.05,
                    })
    return results


def _print_chi2_independence(console: Console, results: list[dict]) -> None:
    if not results:
        console.print("[dim]No pairs to test.[/dim]")
        return
    table = Table(title="卡方独立性检验 | χ² Independence Test", show_lines=True)
    table.add_column("条件 | Condition", style="cyan")
    table.add_column("模板对 | Template Pair")
    table.add_column("χ²", justify="right")
    table.add_column("p值", justify="right")
    table.add_column("显著", justify="center")
    for r in results:
        sig_mark = "[bold green]YES[/]" if r["sig"] else "[dim]no[/]"
        table.add_row(
            r["condition"],
            r["pair"],
            f"{r['chi2']:.2f}",
            f"{r['p']:.4f}",
            sig_mark,
        )
    console.print(table)


def generate_phase1_report(
    experiment_data_dir: Path,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    console = Console(record=True)
    buf = io.StringIO()
    text_console = Console(file=buf, width=120)

    def print_both(*args, **kwargs):
        console.print(*args, **kwargs)
        text_console.print(*args, **kwargs)

    print_both(Panel.fit(
        "[bold]Phase 1 分析报告 | Phase 1 Analysis Report[/bold]",
        border_style="blue",
    ))

    data = load_experiment_data(Path(experiment_data_dir))
    total = sum(len(v) for v in data.values())
    print_both(f"\n加载数据 | Loaded data: {total} records from {len(data)} files")
    for eid, recs in sorted(data.items()):
        print_both(f"  phase1_{eid}.jsonl — {len(recs)} records")

    all_groups: list[ConditionGroup] = []

    for eid in sorted(data.keys()):
        records = data[eid]
        if not records:
            continue
        print_both(f"\n{'='*80}")
        desc = EXPERIMENT_INFO.get(eid, eid)
        print_both(f"[bold]实验 {eid}: {desc}[/bold]")
        print_both(f"{'='*80}")

        groups = group_by_condition(records)
        all_groups.extend(groups)
        print_both(f"分组数 | Condition groups: {len(groups)}")

        metrics_list = [_compute_group_metrics(g) for g in groups]
        _print_metrics_table(print_both.__self__ if hasattr(print_both, '__self__') else console, metrics_list)
        text_only_table = Table(title="核心指标 | Core Metrics", show_lines=True)
        text_only_table.add_column("Condition")
        text_only_table.add_column("n", justify="right")
        text_only_table.add_column("MCR", justify="right")
        text_only_table.add_column("H(bits)", justify="right")
        text_only_table.add_column("H_norm", justify="right")
        text_only_table.add_column("Endpoint", justify="right")
        text_only_table.add_column("Middle", justify="right")
        text_only_table.add_column("MeanPos", justify="right")
        text_only_table.add_column("Mode", justify="right")
        text_only_table.add_column("chi2", justify="right")
        text_only_table.add_column("p", justify="right")
        for m in metrics_list:
            text_only_table.add_row(
                m["label"], str(m["n_recs"]),
                f"{m['mcr']:.3f}", f"{m['entropy']:.3f}", f"{m['norm_entropy']:.3f}",
                f"{m['endpoint_pref']:.3f}", f"{m['middle_pref']:.3f}", f"{m['mean_rel_pos']:.3f}",
                str(m["mode_choice"]), f"{m['chi2_stat']:.2f}", f"{m['chi2_p']:.4f}",
            )
        text_console.print(text_only_table)

        jsd_results = _compute_cross_template_jsd(groups)
        _print_jsd_table(console, jsd_results)
        _print_jsd_table(text_console, jsd_results)

        chi2_results = _compute_chi2_independence(groups)
        _print_chi2_independence(console, chi2_results)
        _print_chi2_independence(text_console, chi2_results)

    print_both(f"\n{'='*80}")
    print_both("[bold]生成可视化 | Generating visualizations...[/bold]")
    print_both(f"{'='*80}")

    plot_choice_heatmap(all_groups, figures_dir / "choice_heatmap.png")
    print_both("  [green]✓[/] choice_heatmap.png")

    plot_mcr_comparison(all_groups, figures_dir / "mcr_comparison.png")
    print_both("  [green]✓[/] mcr_comparison.png")

    plot_choice_distribution(all_groups, figures_dir / "choice_distribution.png")
    print_both("  [green]✓[/] choice_distribution.png")

    plot_jsd_matrix(all_groups, figures_dir / "jsd_matrix.png")
    print_both("  [green]✓[/] jsd_matrix.png")

    plot_entropy_comparison(all_groups, figures_dir / "entropy_comparison.png")
    print_both("  [green]✓[/] entropy_comparison.png")

    plot_temperature_comparison(all_groups, figures_dir / "temperature_comparison.png")
    print_both("  [green]✓[/] temperature_comparison.png")

    print_both(f"\n[bold green]报告完成 | Report complete.[/bold green]")
    print_both(f"图表保存在 | Figures saved to: {figures_dir}")

    report_text = buf.getvalue()
    report_path = output_dir / "phase1_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    console.print(f"文本报告已保存 | Text report saved: {report_path}")
