# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date
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
    "1.1": "基准选择 | Baseline Choice",
    "1.2": "对称性检验 | Symmetry Check",
    "1.3": "选项数量效应 | Option Count Effect",
}

EXPERIMENT_DESC: dict[str, str] = {
    "1.1": "N=3,5,7,10 × T=0.0,0.7 × templates=A,B,C × 30 reps",
    "1.2": "N=5 × T=0.0,0.7 × templates=A,B,C × 30 reps",
    "1.3": "N=3,5,7,10,15,20 × T=0.0,0.7 × templates=A,B,C × 30 reps",
}


class MarkdownBuilder:
    def __init__(self) -> None:
        self._lines: list[str] = []

    def h1(self, text: str) -> None:
        self._lines.append(f"# {text}\n")

    def h2(self, text: str) -> None:
        self._lines.append(f"\n## {text}\n")

    def h3(self, text: str) -> None:
        self._lines.append(f"\n### {text}\n")

    def p(self, text: str) -> None:
        self._lines.append(f"{text}\n")

    def blockquote(self, text: str) -> None:
        self._lines.append(f"> {text}\n")

    def hr(self) -> None:
        self._lines.append("\n---\n")

    def blank(self) -> None:
        self._lines.append("")

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "|" + "|".join(["------" for _ in headers]) + "|"
        self._lines.append(header_line)
        self._lines.append(sep_line)
        for row in rows:
            self._lines.append("| " + " | ".join(str(c) for c in row) + " |")
        self._lines.append("")

    def ol(self, items: list[str]) -> None:
        for i, item in enumerate(items, 1):
            self._lines.append(f"{i}. {item}")
        self._lines.append("")

    def build(self) -> str:
        return "\n".join(self._lines)


def _compute_group_metrics(group: ConditionGroup) -> dict:
    ch = group.choices
    ns = group.num_stalls
    freqs = choice_frequencies(ch, ns)
    mode_choice = max(freqs, key=lambda k: freqs[k]) if freqs else None
    mean_pos = np.mean(relative_position(ch, ns)) if ch else 0.0
    return {
        "label": group.label,
        "experiment_group": group.experiment_group,
        "num_stalls": ns,
        "temperature": group.temperature,
        "template": group.template,
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


def _print_metrics_table_rich(console: Console, metrics_list: list[dict]) -> None:
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


def _print_jsd_table_rich(console: Console, jsd_results: list[dict]) -> None:
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


def _print_chi2_independence_rich(console: Console, results: list[dict]) -> None:
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


def _write_metrics_table_md(md: MarkdownBuilder, metrics_list: list[dict]) -> None:
    headers = ["条件", "n", "MCR", "H(bits)", "H_norm", "端点偏好", "中间偏好", "均值位置", "众数", "χ²", "p"]
    rows = []
    for m in metrics_list:
        sig = "*" if m["chi2_p"] < 0.05 else ""
        rows.append([
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
        ])
    md.table(headers, rows)


def _write_jsd_table_md(md: MarkdownBuilder, jsd_results: list[dict]) -> None:
    if not jsd_results:
        md.p("*No cross-template pairs to compare.*")
        return
    headers = ["条件", "模板对", "JSD"]
    rows = [[r["condition"], r["pair"], f"{r['jsd']:.4f}"] for r in jsd_results]
    md.table(headers, rows)


def _write_chi2_table_md(md: MarkdownBuilder, results: list[dict]) -> None:
    if not results:
        md.p("*No pairs to test.*")
        return
    headers = ["条件", "模板对", "χ²", "p", "显著"]
    rows = []
    for r in results:
        sig_mark = "✓" if r["sig"] else "—"
        rows.append([
            r["condition"],
            r["pair"],
            f"{r['chi2']:.2f}",
            f"{r['p']:.4f}",
            sig_mark,
        ])
    md.table(headers, rows)


def _write_key_findings(
    md: MarkdownBuilder,
    all_metrics: list[dict],
    all_jsd: list[dict],
    all_chi2: list[dict],
) -> None:
    md.h2("关键发现 | Key Findings")

    findings: list[str] = []

    by_mid = sorted(all_metrics, key=lambda m: m["middle_pref"], reverse=True)
    top_mid = by_mid[:min(5, len(by_mid))]
    overall_mid = statistics.mean(m["middle_pref"] for m in all_metrics) if all_metrics else 0.0
    mid_lines = [f"**中间位置偏好 | Middle Position Preference**: "
                 f"Overall average middle preference = **{overall_mid:.3f}**."]
    if top_mid:
        top_desc = ", ".join(f"{m['label']} ({m['middle_pref']:.3f})" for m in top_mid[:3])
        mid_lines.append(f"Strongest middle preference: {top_desc}.")
    findings.append(" ".join(mid_lines))

    if all_jsd:
        max_jsd = max(all_jsd, key=lambda r: r["jsd"])
        avg_jsd = statistics.mean(r["jsd"] for r in all_jsd)
        sig_chi2 = [r for r in all_chi2 if r["sig"]]
        tmpl_lines = [f"**模板敏感性 | Template Sensitivity**: "
                      f"Average cross-template JSD = **{avg_jsd:.4f}**."]
        tmpl_lines.append(f"Highest JSD: {max_jsd['condition']} {max_jsd['pair']} = **{max_jsd['jsd']:.4f}**.")
        if sig_chi2:
            tmpl_lines.append(f"{len(sig_chi2)}/{len(all_chi2)} template pairs show significant χ² independence (p<0.05).")
        else:
            tmpl_lines.append("No template pairs show significant χ² independence.")
        findings.append(" ".join(tmpl_lines))
    else:
        findings.append("**模板敏感性 | Template Sensitivity**: No cross-template data available.")

    t0_mcrs = [m["mcr"] for m in all_metrics if m["temperature"] == 0.0]
    t07_mcrs = [m["mcr"] for m in all_metrics if m["temperature"] > 0.0]
    if t0_mcrs and t07_mcrs:
        avg_t0 = statistics.mean(t0_mcrs)
        avg_t07 = statistics.mean(t07_mcrs)
        diff = avg_t0 - avg_t07
        direction = "higher" if diff > 0 else "lower"
        findings.append(
            f"**温度效应 | Temperature Effect**: "
            f"Average MCR at T=0.0 = **{avg_t0:.3f}**, at T=0.7 = **{avg_t07:.3f}** "
            f"(Δ = {diff:+.3f}, T=0.0 is {direction})."
        )
    else:
        findings.append("**温度效应 | Temperature Effect**: Insufficient temperature variation in data.")

    by_n: dict[int, list[float]] = defaultdict(list)
    for m in all_metrics:
        by_n[m["num_stalls"]].append(m["mcr"])
    if len(by_n) > 1:
        n_trend = sorted(by_n.items())
        trend_parts = [f"N={n}: MCR={statistics.mean(v):.3f}" for n, v in n_trend]
        trend_line = " → ".join(trend_parts)
        first_avg = statistics.mean(n_trend[0][1])
        last_avg = statistics.mean(n_trend[-1][1])
        direction = "increasing" if last_avg > first_avg else "decreasing"
        findings.append(
            f"**选项数量效应 | Option Count Effect**: MCR trend as N increases: {trend_line}. "
            f"Overall trend is **{direction}**."
        )
    else:
        findings.append("**选项数量效应 | Option Count Effect**: Only one N value in data, cannot assess trend.")

    if all_metrics:
        all_mcr_vals = [m["mcr"] for m in all_metrics]
        avg_mcr = statistics.mean(all_mcr_vals)
        best = min(all_metrics, key=lambda m: abs(m["mcr"] - 1.0))
        worst = min(all_metrics, key=lambda m: m["mcr"])
        findings.append(
            f"**整体一致性 | Overall Consistency**: "
            f"Average MCR across all conditions = **{avg_mcr:.3f}**. "
            f"Most consistent: {best['label']} (MCR={best['mcr']:.3f}). "
            f"Least consistent: {worst['label']} (MCR={worst['mcr']:.3f})."
        )

    md.ol(findings)


def generate_phase1_report(
    experiment_data_dir: Path,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    md = MarkdownBuilder()

    console.print(Panel.fit(
        "[bold]Phase 1 分析报告 | Phase 1 Analysis Report[/bold]",
        border_style="blue",
    ))

    data = load_experiment_data(Path(experiment_data_dir))
    total = sum(len(v) for v in data.values())
    console.print(f"\n加载数据 | Loaded data: {total} records from {len(data)} files")
    for eid, recs in sorted(data.items()):
        console.print(f"  phase1_{eid}.jsonl — {len(recs)} records")

    model_name = "unknown"
    all_groups: list[ConditionGroup] = []
    all_metrics: list[dict] = []
    all_jsd: list[dict] = []
    all_chi2: list[dict] = []

    for eid, recs in sorted(data.items()):
        for r in recs:
            model_name = r.model_name
            break

    md.h1("📊 Phase 1 分析报告 | Phase 1 Analysis Report")
    md.blank()
    md.blockquote(f"实验日期 | Experiment date: {date.today().isoformat()}")
    md.blockquote(f"模型 | Model: {model_name}")
    md.blockquote(f"总记录数 | Total records: {total}")
    md.hr()

    for eid in sorted(data.keys()):
        records = data[eid]
        if not records:
            continue
        console.print(f"\n{'='*80}")
        desc = EXPERIMENT_INFO.get(eid, eid)
        full_desc = EXPERIMENT_DESC.get(eid, "")
        console.print(f"[bold]实验 {eid}: {desc}[/bold]")
        console.print(f"{'='*80}")

        groups = group_by_condition(records)
        all_groups.extend(groups)
        console.print(f"分组数 | Condition groups: {len(groups)}")

        metrics_list = [_compute_group_metrics(g) for g in groups]
        all_metrics.extend(metrics_list)

        _print_metrics_table_rich(console, metrics_list)

        jsd_results = _compute_cross_template_jsd(groups)
        all_jsd.extend(jsd_results)
        _print_jsd_table_rich(console, jsd_results)

        chi2_results = _compute_chi2_independence(groups)
        all_chi2.extend(chi2_results)
        _print_chi2_independence_rich(console, chi2_results)

        md.h2(f"实验 {eid}: {desc}")
        md.p(f"**设计**: {full_desc}")
        md.p(f"**分组数 | Condition groups**: {len(groups)}")

        md.h3("核心指标 | Core Metrics")
        _write_metrics_table_md(md, metrics_list)

        md.h3("跨模板 JSD | Cross-Template JSD")
        _write_jsd_table_md(md, jsd_results)

        md.h3("卡方独立性检验 | χ² Independence Test")
        _write_chi2_table_md(md, chi2_results)

        md.hr()

    console.print(f"\n{'='*80}")
    console.print("[bold]生成可视化 | Generating visualizations...[/bold]")
    console.print(f"{'='*80}")

    plot_choice_heatmap(all_groups, figures_dir / "choice_heatmap.png")
    console.print("  [green]✓[/] choice_heatmap.png")

    plot_mcr_comparison(all_groups, figures_dir / "mcr_comparison.png")
    console.print("  [green]✓[/] mcr_comparison.png")

    plot_choice_distribution(all_groups, figures_dir / "choice_distribution.png")
    console.print("  [green]✓[/] choice_distribution.png")

    plot_jsd_matrix(all_groups, figures_dir / "jsd_matrix.png")
    console.print("  [green]✓[/] jsd_matrix.png")

    plot_entropy_comparison(all_groups, figures_dir / "entropy_comparison.png")
    console.print("  [green]✓[/] entropy_comparison.png")

    plot_temperature_comparison(all_groups, figures_dir / "temperature_comparison.png")
    console.print("  [green]✓[/] temperature_comparison.png")

    md.h2("可视化图表 | Visualizations")
    md.table(
        ["图表", "文件"],
        [
            ["选择分布热力图 | Choice Heatmap", "![heatmap](figures/choice_heatmap.png)"],
            ["MCR 对比 | MCR Comparison", "![mcr](figures/mcr_comparison.png)"],
            ["选择频率分布 | Choice Distribution", "![dist](figures/choice_distribution.png)"],
            ["JSD 矩阵 | JSD Matrix", "![jsd](figures/jsd_matrix.png)"],
            ["归一化熵对比 | Entropy Comparison", "![entropy](figures/entropy_comparison.png)"],
            ["温度对比 | Temperature Comparison", "![temp](figures/temperature_comparison.png)"],
        ],
    )
    md.hr()

    _write_key_findings(md, all_metrics, all_jsd, all_chi2)

    report_path = output_dir / "phase1_report.md"
    report_path.write_text(md.build(), encoding="utf-8")

    console.print(f"\n[bold green]报告完成 | Report complete.[/bold green]")
    console.print(f"图表保存在 | Figures saved to: {figures_dir}")
    console.print(f"Markdown 报告已保存 | Markdown report saved: {report_path}")
