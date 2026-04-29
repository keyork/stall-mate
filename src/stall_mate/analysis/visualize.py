# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from stall_mate.analysis.loader import ConditionGroup, choice_distribution
from stall_mate.analysis.metrics import (
    jsd_between_distributions,
    mcr,
    normalized_entropy,
)

matplotlib.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

PALETTE = "Set2"


def _save(fig: plt.Figure, output_path: Path) -> plt.Figure:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_choice_heatmap(
    groups: list[ConditionGroup], output_path: Path
) -> plt.Figure:
    if not groups:
        return plt.figure()
    labels = [g.label for g in groups]
    max_stalls = max(g.num_stalls for g in groups)
    matrix = np.zeros((len(groups), max_stalls))
    for i, g in enumerate(groups):
        dist = choice_distribution(g.choices, g.num_stalls)
        total = dist.sum()
        matrix[i, : g.num_stalls] = dist / total if total > 0 else 0

    fig, ax = plt.subplots(figsize=(max(max_stalls * 0.8, 6), max(len(groups) * 0.45, 4)))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=[str(j + 1) for j in range(max_stalls)],
        yticklabels=labels,
        cbar_kws={"label": "选择频率 | Choice Frequency"},
    )
    ax.set_xlabel("坑位编号 | Stall Position")
    ax.set_ylabel("实验条件 | Condition")
    ax.set_title("选择分布热力图 | Choice Distribution Heatmap")
    fig.tight_layout()
    return _save(fig, output_path)


def plot_mcr_comparison(
    groups: list[ConditionGroup], output_path: Path
) -> plt.Figure:
    if not groups:
        return plt.figure()
    temps = sorted({g.temperature for g in groups})
    fig, axes = plt.subplots(1, len(temps), figsize=(6 * len(temps), 5), squeeze=False)
    for idx, temp in enumerate(temps):
        ax = axes[0, idx]
        sub = [g for g in groups if g.temperature == temp]
        sub.sort(key=lambda g: (g.num_stalls, g.template))
        x_labels = [f"N={g.num_stalls}\n{g.template}" for g in sub]
        values = [mcr(g.choices) for g in sub]
        colors = sns.color_palette(PALETTE, len(sub))
        ax.bar(range(len(sub)), values, color=colors)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("MCR")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"温度 T={temp} | Temperature T={temp}")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    fig.suptitle("MCR 对比 | Mode Consistency Rate Comparison", fontsize=13)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_choice_distribution(
    groups: list[ConditionGroup], output_path: Path
) -> plt.Figure:
    if not groups:
        return plt.figure()
    n_groups = len(groups)
    ncols = min(4, n_groups)
    nrows = (n_groups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    colors = sns.color_palette(PALETTE, n_groups)
    for idx, g in enumerate(groups):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        dist = choice_distribution(g.choices, g.num_stalls)
        total = dist.sum()
        freq = dist / total if total > 0 else dist
        ax.bar(range(1, g.num_stalls + 1), freq, color=colors[idx])
        ax.set_xlabel("坑位 | Stall")
        ax.set_ylabel("频率 | Freq")
        ax.set_title(g.label, fontsize=7)
        ax.set_ylim(0, 1.0)
    for idx in range(n_groups, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)
    fig.suptitle("选择频率分布 | Choice Frequency Distribution", fontsize=13)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_jsd_matrix(
    groups: list[ConditionGroup], output_path: Path
) -> plt.Figure:
    if not groups:
        return plt.figure()
    from collections import defaultdict

    nt_groups: dict[tuple, dict[str, ConditionGroup]] = defaultdict(dict)
    for g in groups:
        nt_groups[(g.experiment_group, g.num_stalls, g.temperature)][g.template] = g

    valid_keys = [k for k, v in nt_groups.items() if len(v) >= 2]
    if not valid_keys:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No pairs to compare", ha="center", va="center")
        return _save(fig, output_path)

    templates = sorted({t for v in nt_groups.values() for t in v})
    pairs = [(i, j) for i in range(len(templates)) for j in range(i + 1, len(templates))]

    nrows = max(1, len(valid_keys))
    ncols = max(1, len(pairs))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for r, key in enumerate(sorted(valid_keys)):
        tmap = nt_groups[key]
        for c, (ti, tj) in enumerate(pairs):
            ax = axes[r][c]
            ta, tb = templates[ti], templates[tj]
            if ta in tmap and tb in tmap:
                da = choice_distribution(tmap[ta].choices, tmap[ta].num_stalls)
                db = choice_distribution(tmap[tb].choices, tmap[tb].num_stalls)
                val = jsd_between_distributions(da, db)
                ax.bar([0], [val], color=sns.color_palette(PALETTE, 1))
                ax.set_xticks([0])
                ax.set_xticklabels([f"{ta} vs {tb}"])
                ax.set_ylim(0, 1)
                ax.set_title(f"G={key[0]} N={key[1]} T={key[2]}\nJSD={val:.4f}", fontsize=8)
            else:
                ax.set_visible(False)

    fig.suptitle("JSD 跨模板对比 | Cross-Template JSD", fontsize=13)
    fig.tight_layout()
    return _save(fig, output_path)


def plot_entropy_comparison(
    groups: list[ConditionGroup], output_path: Path
) -> plt.Figure:
    if not groups:
        return plt.figure()
    fig, ax = plt.subplots(figsize=(max(len(groups) * 0.5, 8), 5))
    x_labels = [g.label for g in groups]
    values = [normalized_entropy(g.choices, g.num_stalls) for g in groups]
    colors = sns.color_palette(PALETTE, len(groups))
    ax.bar(range(len(groups)), values, color=colors)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(x_labels, fontsize=6, rotation=90)
    ax.set_ylabel("归一化熵 | Normalized Entropy")
    ax.set_ylim(0, 1.05)
    ax.set_title("归一化熵对比 | Normalized Entropy Comparison")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="均匀分布 | Uniform")
    ax.legend()
    fig.tight_layout()
    return _save(fig, output_path)


def plot_temperature_comparison(
    groups: list[ConditionGroup], output_path: Path
) -> plt.Figure:
    if not groups:
        return plt.figure()
    t0 = [g for g in groups if g.temperature == 0.0]
    t07 = [g for g in groups if g.temperature == 0.7]

    paired: list[tuple[ConditionGroup, ConditionGroup]] = []
    for g0 in sorted(t0, key=lambda g: (g.experiment_group, g.num_stalls, g.template)):
        for g07 in t07:
            if (
                g0.experiment_group == g07.experiment_group
                and g0.num_stalls == g07.num_stalls
                and g0.template == g07.template
            ):
                paired.append((g0, g07))
                break

    if not paired:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No T=0 / T=0.7 pairs", ha="center", va="center")
        return _save(fig, output_path)

    x_labels = [f"G={a.experiment_group} N={a.num_stalls} {a.template}" for a, _ in paired]
    mcr_t0 = [mcr(a.choices) for a, _ in paired]
    mcr_t07 = [mcr(b.choices) for _, b in paired]

    x = np.arange(len(paired))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(len(paired) * 0.6, 8), 5))
    ax.bar(x - width / 2, mcr_t0, width, label="T=0.0", color=sns.color_palette(PALETTE, 2)[0])
    ax.bar(x + width / 2, mcr_t07, width, label="T=0.7", color=sns.color_palette(PALETTE, 2)[1])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=6, rotation=90)
    ax.set_ylabel("MCR")
    ax.set_ylim(0, 1.05)
    ax.set_title("温度对比 MCR | Temperature MCR Comparison")
    ax.legend()
    fig.tight_layout()
    return _save(fig, output_path)
