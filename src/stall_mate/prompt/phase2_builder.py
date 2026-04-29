# SPDX-License-Identifier: Apache-2.0
"""
条件叠加实验提示词模板 | Phase 2 conditional experiment prompt templates.

Phase 2 在 Phase 1 基础上叠加占用、清洁度、时间压力、社交情境、物理差异等条件。
模板使用 {num_stalls} 和 {conditions_text} 占位符。
Phase 2 layers occupancy, cleanliness, time pressure, social context, and physical
differences on top of the Phase 1 baseline. Templates use {num_stalls} and
{conditions_text} placeholders.
"""

from __future__ import annotations

from typing import Any

from stall_mate.prompt.builder import build_prompt, build_system_message


def build_conditions_text(conditions: dict[str, Any]) -> str:
    """将条件字典转换为自然语言描述 / Convert a conditions dict to a natural-language description.

    Args:
        conditions: 包含以下可能的键：
            - ``conditions_description`` (str): 完整的条件描述文本。
            - ``occupied`` (list[int]): 被占用的坑位编号列表。
            - ``cleanliness`` (str): 清洁度相关描述。
            - ``time_pressure`` (str): 时间压力描述。
            - ``social`` (str): 社交情境描述。
            - ``physical`` (str): 物理环境差异描述。

    Returns:
        拼接后的条件描述字符串；无有效条件时返回空字符串。
    """
    # 如果已有完整描述，直接使用
    if desc := conditions.get("conditions_description"):
        return str(desc)

    parts: list[str] = []

    # 占用状态
    occupied = conditions.get("occupied", [])
    if isinstance(occupied, list) and occupied:
        nums = "、".join(str(s) for s in sorted(occupied))
        parts.append(f"其中第{nums}号坑位有人正在使用")

    # 清洁度
    if cleanliness := conditions.get("cleanliness"):
        parts.append(str(cleanliness))

    # 时间压力
    if time_pressure := conditions.get("time_pressure"):
        parts.append(str(time_pressure))

    # 社交情境
    if social := conditions.get("social"):
        parts.append(str(social))

    # 物理差异
    if physical := conditions.get("physical"):
        parts.append(str(physical))

    if not parts:
        return ""
    text = "。".join(parts)
    if not text.endswith("。"):
        text += "。"
    return text


def build_phase2_prompt(
    template_text: str,
    num_stalls: int,
    conditions: dict[str, Any],
    **kwargs: Any,
) -> str:
    """构建 Phase 2 提示词 / Build a Phase 2 prompt with condition injection.

    模板中必须包含 ``{num_stalls}`` 和 ``{conditions_text}`` 占位符。
    ``{conditions_text}`` 会被替换为 :func:`build_conditions_text` 的输出。

    Args:
        template_text: 包含 ``{num_stalls}`` 和 ``{conditions_text}`` 的模板。
        num_stalls: 坑位总数。
        conditions: 条件字典（见 :func:`build_conditions_text`）。
        **kwargs: 额外占位符键值对。

    Returns:
        完成替换后的提示词字符串。
    """
    conditions_text = build_conditions_text(conditions)
    params: dict[str, Any] = {
        "num_stalls": num_stalls,
        "conditions_text": conditions_text,
        **kwargs,
    }
    return template_text.format_map(params)
