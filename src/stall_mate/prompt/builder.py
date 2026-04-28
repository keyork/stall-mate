# SPDX-License-Identifier: Apache-2.0
"""Prompt builder / 提示词构建器

构建用于 LLM 厕所隔间选择实验的提示词。
Builds prompts for the LLM toilet-stall decision experiment.
"""

from __future__ import annotations

from typing import Any


def build_prompt(template_text: str, num_stalls: int, **kwargs: Any) -> str:
    """将模板中的占位符替换为实际值 / Substitute placeholders in a prompt template.

    Args:
        template_text: 包含 ``{num_stalls}`` 等占位符的模板字符串。
        num_stalls: 坑位总数。
        **kwargs: 额外占位符键值对。

    Returns:
        完成替换后的提示词字符串。

    Raises:
        KeyError: 模板中存在未提供的占位符。
    """
    params: dict[str, Any] = {"num_stalls": num_stalls, **kwargs}
    return template_text.format_map(params)


def build_system_message(num_stalls: int) -> str:
    """返回系统消息，指示模型以结构化 JSON 格式回复 / Return a system message for structured JSON output.

    Args:
        num_stalls: 坑位总数。

    Returns:
        双语系统提示词字符串。
    """
    return (
        "你是一个正在选择公共厕所隔间的人。请用 JSON 格式回复。\n"
        "You are choosing a toilet stall. Respond in JSON format.\n\n"
        f"有效坑位范围 / Valid stall range: 1 to {num_stalls}\n"
        "回复格式 / Response format:\n"
        '{"chosen_stall": <int>, "chain_of_thought": "<str>", "confidence": <float>}'
    )


def build_reverse_prompt(
    template_text: str, num_stalls: int, **kwargs: Any
) -> str:
    """构建方向反转的提示词（用于对称性测试）/ Build a direction-reversed prompt for symmetry tests.

    将模板中的"从左到右"替换为"从右到左"（中英文），然后调用
    :func:`build_prompt` 完成其余占位符替换。

    Args:
        template_text: 原始模板字符串。
        num_stalls: 坑位总数。
        **kwargs: 额外占位符键值对。

    Returns:
        方向反转后的提示词字符串。
    """
    reversed_text = (
        template_text
        .replace("从左到右", "从右到左")
        .replace("from left to right", "from right to left")
    )
    return build_prompt(reversed_text, num_stalls, **kwargs)
