# SPDX-License-Identifier: Apache-2.0
"""Prompt builder / 提示词构建器

构建用于 LLM 厕所隔间选择实验的提示词。
Builds prompts for the LLM toilet-stall decision experiment.
"""

from __future__ import annotations

from typing import Any

# 默认系统消息模板 / Default system message template
_DEFAULT_SYSTEM_MESSAGE_TEMPLATE = (
    "你是一个正在选择公共厕所隔间的人。请用 JSON 格式回复。\n"
    "You are choosing a toilet stall. Respond in JSON format.\n\n"
    "有效坑位范围 / Valid stall range: 1 to {num_stalls}\n"
    "回复格式 / Response format:\n"
    '{{"chosen_stall": <int>, "chain_of_thought": "<str>", "confidence": <float>}}'
)

# 默认方向反转替换对 / Default direction reversal pairs
_DEFAULT_REVERSAL_PAIRS: list[dict[str, str]] = [
    {"source": "从左到右", "target": "从右到左"},
    {"source": "from left to right", "target": "from right to left"},
]


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


def build_system_message(
    num_stalls: int,
    template: str = _DEFAULT_SYSTEM_MESSAGE_TEMPLATE,
) -> str:
    """返回系统消息，指示模型以结构化 JSON 格式回复 / Return a system message for structured JSON output.

    Args:
        num_stalls: 坑位总数。
        template: 系统消息模板，包含 ``{num_stalls}`` 占位符。
            默认使用模块内置模板。

    Returns:
        完成替换后的系统提示词字符串。
    """
    return template.format_map({"num_stalls": num_stalls})


def build_reverse_prompt(
    template_text: str,
    num_stalls: int,
    reversal_pairs: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> str:
    """构建方向反转的提示词（用于对称性测试）/ Build a direction-reversed prompt for symmetry tests.

    按顺序将 reversal_pairs 中的 source 替换为 target，然后调用
    :func:`build_prompt` 完成其余占位符替换。

    Args:
        template_text: 原始模板字符串。
        num_stalls: 坑位总数。
        reversal_pairs: 方向替换对列表 ``[{"source": "...", "target": "..."}]``。
            默认使用模块内置的中英文替换对。
        **kwargs: 额外占位符键值对。

    Returns:
        方向反转后的提示词字符串。
    """
    pairs = reversal_pairs if reversal_pairs is not None else _DEFAULT_REVERSAL_PAIRS
    reversed_text = template_text
    for pair in pairs:
        reversed_text = reversed_text.replace(pair["source"], pair["target"])
    return build_prompt(reversed_text, num_stalls, **kwargs)
