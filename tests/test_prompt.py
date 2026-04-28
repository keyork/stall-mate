# SPDX-License-Identifier: Apache-2.0
"""提示词构建器测试 | Tests for stall_mate.prompt."""

import pytest

from stall_mate.prompt import (
    build_prompt,
    build_reverse_prompt,
    build_system_message,
)

TEMPLATE_A = "一排{num_stalls}个坑位，全部空闲。你选第几个？只回答数字。"
TEMPLATE_B = (
    "你走进一间公共厕所，面前有一排{num_stalls}个独立的坑位，"
    "从左到右依次编号1到{num_stalls}。"
    "现在所有坑位都没有人使用。你会选择哪一个坑位？请直接回答你的选择。"
)
TEMPLATE_D = "面前有一排{num_stalls}个坑位，全部空闲。你可以选最左边的、最右边的、中间的、或者靠左/靠右的某一个。你怎么选？"


class TestBuildPrompt:
    def test_template_a_substitutes_num_stalls(self):
        result = build_prompt(TEMPLATE_A, 5)
        assert "一排5个坑位" in result

    def test_template_b_substitutes_num_stalls(self):
        result = build_prompt(TEMPLATE_B, 3)
        assert "1到3" in result

    def test_template_d_substitutes_num_stalls(self):
        result = build_prompt(TEMPLATE_D, 10)
        assert "10个坑位" in result

    def test_no_remaining_placeholders(self):
        result = build_prompt(TEMPLATE_A, 7)
        assert "{" not in result
        assert "}" not in result

    def test_missing_placeholder_raises_key_error(self):
        template = "Hello {name}, there are {num_stalls} stalls."
        with pytest.raises(KeyError):
            build_prompt(template, 5)

    def test_extra_kwargs_substituted(self):
        template = "Pick stall {num_stalls} with level {level}."
        result = build_prompt(template, 3, level="high")
        assert "{level}" not in result
        assert "high" in result


class TestBuildSystemMessage:
    def test_contains_range_and_fields(self):
        msg = build_system_message(5)
        assert "1 to 5" in msg
        assert "chosen_stall" in msg
        assert "chain_of_thought" in msg

    def test_different_num_stalls(self):
        msg = build_system_message(3)
        assert "1 to 3" in msg

    def test_custom_template(self):
        template = "Pick a stall from 1 to {num_stalls}. Reply JSON."
        msg = build_system_message(7, template=template)
        assert "1 to 7" in msg
        assert "Reply JSON" in msg


class TestBuildReversePrompt:
    def test_reverses_direction(self):
        result = build_reverse_prompt(TEMPLATE_B, 3)
        assert "从右到左" in result
        assert "从左到右" not in result

    def test_english_direction_reversed(self):
        template = "Stalls from left to right, pick one of {num_stalls}."
        result = build_reverse_prompt(template, 4)
        assert "from right to left" in result
        assert "from left to right" not in result

    def test_custom_reversal_pairs(self):
        pairs = [{"source": "left", "target": "right"}]
        template = "Pick from {num_stalls} stalls, left side preferred."
        result = build_reverse_prompt(template, 5, reversal_pairs=pairs)
        assert "right side" in result
        assert "left side" not in result
