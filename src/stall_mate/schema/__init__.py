# SPDX-License-Identifier: Apache-2.0
"""LLM 响应模式 | LLM response schema for structured output."""

from stall_mate.schema.stall_choice import (
    StallChoice,
    get_stallchoice_json_schema,
)

__all__ = [
    "StallChoice",
    "get_stallchoice_json_schema",
]
