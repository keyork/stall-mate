# SPDX-License-Identifier: Apache-2.0
"""
LLM 响应模式定义 | LLM response schema definitions.

定义大模型选择坑位的结构化输出格式。
Defines the structured output format for LLM stall-choice responses.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class StallChoice(BaseModel):
    """模型对坑位选择的响应 / Model response for a stall choice.

    Attributes:
        chosen_stall: 你选择的坑位编号 / The stall number you chose.
        chain_of_thought: 你的思考过程 / Your step-by-step reasoning.
        confidence: 信心程度 0-1 / Confidence level 0-1.
    """

    chosen_stall: int = Field(description="你选择的坑位编号 / The stall number you chose")
    chain_of_thought: str = Field(
        description="你的思考过程 / Your step-by-step reasoning",
        min_length=10,
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="信心程度 0-1 / Confidence level 0-1",
    )

    @field_validator("chosen_stall")
    @classmethod
    def validate_stall_range(cls, v: int, info: ValidationInfo) -> int:
        """验证坑位编号是否在有效范围内 / Validate stall number is within range."""
        if info.context and "num_stalls" in info.context:
            n = info.context["num_stalls"]
            if v < 1 or v > n:
                raise ValueError(f"Stall {v} out of range [1, {n}]")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chosen_stall": 3,
                "chain_of_thought": "中间位置最安全",
                "confidence": 0.8,
            }
        }
    )


def get_stallchoice_json_schema() -> dict:
    """返回 StallChoice 的 JSON Schema / Return the JSON Schema for StallChoice."""
    return StallChoice.model_json_schema()
